from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F

class HuggingFaceLLaMAAdapter(nn.Module):
    def __init__(self, model_name, adapter_prompt_length=10, adapter_start_layer=2):
        super().__init__()
        
        # Load base model and tokenizer from Hugging Face
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Configuration for adapter
        self.adapter_prompt_length = adapter_prompt_length
        self.adapter_start_layer = adapter_start_layer

        # Learnable adapter parameters
        self.adapter_wte = nn.Embedding(adapter_prompt_length, self.model.config.hidden_size).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.gating_factor = nn.Parameter(torch.zeros(1)).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        # Whisper integration
        self.projection_rms_key = nn.LayerNorm(1280).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.projection_key_matrix_down = nn.Parameter(torch.eye(1280, 1280 // 8)).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.projection_key_matrix_up = nn.Parameter(torch.eye(1280 // 8, 1280)).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        self.projection_rms_value = nn.LayerNorm(1280).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.projection_value_matrix_down = nn.Parameter(torch.eye(1280, 1280 // 8)).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.projection_value_matrix_up = nn.Parameter(torch.eye(1280 // 8, 1280)).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        self.query_projection = nn.Linear(self.model.config.hidden_size, 128).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    def forward(self, input_ids, attention_mask=None, audio_features=None, custom_sep_token=None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        if audio_features is not None:
            audio_features = audio_features.to(device)

        # Use custom separator token if provided
        if custom_sep_token is not None:
            sep_token_id = self.tokenizer.convert_tokens_to_ids(custom_sep_token)
        else:
            sep_token_id = None  # Default case when no custom token is provided

        # Split input_ids into separate sentences using sep_token_id
        if sep_token_id is not None:
            sep_indices = (input_ids == sep_token_id).nonzero(as_tuple=True)[1]
            if sep_indices.numel() > 0:
                split_sizes = torch.diff(torch.cat([torch.tensor([0], device=device), sep_indices, torch.tensor([input_ids.size(1)], device=device)]))
                input_ids_split = torch.split(input_ids, split_sizes.tolist(), dim=1)

                # Adjust attention mask for each split
                attention_mask_split = torch.split(attention_mask, split_sizes.tolist(), dim=1) if attention_mask is not None else [None] * len(input_ids_split)
            else:
                input_ids_split = [input_ids]  # No custom sep tokens found
                attention_mask_split = [attention_mask]
        else:
            input_ids_split = [input_ids]
            attention_mask_split = [attention_mask]

        # Generate base model outputs for each sentence separately
        hidden_states_list = []
        for split_input_ids, split_attention_mask in zip(input_ids_split, attention_mask_split):
            if split_input_ids.size(1) > 0:  # Ensure non-empty input
                outputs = self.model.base_model(
                    input_ids=split_input_ids,
                    attention_mask=split_attention_mask,
                    output_hidden_states=True,
                    return_dict=True
                )
                hidden_states_list.append(outputs.hidden_states[-1])  # Append only the last hidden state

        # Concatenate hidden states back
        if hidden_states_list:
            hidden_states = torch.cat(hidden_states_list, dim=1)
        else:
            raise ValueError("No valid input sequences found after splitting.")

        # Get LLM-specific head and dimension information
        num_heads = self.model.config.num_attention_heads
        head_dim = self.model.config.hidden_size // num_heads

        # Apply adapters starting from the configured layer
        for layer_idx in range(self.adapter_start_layer, len(hidden_states)):
            if layer_idx < self.adapter_start_layer:
                continue

            layer_hidden_state = hidden_states[layer_idx]

            # Add adapter prompt
            adapter_prompt = self.adapter_wte.weight.unsqueeze(0).repeat(layer_hidden_state.size(0), 1, 1)
            adapted_hidden_state = torch.cat([adapter_prompt, layer_hidden_state], dim=1)

            # Gating mechanism
            gated_hidden_state = self.gating_factor * adapted_hidden_state

            # Integrate Whisper features if provided
            if audio_features is not None:
                # Process Whisper keys
                whisper_key = self.projection_rms_key(audio_features)
                whisper_key = whisper_key @ self.projection_key_matrix_down
                whisper_key = F.silu(whisper_key)
                whisper_key = whisper_key @ self.projection_key_matrix_up

                # Process Whisper values
                whisper_value = self.projection_rms_value(audio_features)
                whisper_value = whisper_value @ self.projection_value_matrix_down
                whisper_value = F.silu(whisper_value)
                whisper_value = whisper_value @ self.projection_value_matrix_up

                # Padding for Whisper keys and values dynamically
                padded_keys = torch.zeros((audio_features.size(0), num_heads, 1500, head_dim), device=device)
                padded_keys[:, :20, :, :head_dim // 2] = whisper_key.view(audio_features.size(0), 20, 1500, head_dim // 2)

                padded_values = torch.zeros((audio_features.size(0), num_heads, 1500, head_dim), device=device)
                padded_values[:, :20, :, :head_dim // 2] = whisper_value.view(audio_features.size(0), 20, 1500, head_dim // 2)

                # Attention with Whisper
                query = self.query_projection(gated_hidden_state)
                attention_output = F.scaled_dot_product_attention(query, padded_keys, padded_values)

                gated_hidden_state = gated_hidden_state + attention_output

            hidden_states[layer_idx] = gated_hidden_state

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        outputs.logits = hidden_states

        return outputs

    def freeze_base_model(self):
        """Freeze all parameters except adapter-specific ones."""
        for name, param in self.named_parameters():
            if "adapter_wte" in name or "gating_factor" in name or "projection" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

# Example usage
if __name__ == "__main__":
    model_name = "facebook/opt-125m"  # Replace with desired model
    adapter_model = HuggingFaceLLaMAAdapter(model_name)
    adapter_model.freeze_base_model()

    tokenizer = adapter_model.tokenizer
    tokenizer.add_tokens(["<custom_token>"])  # Add custom token to the tokenizer

    inputs = tokenizer("Hello <custom_token> how are you?", return_tensors="pt", add_special_tokens=True).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # Optional: Provide dummy audio features for Whisper integration
    audio_features = torch.randn(1, 1280).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    custom_token = "<custom_token>"
    outputs = adapter_model(
        input_ids=inputs["input_ids"], 
        attention_mask=inputs["attention_mask"], 
        audio_features=audio_features, 
        custom_sep_token=custom_token
    )
    print("Logits Shape:", outputs.logits.shape)
    print("Logits Sample:", outputs.logits[0, :5, :5])


    # Logits에서 단어 ID 선택
    predicted_ids = torch.argmax(outputs.logits, dim=-1)

    # 토크나이저를 사용해 텍스트로 변환
    generated_text = adapter_model.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)

    print("Generated Text:", generated_text)
