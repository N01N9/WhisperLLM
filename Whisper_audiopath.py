from transformers import WhisperProcessor, WhisperForConditionalGeneration
from torch import Tensor
import numpy as np
import torch
import librosa

class CustomDecodingResult:
    """Class to hold all hypotheses from beam search."""
    def __init__(self, audio_features, language, tokens, texts, avg_logprob, no_speech_prob, temperature, compression_ratio):
        self.audio_features = audio_features
        self.language = language
        self.tokens = tokens
        self.texts = texts
        self.avg_logprob = avg_logprob
        self.no_speech_prob = no_speech_prob
        self.temperature = temperature
        self.compression_ratio = compression_ratio

    @staticmethod
    def from_transformer_output(audio_features, languages, tokens, texts, avg_logprobs, no_speech_probs, temperatures, compression_ratios):
        return [
            CustomDecodingResult(
                audio_features=audio_feature,
                language=language,
                tokens=token,
                texts=text,
                avg_logprob=avg_logprob,
                no_speech_prob=no_speech_prob,
                temperature=temperature,
                compression_ratio=compression_ratio,
            )
            for audio_feature, language, token, text, avg_logprob, no_speech_prob, temperature, compression_ratio in zip(
                audio_features, languages, tokens, texts, avg_logprobs, no_speech_probs, temperatures, compression_ratios
            )
        ]

class CustomReturnAllSamplesRanker:
    """Return all n-best hypotheses."""
    def rank(self, tokens, sum_logprobs):
        def scores(logprobs, lengths):
            return [logprob / length for logprob, length in zip(logprobs, lengths)]

        lengths = [[len(t) for t in s] for s in tokens]
        return [[scores(p, l) for p, l in zip(sum_logprobs, lengths)]]

def transcribe_with_nbest(
    model, 
    audio_path, 
    n_best=3, 
    verbose=True,
    sampling_rate=16000,
    **decode_options
):
    """
    Transcribe an audio file and return n-best results.

    Parameters:
        model: Whisper Model
        audio_path: Path to the audio file
        n_best: Number of top hypotheses to return
        verbose: Display progress details
        sampling_rate: Sampling rate for loading audio
        decode_options: Additional options for decoding
    Returns:
        List of dictionaries containing n-best hypotheses
    """
    processor = WhisperProcessor.from_pretrained(model.config.name_or_path)

    # Check if GPU is available
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = model.to(device)

    # Load audio file
    audio, sr = librosa.load(audio_path, sr=sampling_rate)
    input_features = processor(audio, return_tensors="pt").input_features.to(device)

    with torch.no_grad():
        audio_features = model.model.encoder(input_features).last_hidden_state

    # Generate attention mask
    attention_mask = torch.ones(input_features.shape[-1], dtype=torch.long).to(device)

    beam_size = max(n_best, decode_options.get("beam_size", 5))

    decode_options.update({
        "num_beams": beam_size, 
        "num_return_sequences": n_best,
        "temperature": 0.9,  # Introduce randomness
        "top_k": 50,         # Use top-k sampling
        "top_p": 0.95,       # Use nucleus sampling
        "do_sample": True,   # Enable sampling mode
        "task": "transcribe"  # Explicitly set task to transcription
    })

    with torch.no_grad():
        outputs = model.generate(
            input_features,
            attention_mask=attention_mask,
            **decode_options
        )

    results = processor.batch_decode(outputs, skip_special_tokens=True)
    if verbose:
        print("\n".join(f"Hypothesis {i+1}: {text}" for i, text in enumerate(results)))

    return audio_features, [{"text": result} for result in results]

# Example Usage
if __name__ == "__main__":
    model_name = "openai/whisper-large-v3-turbo"
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name).eval()

    audio_path = r"C:\Users\user\Downloads\test2_16k_mono4.wav"  # Replace with the path to your audio file
    n_best_hypotheses = transcribe_with_nbest(
        model=model,
        audio_path=audio_path,
        n_best=15,
        verbose=True
    )
    print(n_best_hypotheses)

