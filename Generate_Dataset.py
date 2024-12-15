from datasets import load_dataset, Dataset, Audio
from tqdm import tqdm
from Whisper import transcribe_with_nbest
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import torch
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Whisper 모델 초기화
model_name = "openai/whisper-large-v3-turbo"
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name).eval()

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Whisper 기반 transcription 및 n-best 추출 함수 (Whisper.py 사용)
def process_audio_with_whisper(example, n_best=3):
    try:
        # 오디오 데이터와 샘플링 레이트 가져오기
        audio_info = example["audio"]  # dict containing "array" and "sampling_rate"
        audio_array = audio_info["array"]
        sampling_rate = audio_info["sampling_rate"]

        # Whisper를 사용해 transcription 수행
        audio_features, n_best_hypotheses = transcribe_with_nbest(
            model=model,
            audio_array=audio_array,
            sampling_rate=sampling_rate,
            n_best=n_best,
            verbose=False
        )
        return {
            "audio_features": audio_features.cpu().numpy(),  # Whisper가 추출한 audio features
            "n_best": [item["text"] for item in n_best_hypotheses],  # n-best transcription 리스트
            "sentence": example["sentence"]  # 원본 텍스트
        }
    except Exception as e:
        print(f"Error processing audio file: {e}")
        return None

# 1. 데이터셋 로드 및 전처리
# 최대 25000개 샘플 처리
batch_size = 250  # 배치 크기 설정
dataset = load_dataset("mozilla-foundation/common_voice_17_0", "en", split="train", streaming=True)
dataset = dataset.cast_column("audio", Audio())  # 오디오 데이터를 array 형식으로 변환

# 샘플 처리 시 인덱스를 먼저 접근해 디코딩/리샘플링 최소화
processed_data = []
batch_index = 0  # 배치 인덱스

for idx, example in tqdm(enumerate(dataset.take(25000)), desc="Processing audio files", total=25000):
    try:
        processed_example = process_audio_with_whisper(example, n_best=15)
        if processed_example is not None:
            processed_data.append(processed_example)

        # 5000개마다 저장
        if len(processed_data) >= batch_size:
            temp_dataset = Dataset.from_dict({
                "audio_features": [item["audio_features"] for item in processed_data],
                "n_best": [item["n_best"] for item in processed_data],
                "sentence": [item["sentence"] for item in processed_data],
            })
            temp_dataset.save_to_disk(f"processed_batch_en_{batch_index}.arrow")  # 디스크에 저장
            processed_data = []  # 메모리 초기화
            batch_index += 1

    except Exception as e:
        print(f"Error at index {idx}: {e}")

# 남은 데이터 저장
if processed_data:
    temp_dataset = Dataset.from_dict({
        "audio_features": [item["audio_features"] for item in processed_data],
        "n_best": [item["n_best"] for item in processed_data],
        "sentence": [item["sentence"] for item in processed_data],
    })
    temp_dataset.save_to_disk(f"processed_batch_en_{batch_index}.arrow")
    print(f"Final batch saved: processed_batch_en_{batch_index}.arrow")

print("All batches processed and saved.")
