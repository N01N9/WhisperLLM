from datasets import Dataset
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

dataset = Dataset.load_from_disk("processed_dataset.arrow")

print(f"Dataset loaded with {len(dataset)} entries.")
print(dataset[0])