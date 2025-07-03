# -*- coding: utf-8 -*-
"""

@author: r_jairam
"""

import os
from datasets import Dataset, DatasetDict, Audio, ClassLabel
from transformers import AutoProcessor
from huggingface_hub import login

# Define paths
train_dir = "/train"
test_dir = "/test"
valid_dir = "/valid"

language_mapping = {
    "labelname_1": 0,
    "labelname_2": 1,
    "labelname_3": 2,
    "labelname_4": 3,
}

def create_data_dict(data_dir):
    data_dict = {"audio": [], "label": []}
    for label_name in os.listdir(data_dir):
        label_path = os.path.join(data_dir, label_name)
        if os.path.isdir(label_path):
            label = language_mapping.get(label_name)
            for fname in os.listdir(label_path):
                data_dict["audio"].append(os.path.join(label_path, fname))
                data_dict["label"].append(label)
    return data_dict

# Create datasets
train_data = Dataset.from_dict(create_data_dict(train_dir))
test_data = Dataset.from_dict(create_data_dict(test_dir))
valid_data = Dataset.from_dict(create_data_dict(valid_dir))

labels = ClassLabel(num_classes=len(language_mapping), names=list(language_mapping.keys()))
speech = DatasetDict({
    "train": train_data,
    "test": test_data,
    "valid": valid_data,
})

speech = speech.cast_column("audio", Audio(sampling_rate=16000))
speech = speech.cast_column("label", labels)

# Preprocess using Whisper processor
processor = AutoProcessor.from_pretrained("openai/whisper-large-v2")

def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt").input_features[0]
    return batch

dataset = speech.map(prepare_dataset, remove_columns=["audio"], num_proc=1)

# Upload to Hugging Face Hub
login(token="token")
dataset.push_to_hub("username/corpus_name", token="token")
