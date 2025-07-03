# -*- coding: utf-8 -*-
"""
Feature extraction using Whisper and upload to Hugging Face Hub.
Author: r_jairam
"""

import os
import logging
from datasets import Dataset, DatasetDict, Audio, ClassLabel
from transformers import AutoProcessor
from huggingface_hub import login

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_data_dict(data_dir: str, label_map: dict) -> dict:
    """
    Create a dictionary of audio file paths and corresponding labels.

    Args:
        data_dir (str): Path to directory with subfolders as class labels.
        label_map (dict): Mapping of label names to numeric indices.

    Returns:
        dict: Dictionary with keys 'audio' and 'label'.
    """
    data = {"audio": [], "label": []}

    for label_name in os.listdir(data_dir):
        label_path = os.path.join(data_dir, label_name)
        if not os.path.isdir(label_path):
            continue

        label = label_map.get(label_name)
        if label is None:
            logger.warning(f"Label '{label_name}' not found in label_map. Skipping.")
            continue

        for fname in os.listdir(label_path):
            if fname.endswith(".wav"):
                data["audio"].append(os.path.join(label_path, fname))
                data["label"].append(label)

    logger.info(f"Collected {len(data['audio'])} audio files from {data_dir}")
    return data


def prepare_dataset(batch, processor):
    """
    Apply Whisper processor to extract input features.

    Args:
        batch (dict): Batch with 'audio' field.
        processor: Whisper processor object.

    Returns:
        dict: Updated batch with 'input_features'.
    """
    audio = batch["audio"]
    features = processor.feature_extractor(
        audio["array"],
        sampling_rate=audio["sampling_rate"],
        return_tensors="pt"
    )
    batch["input_features"] = features.input_features[0]
    return batch


def main():
    # === CONFIGURATION ===
    train_dir = "/train"
    test_dir = "/test"
    valid_dir = "/valid"

    label_map = {
        "labelname_1": 0,
        "labelname_2": 1,
        "labelname_3": 2,
        "labelname_4": 3,
    }

    model_id = "openai/whisper-large-v2"
    repo_id = "username/corpus_name"
    hf_token = "your_hf_token_here"
    sampling_rate = 16000
    # ======================

    logger.info("Loading processor...")
    processor = AutoProcessor.from_pretrained(model_id)

    logger.info("Preparing datasets...")
    datasets = {
        "train": Dataset.from_dict(create_data_dict(train_dir, label_map)),
        "test": Dataset.from_dict(create_data_dict(test_dir, label_map)),
        "valid": Dataset.from_dict(create_data_dict(valid_dir, label_map)),
    }

    class_label = ClassLabel(num_classes=len(label_map), names=list(label_map.keys()))
    speech = DatasetDict(datasets)
    speech = speech.cast_column("audio", Audio(sampling_rate=sampling_rate))
    speech = speech.cast_column("label", class_label)

    logger.info("Extracting features...")
    processed_dataset = speech.map(
        lambda x: prepare_dataset(x, processor),
        remove_columns=["audio"],
        num_proc=1,
        desc="Extracting input features"
    )

    logger.info("Logging in to Hugging Face Hub...")
    login(token=hf_token)

    logger.info(f"Pushing dataset to hub: {repo_id}")
    processed_dataset.push_to_hub(repo_id, token=hf_token)

    logger.info("Dataset successfully uploaded.")


if __name__ == "__main__":
    main()
