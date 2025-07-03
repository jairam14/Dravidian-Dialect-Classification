# Dialect Classification (Whisper + LoRA)

This repository contains code for fine-tuning the Whisper model using LoRA-32 to classify dialects.

## Structure

- `train_lora.py` — LoRA fine-tuning script
- `evaluate_model.py` — Evaluate and visualize performance
- `config.yaml` — All model & training parameters
- `requirements.txt` — Dependencies

## Usage

1. Update `config.yaml` with your Hugging Face token and dataset
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
