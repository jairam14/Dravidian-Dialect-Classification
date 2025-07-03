# Dialect Classification Pipeline (Whisper Feature Extraction + HF Upload)

This repository contains a Python pipeline for preparing and uploading dialect-specific speech datasets (Tamil, Malayalam, Kannada) to the Hugging Face Hub.

##  Project Structure

- `feature_extraction_and_upload.py`: Main script to extract Whisper features and push to Hub.
- `config.yaml`: Configuration for language, directories, and Hugging Face details.
- `language_mapping.json`: Mapping of dialect labels to numeric class IDs.
- `requirements.txt`: Required Python dependencies.

##  Usage

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
