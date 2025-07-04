# Dravidian-Dialect-Classification
###   WhispAdapt: Whisper-based Lightweight Dialect Classification

This repository contains the codebase for **WhispAdapt**, a framework designed for dialect classification in Dravidian languages. It builds on a pretrained Whisper encoder, which serves as a robust feature extractor and is adapted using Parameter-Efficient Fine-Tuning (PEFT) techniques such as LoRA and QLoRA. Instead of fine-tuning the entire model, only a small number of trainable parameters are introduced through lightweight adapters, enabling efficient adaptation to new tasks. A lightweight classification head is placed on top of the Whisper encoder to perform dialect prediction.

---
