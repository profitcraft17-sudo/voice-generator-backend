---
title: Voice Generator Backend
emoji: 🎙️
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 3.50.0
app_file: app.py
pinned: false
license: mit
---

# RVC Multi-Voice Inference Engine Backend

This is the dedicated backend core engine for the AI Custom Voice Generator platform. It processes text segments sent from the Vercel frontend, runs Text-to-Speech tracking, and overlays the custom trained Retrieval-based Voice Conversion (RVC) model weights.

## 📁 Repository Structure Inside Hugging Face:
* `app.py` - Core FastAPI & Gradio execution matrix.
* `config.py` - Multi-voice runtime thresholds and parameters calibration room.
* `requirements.txt` - System packages mapping dependencies.
* `models/` - Dynamic folder holding the voice weight bundles (`.pth` and `.index`).

## 🚀 Connection Setup Note:
The system automatically exposes a fast binary stream pipeline endpoint at `/api/predict` which bridges directly with the `app.js` routing logic on the frontend repository.
