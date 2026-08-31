# 🛡️ DeepMock — Fake Audio & Malicious Link Detection System

DeepMock is a full-stack AI-powered security application designed to detect **AI-generated/synthetic audio** and identify **potentially malicious or phishing URLs**.

The system combines a **Convolutional Neural Network (CNN)** for audio classification with a rule-based URL analysis module, providing users with a simple interface for analyzing uploaded audio, live recordings, and suspicious links.

---

## 🚀 Key Features

- 🎙️ AI-Generated Audio Detection
- 🧠 CNN-based audio classification
- 🎤 Live browser audio recording
- 🔗 Malicious / phishing URL detection
- 📁 Audio file upload and analysis
- ⚡ Real-time prediction results
- 🌐 Web-based interface
- 🖥️ Flask REST API backend
- 📊 Model evaluation and performance analysis

---

## 🏗️ System Architecture

```text
                    ┌─────────────────────┐
                    │      DeepMock       │
                    │    Web Interface    │
                    └──────────┬──────────┘
                               │
                 ┌─────────────┴─────────────┐
                 │                           │
                 ▼                           ▼
        ┌─────────────────┐        ┌─────────────────┐
        │  Audio Analysis │        │  Link Analysis  │
        └────────┬────────┘        └────────┬────────┘
                 │                          │
                 ▼                          ▼
        ┌─────────────────┐        ┌─────────────────┐
        │ Feature         │        │ URL Risk Factor │
        │ Extraction      │        │ Analysis        │
        └────────┬────────┘        └────────┬────────┘
                 │                          │
                 ▼                          ▼
        ┌─────────────────┐        ┌─────────────────┐
        │ CNN Model       │        │ Rule-Based      │
        │ Classification  │        │ Detection       │
        └────────┬────────┘        └────────┬────────┘
                 │                          │
                 └────────────┬─────────────┘
                              ▼
                     ┌─────────────────┐
                     │ Detection Result│
                     │   REAL / FAKE   │
                     │   SAFE / RISKY  │
                     └─────────────────┘
DeepMock/
│
├── app.py
│   └── Flask backend and API endpoints
│
├── index.html
│   └── Web-based user interface
│
├── train_model.ipynb
│   └── CNN training and evaluation pipeline
│
├── saved_model/
│   ├── spoofcnn_model.pth
│   │   └── Trained CNN model
│   │
│   ├── fake_audio_model.pkl
│   ├── scaler.pkl
│   ├── feature_columns.json
│   └── metadata.json
│
├── requirements.txt
│   └── Python dependencies
│
├── packages.txt
│   └── System-level dependencies
│
├── render.yaml
│   └── Render deployment configuration
│
├── runtime.txt
│   └── Python runtime configuration
│
├── sort_dataset.py
│   └── Dataset organization utility
│
├── evaluation_plots.png
├── feature_distributions.png
└── feature_importance.png


