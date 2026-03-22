# 🛡️ DeepMock — Fake Audio & Link Detection System

A full-stack AI security project that detects deepfake/synthetic audio and malicious URLs in real time.

---

## 📁 Project Structure

```
fake-audio-detection/
├── frontend/
│   └── index.html              ← Single-page UI (open in browser)
│
├── backend/
│   ├── app.py                  ← Flask REST API
│   └── requirements.txt        ← Python dependencies
│
├── notebook/
│   ├── train_model.ipynb       ← Training pipeline (Jupyter)
│   └── saved_model/            ← Created after training
│       ├── fake_audio_model.pkl
│       ├── scaler.pkl
│       ├── feature_columns.json
│       └── metadata.json
│
└── dataset/
    ├── real/                   ← Place genuine audio here (.wav/.mp3)
    └── fake/                   ← Place AI-generated audio here (.wav/.mp3)
```

---

## ⚡ Quick Start

### 1. Train the Model (Jupyter Notebook)

```bash
cd notebook
pip install jupyter
jupyter notebook train_model.ipynb
# Run all cells — works in demo mode even without audio files
```

**To use a real dataset:**
- Download [ASVspoof 2019](https://www.asvspoof.org/) or [WaveFake](https://github.com/joepenna/wavefake)
- Place real audio in `dataset/real/`
- Place fake/synthesized audio in `dataset/fake/`
- Re-run the notebook

---

### 2. Start the Backend API

```bash
cd backend
pip install -r requirements.txt
python app.py
# API runs at http://localhost:5000
```

**API Endpoints:**

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Check API + model status |
| POST | `/analyze/audio` | Upload audio file for detection |
| POST | `/analyze/recorded` | Submit browser-recorded audio blob |
| POST | `/analyze/link` | Scan a URL for phishing indicators |
| POST | `/analyze/batch-links` | Scan up to 20 URLs at once |

---

### 3. Open the Frontend

```bash
# Simply open in any modern browser:
open frontend/index.html
# or
python -m http.server 8080   # then visit http://localhost:8080/frontend/
```

> The frontend works **offline in demo mode** — it simulates results when the API is not running. For real ML inference, start the backend first.

---

## 🎙️ Live Judge Recording Feature

The **Record Live** tab lets judges record audio directly in the browser and immediately classifies it:

1. Click **Record Live** tab
2. Press the red button → grant microphone access
3. Speak into the microphone
4. Press stop → click **Analyse Recording**
5. Results appear within seconds

---

## 🤖 Model Details

| Component | Details |
|-----------|---------|
| Features | 57 features: MFCCs ×20, spectral centroid/bandwidth/rolloff/contrast, chroma, ZCR, RMS, tempo, mel spectrogram, F0 (pitch) |
| Models | Random Forest (300 trees) + Gradient Boosting + XGBoost — soft voting ensemble |
| Evaluation | 5-fold cross-validation, ROC-AUC, confusion matrix |
| Inference | ~500ms on CPU for a 10-second clip |

---

## 🔗 Link Detection Logic

Risk factors checked:
- HTTP vs HTTPS
- Raw IP addresses in URL
- Known URL shorteners (bit.ly, tinyurl, etc.)
- Suspicious keywords (free, win, prize, urgent, verify)
- Excessive subdomains
- @ symbol bypass attempts
- High-risk TLDs (.xyz, .tk, .ml, .ga, .cf)
- URL length anomalies

---

## 📦 Recommended Datasets

| Dataset | Size | Source |
|---------|------|--------|
| ASVspoof 2019 | ~17GB | https://www.asvspoof.org/ |
| WaveFake | ~48GB | https://github.com/joepenna/wavefake |
| FakeAVCeleb | ~20GB | https://github.com/DASH-Lab/FakeAVCeleb |
| Half-Truth | ~5GB | https://github.com/nii-yamagishilab/half-truth |

---

## 🛠 Tech Stack

- **Frontend**: Vanilla HTML/CSS/JS (no build step required)
- **Backend**: Python · Flask · librosa · scikit-learn · XGBoost
- **Notebook**: Jupyter · numpy · pandas · matplotlib · seaborn
- **Audio processing**: librosa, soundfile, sounddevice
