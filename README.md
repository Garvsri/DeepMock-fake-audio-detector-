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
Install Dependencies

Create a virtual environment:

python -m venv venv

Activate it on Windows:

venv\Scripts\activate

Install the required packages:

pip install -r requirements.txt
3. Run the Application

Start the Flask backend:

python app.py

The API will run locally at:

http://localhost:5000

Open the frontend in your browser:

http://localhost:5000
ake Audio Detection

DeepMock uses a Convolutional Neural Network (CNN) to classify audio as either genuine or AI-generated/synthetic.

Audio Detection Pipeline
Audio Input
     ↓
Audio Preprocessing
     ↓
Feature Extraction
     ↓
CNN Model
     ↓
Prediction
     ↓
REAL / FAKE

The CNN learns patterns from audio representations and uses these learned features to distinguish between real human speech and synthetic/manipulated audio.

Supported Audio Input

The application can analyze:

.wav
.mp3
Browser-recorded audio
🧠 CNN Model

The main audio detection model is:

saved_model/spoofcnn_model.pth

The trained model is integrated directly into the Flask backend and is loaded during application startup.

Model Workflow
Raw Audio
    ↓
Preprocessing
    ↓
Audio Feature Extraction
    ↓
CNN
    ↓
Probability Score
    ↓
REAL / FAKE

The CNN-based approach allows DeepMock to learn relevant patterns from the audio data rather than relying exclusively on manually defined classification rules.

🎤 Live Recording

DeepMock includes a Record Live feature that allows users to record audio directly from their browser.

Steps
Open the Record Live tab.
Allow microphone access.
Press the record button.
Speak into the microphone.
Stop recording.
Select Analyse Recording.
The audio is sent to the backend.
The CNN model analyzes the recording.
The prediction is displayed on the screen.
Microphone
    ↓
Browser Recording
    ↓
Audio Blob
    ↓
Flask API
    ↓
CNN Model
    ↓
Prediction
🔗 Malicious Link Detection

DeepMock also provides a URL analysis module that evaluates URLs for common phishing and malicious-link indicators.

Risk Factors

The system checks for:

HTTP instead of HTTPS
Raw IP addresses
Known URL shorteners
Suspicious keywords
Excessive subdomains
@ symbol manipulation
Suspicious top-level domains
Unusually long URLs
Other URL structure anomalies
Example
User URL
   ↓
URL Parser
   ↓
Risk Factor Analysis
   ↓
Risk Score
   ↓
SAFE / SUSPICIOUS
🔌 API Endpoints
Method	Endpoint	Description
GET	/health	Check API and model status
POST	/analyze/audio	Analyze uploaded audio
POST	/analyze/recorded	Analyze browser-recorded audio
POST	/analyze/link	Analyze a single URL
POST	/analyze/batch-links	Analyze multiple URLs
📊 Model Evaluation

The project includes evaluation and visualization components for analyzing model performance.

Generated evaluation files include:

evaluation_plots.png
feature_distributions.png
feature_importance.png

These visualizations can be used to analyze:

Classification performance
Feature distributions
Feature importance
Model behavior
📦 Recommended Datasets

DeepMock can be trained or evaluated using publicly available synthetic-speech datasets.

Dataset	Approx. Size	Source
ASVspoof 2019	~17 GB	https://www.asvspoof.org/
WaveFake	~48 GB	https://github.com/joepenna/wavefake
FakeAVCeleb	~20 GB	https://github.com/DASH-Lab/FakeAVCeleb
Half-Truth	~5 GB	https://github.com/nii-yamagishilab/half-truth

Dataset sizes are approximate and may vary depending on the downloaded subset/version.

🛠️ Technology Stack
Frontend
HTML5
CSS3
JavaScript
Web Audio API
Backend
Python
Flask
NumPy
Pandas
Librosa
SoundFile
Machine Learning
PyTorch
Convolutional Neural Network (CNN)
Scikit-learn
Matplotlib
Audio Processing
Librosa
SoundFile
FFmpeg
Deployment
Render
🔒 Security Considerations

DeepMock is designed as a detection and analysis tool and should not be treated as a definitive security authority.

A URL classified as safe may still be malicious, and synthetic-audio detection performance can vary depending on the audio quality, generation method, and dataset used during training.

Users should avoid entering sensitive credentials or personal information into suspicious websites even when a URL is classified as low-risk.

🚀 Deployment

DeepMock can be deployed using Render.

The repository contains:

render.yaml
runtime.txt
packages.txt
requirements.txt

These files provide the configuration required for deployment.

🔮 Future Improvements

Potential future improvements include:

🎯 Improved CNN architectures
🔊 Support for additional audio formats
🧠 Transformer-based audio detection
📈 Improved confidence calibration
🌐 More advanced URL threat intelligence
🔍 Integration with external threat databases
📱 Responsive mobile interface
📊 Detailed detection reports
🔐 Additional security controls

