"""
Fake Audio & Link Detection System - Backend API
Flask REST API for audio analysis and link scanning
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import librosa
import joblib
import os
import re
import requests
import hashlib
import tempfile
import soundfile as sf
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
MODEL_PATH = "saved_model/fake_audio_model.pkl"
SCALER_PATH = "saved_model/scaler.pkl"

# ──────────────────────────────────────────────
# Feature extraction (must match notebook exactly)
# ──────────────────────────────────────────────

def extract_features(audio_path):
    """Extract audio features identical to training pipeline."""
    y, sr = librosa.load(audio_path, sr=22050, duration=30)

    features = {}

    # MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    for i in range(20):
        features[f'mfcc_{i}_mean'] = float(np.mean(mfccs[i]))
        features[f'mfcc_{i}_std']  = float(np.std(mfccs[i]))

    # Spectral features
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spec_rolloff   = librosa.feature.spectral_rolloff(y=y, sr=sr)
    spec_contrast  = librosa.feature.spectral_contrast(y=y, sr=sr)

    features['spectral_centroid_mean']  = float(np.mean(spec_centroid))
    features['spectral_centroid_std']   = float(np.std(spec_centroid))
    features['spectral_bandwidth_mean'] = float(np.mean(spec_bandwidth))
    features['spectral_rolloff_mean']   = float(np.mean(spec_rolloff))
    features['spectral_contrast_mean']  = float(np.mean(spec_contrast))

    # Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features['chroma_mean'] = float(np.mean(chroma))
    features['chroma_std']  = float(np.std(chroma))

    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y)
    features['zcr_mean'] = float(np.mean(zcr))
    features['zcr_std']  = float(np.std(zcr))

    # RMS Energy
    rms = librosa.feature.rms(y=y)
    features['rms_mean'] = float(np.mean(rms))
    features['rms_std']  = float(np.std(rms))

    # Tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    features['tempo'] = float(tempo)

    # Mel spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    features['mel_mean'] = float(np.mean(mel))
    features['mel_std']  = float(np.std(mel))

    # Fundamental frequency stats
    f0, voiced_flag, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'),
                                      fmax=librosa.note_to_hz('C7'))
    f0_clean = f0[~np.isnan(f0)] if f0 is not None else np.array([0])
    features['f0_mean']      = float(np.mean(f0_clean)) if len(f0_clean) > 0 else 0
    features['f0_std']       = float(np.std(f0_clean))  if len(f0_clean) > 0 else 0
    features['voiced_ratio'] = float(np.mean(voiced_flag)) if voiced_flag is not None else 0

    return features


def load_model():
    """Load trained model and scaler."""
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        model  = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler
    return None, None


def predict_audio(audio_path):
    """Run inference on an audio file."""
    model, scaler = load_model()

    features_dict = extract_features(audio_path)
    feature_names = sorted(features_dict.keys())
    feature_vector = np.array([features_dict[k] for k in feature_names]).reshape(1, -1)

    if model is None:
        # Heuristic fallback when model not trained yet
        zcr   = features_dict.get('zcr_mean', 0)
        rms   = features_dict.get('rms_mean', 0)
        f0_std = features_dict.get('f0_std', 0)

        is_fake = (zcr < 0.05 and f0_std < 20) or rms < 0.01
        confidence = 0.65 if is_fake else 0.72
        return {
            'prediction': 'FAKE' if is_fake else 'REAL',
            'confidence': confidence,
            'features': features_dict,
            'model_used': 'heuristic_fallback'
        }

    feature_vector_scaled = scaler.transform(feature_vector)
    prediction  = model.predict(feature_vector_scaled)[0]
    proba       = model.predict_proba(feature_vector_scaled)[0]
    confidence  = float(np.max(proba))

    return {
        'prediction': 'FAKE' if prediction == 1 else 'REAL',
        'confidence': confidence,
        'features': features_dict,
        'model_used': 'trained_model'
    }


# ──────────────────────────────────────────────
# Link / URL Analysis
# ──────────────────────────────────────────────

SUSPICIOUS_PATTERNS = [
    r'bit\.ly', r'tinyurl', r'goo\.gl', r't\.co',
    r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}',   # raw IP
    r'[a-z0-9]{20,}\.',                          # random-hash subdomain
    r'(free|win|prize|claim|urgent|verify)',
    r'(login|signin|secure|update|confirm).*\.(xyz|tk|ml|ga|cf)',
    r'paypal.*\.(?!com)',
    r'bank.*\.(xyz|tk|info|biz)',
]

KNOWN_SAFE_DOMAINS = {
    'google.com', 'youtube.com', 'github.com', 'microsoft.com',
    'apple.com', 'amazon.com', 'wikipedia.org', 'stackoverflow.com'
}


def analyze_url(url: str) -> dict:
    """Heuristic URL risk scoring."""
    url = url.strip()
    risk_score   = 0
    risk_factors = []

    try:
        from urllib.parse import urlparse
        parsed = urlparse(url if url.startswith('http') else 'http://' + url)
        domain = parsed.netloc.lower().replace('www.', '')
    except Exception:
        domain = url

    if domain in KNOWN_SAFE_DOMAINS:
        return {'risk': 'LOW', 'score': 5, 'factors': ['Known safe domain'], 'domain': domain}

    for pattern in SUSPICIOUS_PATTERNS:
        if re.search(pattern, url, re.IGNORECASE):
            risk_score += 20
            risk_factors.append(f'Pattern match: {pattern}')

    if not url.startswith('https'):
        risk_score += 15
        risk_factors.append('No HTTPS')

    if url.count('.') > 4:
        risk_score += 10
        risk_factors.append('Excessive subdomains')

    if len(url) > 150:
        risk_score += 10
        risk_factors.append('Unusually long URL')

    if re.search(r'@', url):
        risk_score += 25
        risk_factors.append('@ symbol in URL (credential bypass attempt)')

    risk_score = min(risk_score, 100)
    risk_level = 'LOW' if risk_score < 30 else ('MEDIUM' if risk_score < 60 else 'HIGH')

    return {
        'risk': risk_level,
        'score': risk_score,
        'factors': risk_factors if risk_factors else ['No suspicious patterns detected'],
        'domain': domain,
        'url': url
    }


# ──────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────

@app.route('/health', methods=['GET'])
def health():
    model, _ = load_model()
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'timestamp': datetime.utcnow().isoformat()
    })


@app.route('/analyze/audio', methods=['POST'])
def analyze_audio():
    """Analyze uploaded audio for deepfake detection."""
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    file = request.files['audio']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    allowed = {'.wav', '.mp3', '.ogg', '.flac', '.m4a', '.webm'}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed:
        return jsonify({'error': f'Unsupported format: {ext}'}), 400

    try:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name

        result = predict_audio(tmp_path)
        os.unlink(tmp_path)

        return jsonify({
            'success': True,
            'filename': file.filename,
            'result': result,
            'timestamp': datetime.utcnow().isoformat()
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/analyze/recorded', methods=['POST'])
def analyze_recorded():
    """Analyze in-browser recorded audio blob."""
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio data provided'}), 400

    audio_blob = request.files['audio']

    try:
        import subprocess, sys

        # Save original webm
        with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as tmp:
            audio_blob.save(tmp.name)
            webm_path = tmp.name

        # Convert webm to wav using librosa directly
        wav_path = webm_path.replace('.webm', '.wav')

        try:
            # Try loading webm directly with librosa
            y, sr = librosa.load(webm_path, sr=22050)
            import soundfile as sf
            sf.write(wav_path, y, sr)
            process_path = wav_path
        except Exception:
            # fallback — process webm directly
            process_path = webm_path

        result = predict_audio(process_path)

        # Cleanup
        for p in [webm_path, wav_path]:
            try:
                os.unlink(p)
            except:
                pass

        return jsonify({
            'success': True,
            'source': 'live_recording',
            'result': result,
            'timestamp': datetime.utcnow().isoformat()
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/analyze/batch-links', methods=['POST'])
def batch_links():
    """Analyze multiple URLs at once."""
    data = request.get_json()
    if not data or 'urls' not in data:
        return jsonify({'error': 'No URLs provided'}), 400

    results = []
    for url in data['urls'][:20]:   # cap at 20
        results.append({'url': url, **analyze_url(url)})

    return jsonify({'success': True, 'results': results,
                    'timestamp': datetime.utcnow().isoformat()})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
