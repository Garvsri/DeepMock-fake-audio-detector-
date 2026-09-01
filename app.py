"""
DeepMock - Fake Audio & Link Detection System
Backend API

Audio:
    Log-Mel Spectrogram -> SpoofCNN -> REAL / FAKE

URL:
    Heuristic URL risk analysis
"""

# ============================================================
# IMPORTS
# ============================================================

import os
import re
import tempfile
import warnings
from datetime import datetime
from urllib.parse import urlparse

import numpy as np
import librosa
import soundfile as sf

import torch
import torch.nn as nn
import torch.nn.functional as F

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from url_detector import predict_url

warnings.filterwarnings("ignore")


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)

CORS(
    app,
    resources={r"/*": {"origins": "*"}}
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CNN_MODEL_PATH = os.path.join(
    BASE_DIR,
    "saved_model",
    "spoofcnn_model.pth"
)

# Old model path kept only for reference / compatibility
OLD_MODEL_PATH = os.path.join(
    BASE_DIR,
    "saved_model",
    "fake_audio_model.pkl"
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("========================================")
print("DeepMock Backend Starting")
print("Device:", DEVICE)
print("CNN Model:", CNN_MODEL_PATH)
print("========================================")


# ============================================================
# SPOOFCNN MODEL
# ============================================================

class SpoofCNN(nn.Module):

    def __init__(self, dropout=0.3):

        super().__init__()

        def block(c_in, c_out):

            return nn.Sequential(

                nn.Conv2d(
                    c_in,
                    c_out,
                    kernel_size=3,
                    padding=1
                ),

                nn.BatchNorm2d(c_out),

                nn.ReLU(inplace=True),

                nn.Conv2d(
                    c_out,
                    c_out,
                    kernel_size=3,
                    padding=1
                ),

                nn.BatchNorm2d(c_out),

                nn.ReLU(inplace=True),

                nn.MaxPool2d(2)
            )

        self.features = nn.Sequential(

            block(1, 32),

            block(32, 64),

            block(64, 128),

            block(128, 128)
        )

        self.global_pool = nn.AdaptiveAvgPool2d(
            (1, 1)
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Dropout(dropout),

            nn.Linear(128, 64),

            nn.ReLU(inplace=True),

            nn.Dropout(dropout),

            nn.Linear(64, 1)
        )


    def forward(self, x):

        x = self.features(x)

        x = self.global_pool(x)

        x = self.classifier(x)

        return x.squeeze(1)


# ============================================================
# LOAD CNN MODEL
# ============================================================

def load_cnn_model():

    if not os.path.exists(CNN_MODEL_PATH):

        print("❌ CNN model not found:")
        print(CNN_MODEL_PATH)

        return None

    try:

        # ----------------------------------------------------
        # Create model architecture
        # ----------------------------------------------------

        model = SpoofCNN()

        # ----------------------------------------------------
        # Load checkpoint
        # ----------------------------------------------------

        checkpoint = torch.load(
            CNN_MODEL_PATH,
            map_location=DEVICE
        )

        # ----------------------------------------------------
        # CHECKPOINT DEBUG
        # ----------------------------------------------------

        print("========================================")
        print("CHECKPOINT DEBUG")
        print("Checkpoint path:", CNN_MODEL_PATH)
        print("Checkpoint type:", type(checkpoint))

        if isinstance(checkpoint, dict):

            print(
                "Number of tensors:",
                len(checkpoint)
            )

            if "classifier.5.bias" in checkpoint:

                print(
                    "classifier.5.bias:",
                    checkpoint["classifier.5.bias"]
                )

            if "classifier.5.weight" in checkpoint:

                print(
                    "classifier.5.weight mean:",
                    checkpoint["classifier.5.weight"].mean().item()
                )

        print("========================================")

        # ----------------------------------------------------
        # Load state dictionary
        # ----------------------------------------------------

        if isinstance(checkpoint, dict):

            model.load_state_dict(
                checkpoint,
                strict=True
            )

        else:

            raise RuntimeError(
                "Unexpected CNN checkpoint format."
            )

        # ----------------------------------------------------
        # Move model to device
        # ----------------------------------------------------

        model = model.to(DEVICE)

        # ----------------------------------------------------
        # IMPORTANT:
        # BatchNorm uses learned running statistics
        # during inference.
        # ----------------------------------------------------

        model.eval()

        # ----------------------------------------------------
        # MODEL WEIGHT DIAGNOSTICS
        # ----------------------------------------------------

        print("========================================")
        print("SPOOFCNN WEIGHT CHECK")

        total_params = 0
        nonzero_params = 0

        for name, param in model.named_parameters():

            total_params += param.numel()

            nonzero_params += (
                torch.count_nonzero(param).item()
            )

            print(
                f"{name}: "
                f"shape={tuple(param.shape)}, "
                f"mean={param.mean().item():.6f}, "
                f"std={param.std().item():.6f}"
            )

        print("Total parameters:", total_params)
        print("Non-zero parameters:", nonzero_params)

        print("========================================")

        print("✅ SpoofCNN loaded successfully")

        return model

    except Exception as e:

        print("❌ Failed to load SpoofCNN")
        print("Error:", str(e))

        return None


# ============================================================
# LOAD ONCE WHEN SERVER STARTS
# ============================================================

cnn_model = load_cnn_model()


# ============================================================
# CNN FEATURE SENSITIVITY TEST
# ============================================================

print("========================================")
print("CNN FEATURE SENSITIVITY TEST")

if cnn_model is not None:

    cnn_model.eval()

    with torch.no_grad():

        # ----------------------------------------------------
        # Test input 1: all zeros
        # ----------------------------------------------------

        x1 = torch.zeros(
            1, 1, 80, 250
        ).to(DEVICE)

        # ----------------------------------------------------
        # Test input 2: all ones
        # ----------------------------------------------------

        x2 = torch.ones(
            1, 1, 80, 250
        ).to(DEVICE)

        # ----------------------------------------------------
        # Test input 3: random
        # ----------------------------------------------------

        x3 = torch.randn(
            1, 1, 80, 250
        ).to(DEVICE)

        # ----------------------------------------------------
        # Extract CNN features
        # ----------------------------------------------------

        f1 = cnn_model.features(x1)

        f2 = cnn_model.features(x2)

        f3 = cnn_model.features(x3)

        # ----------------------------------------------------
        # Diagnostics
        # ----------------------------------------------------

        print(
            "Feature 1 shape:",
            f1.shape
        )

        print(
            "Feature 1 mean:",
            f1.mean().item()
        )

        print(
            "Feature 1 std:",
            f1.std().item()
        )

        print(
            "Feature 2 mean:",
            f2.mean().item()
        )

        print(
            "Feature 2 std:",
            f2.std().item()
        )

        print(
            "Feature 3 mean:",
            f3.mean().item()
        )

        print(
            "Feature 3 std:",
            f3.std().item()
        )

        # ----------------------------------------------------
        # Compare features
        # ----------------------------------------------------

        zero_ones_diff = torch.mean(
            torch.abs(f1 - f2)
        ).item()

        zero_random_diff = torch.mean(
            torch.abs(f1 - f3)
        ).item()

        print(
            "ZERO vs ONES difference:",
            zero_ones_diff
        )

        print(
            "ZERO vs RANDOM difference:",
            zero_random_diff
        )

else:

    print(
        "❌ CNN model is None."
    )

print("========================================")
# ============================================================
# AUDIO PREPROCESSING
# ============================================================

# ============================================================
# AUDIO PREPROCESSING
# MUST MATCH COLAB TRAINING PREPROCESSING
# ============================================================

SAMPLE_RATE = 16000

N_MELS = 80

N_FFT = 1024

HOP_LENGTH = 256

TARGET_FRAMES = 250

AUDIO_DURATION = 4.0

# These values came from the training configuration
FEATURE_MEAN = -57.714630126953125
FEATURE_STD = 19.16043472290039


def audio_to_logmel(audio_path):

    """
    Convert audio into the same format used during
    SpoofCNN training.

    Output:
        torch.Tensor
        shape = [1, 1, 80, 250]
    """

    # --------------------------------------------------------
    # Load audio
    # --------------------------------------------------------

    y, sr = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    if y is None or len(y) == 0:
        raise ValueError("Could not read audio data.")

    # --------------------------------------------------------
    # Remove NaN / Inf
    # --------------------------------------------------------

    y = np.nan_to_num(
        y,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    # --------------------------------------------------------
    # Make audio exactly 4 seconds
    # --------------------------------------------------------

    target_samples = int(
        SAMPLE_RATE * AUDIO_DURATION
    )

    if len(y) < target_samples:

        y = np.pad(
            y,
            (0, target_samples - len(y))
        )

    else:

        y = y[:target_samples]

    # --------------------------------------------------------
    # Mel Spectrogram
    # --------------------------------------------------------

    mel = librosa.feature.melspectrogram(

        y=y,

        sr=SAMPLE_RATE,

        n_mels=N_MELS,

        n_fft=N_FFT,

        hop_length=HOP_LENGTH,

        power=2.0
    )

    # --------------------------------------------------------
    # Power -> dB
    # --------------------------------------------------------

    log_mel = librosa.power_to_db(
        mel,
        ref=np.max
    )

    # --------------------------------------------------------
    # Convert to torch
    # --------------------------------------------------------

    spectrogram = torch.tensor(
        log_mel,
        dtype=torch.float32
    )

    # --------------------------------------------------------
    # Add batch + channel dimensions
    #
    # [80, time]
    #      ↓
    # [1, 1, 80, time]
    # --------------------------------------------------------

    spectrogram = (
        spectrogram
        .unsqueeze(0)
        .unsqueeze(0)
    )

    # --------------------------------------------------------
    # Force exactly 80 x 250
    # --------------------------------------------------------

    spectrogram = F.interpolate(

        spectrogram,

        size=(
            N_MELS,
            TARGET_FRAMES
        ),

        mode="bilinear",

        align_corners=False
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # Use TRAINING mean/std.
    #
    # DO NOT calculate mean/std from each test file.
    # --------------------------------------------------------

    spectrogram = (
        spectrogram - FEATURE_MEAN
    ) / FEATURE_STD

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------

    print("========================================")
    print("AUDIO PREPROCESSING")
    print("File:", audio_path)
    print("Sample rate:", SAMPLE_RATE)
    print("Input shape:", spectrogram.shape)
    print("Min:", spectrogram.min().item())
    print("Max:", spectrogram.max().item())
    print("Mean:", spectrogram.mean().item())
    print("Std:", spectrogram.std().item())
    print("========================================")

    return spectrogram


# ============================================================
# CNN AUDIO PREDICTION
# ============================================================

def predict_audio_cnn(audio_path):

    """
    Run SpoofCNN inference.

    Returns:
        prediction
        confidence
        probability
        model_used
    """

    global cnn_model


    if cnn_model is None:

        raise RuntimeError(
            "SpoofCNN model is not loaded."
        )


    # --------------------------------------------------------
    # Create CNN input
    # --------------------------------------------------------

    x = audio_to_logmel(audio_path)

    x = x.to(DEVICE)


    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    with torch.no_grad():

        logits = cnn_model(x)

        probability = torch.sigmoid(logits)

        probability = float(
            probability.item()
        )

        logit = float(
            logits.item()
        )
        print("========================================")
        print("AUDIO:", audio_path)
        print("INPUT SHAPE:", x.shape)
        print("INPUT MIN:", float(x.min()))
        print("INPUT MAX:", float(x.max()))
        print("INPUT MEAN:", float(x.mean()))
        print("INPUT STD:", float(x.std()))
        print("MODEL LOGIT:", logit)
        print("FAKE PROBABILITY:", probability)
        print("========================================")

    # --------------------------------------------------------
    # Classification
    #
    # IMPORTANT:
    #
    # This assumes:
    #     0 = REAL
    #     1 = FAKE
    #
    # which matches the previous DeepMock convention.
    # --------------------------------------------------------

    if probability >= 0.5:

        prediction = "FAKE"

        confidence = probability

    else:

        prediction = "REAL"

        confidence = 1.0 - probability


    return {

        "prediction": prediction,

        "confidence": confidence,

        "fake_probability": probability,

        "real_probability": 1.0 - probability,

        "logit": logit,

        "model_used": "SpoofCNN",

        "input_shape": list(x.shape)
    }


# ============================================================
# AUDIO ROUTER FUNCTION
# ============================================================

def predict_audio(audio_path):

    """
    Main audio prediction function.

    SpoofCNN is now the primary audio detector.
    """

    result = predict_audio_cnn(
        audio_path
    )

    return result


# ============================================================
# URL / LINK ANALYSIS
# ============================================================
@app.route("/detect-url", methods=["POST"])
def detect_url():
    try:
        data = request.get_json()

        if not data:
            return jsonify({
                "success": False,
                "error": "No JSON data received"
            }), 400

        url = data.get("url", "").strip()

        if not url:
            return jsonify({
                "success": False,
                "error": "Please enter a URL"
            }), 400

        result = predict_url(url)

        return jsonify({
            "success": True,
            "result": result
        })

    except Exception as e:
        print("URL detection error:", e)

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500
SUSPICIOUS_PATTERNS = [

    r'bit\.ly',

    r'tinyurl',

    r'goo\.gl',

    r't\.co',

    r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}',

    r'[a-z0-9]{20,}\.',

    r'(free|win|prize|claim|urgent|verify)',

    r'(login|signin|secure|update|confirm).*\.(xyz|tk|ml|ga|cf)',

    r'paypal.*\.(?!com)',

    r'bank.*\.(xyz|tk|info|biz)',
]


KNOWN_SAFE_DOMAINS = {

    'google.com',

    'youtube.com',

    'github.com',

    'microsoft.com',

    'apple.com',

    'amazon.com',

    'wikipedia.org',

    'stackoverflow.com'
}


def analyze_url(url: str) -> dict:

    """
    Analyze URL using DeepMock's existing
    heuristic risk scoring.
    """

    url = url.strip()

    risk_score = 0

    risk_factors = []


    # --------------------------------------------------------
    # Parse URL
    # --------------------------------------------------------

    try:

        parsed = urlparse(

            url
            if url.startswith("http")
            else "http://" + url
        )

        domain = parsed.netloc.lower()

        domain = domain.replace(
            "www.",
            ""
        )

    except Exception:

        domain = url


    # --------------------------------------------------------
    # Known safe domains
    # --------------------------------------------------------

    if domain in KNOWN_SAFE_DOMAINS:

        return {

            'risk': 'LOW',

            'score': 5,

            'factors': [
                'Known safe domain'
            ],

            'domain': domain,

            'url': url
        }


    # --------------------------------------------------------
    # Suspicious patterns
    # --------------------------------------------------------

    for pattern in SUSPICIOUS_PATTERNS:

        if re.search(
            pattern,
            url,
            re.IGNORECASE
        ):

            risk_score += 20

            risk_factors.append(
                f'Pattern match: {pattern}'
            )


    # --------------------------------------------------------
    # HTTPS
    # --------------------------------------------------------

    if not url.startswith('https'):

        risk_score += 15

        risk_factors.append(
            'No HTTPS'
        )


    # --------------------------------------------------------
    # Excessive subdomains
    # --------------------------------------------------------

    if url.count('.') > 4:

        risk_score += 10

        risk_factors.append(
            'Excessive subdomains'
        )


    # --------------------------------------------------------
    # Long URL
    # --------------------------------------------------------

    if len(url) > 150:

        risk_score += 10

        risk_factors.append(
            'Unusually long URL'
        )


    # --------------------------------------------------------
    # @ symbol
    # --------------------------------------------------------

    if re.search(r'@', url):

        risk_score += 25

        risk_factors.append(
            '@ symbol in URL '
            '(credential bypass attempt)'
        )


    # --------------------------------------------------------
    # Limit score
    # --------------------------------------------------------

    risk_score = min(
        risk_score,
        100
    )


    if risk_score < 30:

        risk_level = 'LOW'

    elif risk_score < 60:

        risk_level = 'MEDIUM'

    else:

        risk_level = 'HIGH'


    return {

        'risk': risk_level,

        'score': risk_score,

        'factors':
            risk_factors
            if risk_factors
            else [
                'No suspicious patterns detected'
            ],

        'domain': domain,

        'url': url
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route(
    '/health',
    methods=['GET']
)
def health():

    return jsonify({

        'status': 'ok',

        # Audio detection
        'audio_model_loaded':
            cnn_model is not None,

        'audio_model':
            'SpoofCNN',

        # URL detection
        'url_model_loaded':
            True,

        'url_models': [
            'Decision Tree',
            'Gradient Boosting',
            'XGBoost'
        ],

        'url_ensemble':
            'Majority Voting',

        'device':
            str(DEVICE),

        'timestamp':
            datetime.utcnow().isoformat()
    })

# ============================================================
# AUDIO UPLOAD ANALYSIS
# ============================================================

@app.route(
    '/analyze/audio',
    methods=['POST']
)
def analyze_audio():

    """
    Analyze uploaded audio file
    using SpoofCNN.
    """

    # --------------------------------------------------------
    # Check file
    # --------------------------------------------------------

    if 'audio' not in request.files:

        return jsonify({

            'error':
                'No audio file provided'

        }), 400


    file = request.files['audio']


    if file.filename == '':

        return jsonify({

            'error':
                'No file selected'

        }), 400


    # --------------------------------------------------------
    # Allowed formats
    # --------------------------------------------------------

    allowed = {

        '.wav',

        '.mp3',

        '.ogg',

        '.flac',

        '.m4a',

        '.webm'
    }


    ext = os.path.splitext(
        file.filename
    )[1].lower()


    if ext not in allowed:

        return jsonify({

            'error':
                f'Unsupported format: {ext}'

        }), 400


    tmp_path = None


    try:

        # ----------------------------------------------------
        # Save temporary audio
        # ----------------------------------------------------

        with tempfile.NamedTemporaryFile(

            suffix=ext,

            delete=False

        ) as tmp:

            file.save(
                tmp.name
            )

            tmp_path = tmp.name


        # ----------------------------------------------------
        # CNN prediction
        # ----------------------------------------------------

        result = predict_audio(
            tmp_path
        )


        return jsonify({

            'success': True,

            'filename':
                file.filename,

            'result':
                result,

            'timestamp':
                datetime.utcnow().isoformat()
        })


    except Exception as e:

        print(
            "❌ Audio analysis error:",
            str(e)
        )


        return jsonify({

            'success': False,

            'error':
                str(e)

        }), 500


    finally:

        # ----------------------------------------------------
        # Cleanup
        # ----------------------------------------------------

        if tmp_path is not None:

            try:

                if os.path.exists(
                    tmp_path
                ):

                    os.unlink(
                        tmp_path
                    )

            except Exception:

                pass


# ============================================================
# RECORDED AUDIO ANALYSIS
# ============================================================

@app.route(
    '/analyze/recorded',
    methods=['POST']
)
def analyze_recorded():

    """
    Analyze audio recorded directly
    from the browser.
    """

    if 'audio' not in request.files:

        return jsonify({

            'error':
                'No audio data provided'

        }), 400


    audio_blob = request.files[
        'audio'
    ]


    webm_path = None

    wav_path = None


    try:

        # ----------------------------------------------------
        # Save WebM
        # ----------------------------------------------------

        with tempfile.NamedTemporaryFile(

            suffix='.webm',

            delete=False

        ) as tmp:

            audio_blob.save(
                tmp.name
            )

            webm_path = tmp.name


        # ----------------------------------------------------
        # Convert to WAV
        # ----------------------------------------------------

        wav_path = webm_path.replace(
            '.webm',
            '.wav'
        )


        try:

            y, sr = librosa.load(

                webm_path,

                sr=SAMPLE_RATE,

                mono=True
            )


            sf.write(

                wav_path,

                y,

                sr
            )


            process_path = wav_path


        except Exception:

            # If direct loading fails,
            # try original file.

            process_path = webm_path


        # ----------------------------------------------------
        # CNN prediction
        # ----------------------------------------------------

        result = predict_audio(
            process_path
        )


        return jsonify({

            'success': True,

            'source':
                'live_recording',

            'result':
                result,

            'timestamp':
                datetime.utcnow().isoformat()
        })


    except Exception as e:

        print(
            "❌ Recorded audio error:",
            str(e)
        )


        return jsonify({

            'success': False,

            'error':
                str(e)

        }), 500


    finally:

        # ----------------------------------------------------
        # Cleanup
        # ----------------------------------------------------

        for path in [
            webm_path,
            wav_path
        ]:

            if path is not None:

                try:

                    if os.path.exists(
                        path
                    ):

                        os.unlink(
                            path
                        )

                except Exception:

                    pass

# ============================================================
# BATCH URL ANALYSIS - ML ENSEMBLE
# ============================================================

@app.route(
    '/analyze/batch-links',
    methods=['POST']
)
def batch_links():

    """
    Analyze multiple URLs using:

    1. Decision Tree
    2. Gradient Boosting
    3. XGBoost

    Final result is produced using
    majority voting.
    """

    data = request.get_json(silent=True)

    if not data or 'urls' not in data:

        return jsonify({
            'success': False,
            'error': 'No URLs provided'
        }), 400

    urls = data['urls']

    if not isinstance(urls, list):

        return jsonify({
            'success': False,
            'error': 'URLs must be provided as a list'
        }), 400

    results = []

    for url in urls[:20]:

        url = str(url).strip()

        if not url:
            continue

        try:

            result = predict_url(url)

            results.append(result)

        except Exception as e:

            results.append({
                'url': url,
                'prediction': 'ERROR',
                'error': str(e)
            })

    return jsonify({

        'success': True,

        'results': results,

        'total_urls': len(results),

        'timestamp':
            datetime.utcnow().isoformat()

    })

    """
    Analyze multiple URLs.
    """

    data = request.get_json()


    if not data or 'urls' not in data:

        return jsonify({

            'error':
                'No URLs provided'

        }), 400


    results = []


    for url in data['urls'][:20]:

        results.append({

            'url': url,

            **analyze_url(url)
        })


    return jsonify({

        'success': True,

        'results': results,

        'timestamp':
            datetime.utcnow().isoformat()
    })


# ============================================================
# FRONTEND
# ============================================================

@app.route("/")
def serve_frontend():

    return send_from_directory(
        BASE_DIR,
        "index.html"
    )


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":

    print()
    print("========================================")
    print(" DeepMock Server")
    print("========================================")
    print("Audio model : SpoofCNN")
    print("Model path  :", CNN_MODEL_PATH)
    print("Device      :", DEVICE)
    print("========================================")
    print()

    app.run(

        host="0.0.0.0",

        port=10000,

        debug=False
    )