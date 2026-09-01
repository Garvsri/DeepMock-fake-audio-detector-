import os
import re
import joblib
import numpy as np
import pandas as pd

from urllib.parse import urlparse


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Your folder from the screenshot
MODEL_DIR = os.path.join(
    BASE_DIR,
    "url_models",
    "url_models"
)


# ============================================================
# SUSPICIOUS URL KEYWORDS
# ============================================================

SUSPICIOUS_KEYWORDS = [
    "login",
    "signin",
    "verify",
    "verification",
    "secure",
    "account",
    "update",
    "confirm",
    "confirmation",
    "password",
    "credential",
    "wallet",
    "banking",
    "payment",
    "pay",
    "invoice",
    "refund",
    "bonus",
    "prize",
    "free",
    "gift",
    "crypto",
    "bitcoin",
    "unlock",
    "support",
    "alert",
    "suspended",
    "urgent",
    "recover",
    "authenticate"
]


# ============================================================
# URL SHORTENERS
# ============================================================

SHORTENERS = {
    "bit.ly",
    "tinyurl.com",
    "t.co",
    "goo.gl",
    "is.gd",
    "buff.ly",
    "ow.ly",
    "rebrand.ly",
    "cutt.ly",
    "shorturl.at",
    "rb.gy",
    "lnkd.in"
}


# ============================================================
# ENTROPY
# ============================================================

def calculate_entropy(text):

    if not text:
        return 0.0

    values = np.frombuffer(
        text.encode("utf-8", "ignore"),
        dtype=np.uint8
    )

    counts = np.bincount(
        values,
        minlength=256
    )

    probabilities = counts[
        counts > 0
    ] / len(text)

    return float(
        -(
            probabilities *
            np.log2(probabilities)
        ).sum()
    )


# ============================================================
# URL FEATURE EXTRACTION
# ============================================================

def extract_url_features(url):

    url = str(url).strip()

    # --------------------------------------------------------
    # Add http:// if URL has no scheme
    # --------------------------------------------------------

    if re.match(
        r"^[A-Za-z][A-Za-z0-9+.-]*://",
        url
    ):
        parsed_url = urlparse(url)

    else:
        parsed_url = urlparse(
            "http://" + url
        )


    # --------------------------------------------------------
    # Extract URL components
    # --------------------------------------------------------

    domain = (
        parsed_url.hostname or ""
    ).lower()

    path = (
        parsed_url.path or ""
    )

    query = (
        parsed_url.query or ""
    )

    full_url = url.lower()


    # --------------------------------------------------------
    # Detect IP address
    # --------------------------------------------------------

    try:

        import ipaddress

        has_ip = int(
            bool(
                domain and
                ipaddress.ip_address(domain)
            )
        )

    except Exception:

        has_ip = 0


    # --------------------------------------------------------
    # Tokenization
    # --------------------------------------------------------

    tokens = re.findall(
        r"[a-z0-9]+",
        full_url
    )


    # --------------------------------------------------------
    # Suspicious keyword count
    # --------------------------------------------------------

    suspicious_count = sum(
        full_url.count(keyword)
        for keyword in SUSPICIOUS_KEYWORDS
    )


    # --------------------------------------------------------
    # Character statistics
    # --------------------------------------------------------

    num_digits = sum(
        character.isdigit()
        for character in url
    )

    num_letters = sum(
        character.isalpha()
        for character in url
    )

    num_special = sum(
        not character.isalnum()
        for character in url
    )


    # ========================================================
    # FEATURE DICTIONARY
    # ========================================================

    features = {

        # ----------------------------------------------------
        # Length features
        # ----------------------------------------------------

        "url_length": len(url),

        "domain_length": len(domain),

        "path_length": len(path),

        "query_length": len(query),

        "fragment_length": len(
            parsed_url.fragment or ""
        ),


        # ----------------------------------------------------
        # Special character counts
        # ----------------------------------------------------

        "num_dots": url.count("."),

        "num_hyphens": url.count("-"),

        "num_underscores": url.count("_"),

        "num_slashes": url.count("/"),

        "num_backslashes": url.count("\\"),

        "num_at": url.count("@"),

        "num_question": url.count("?"),

        "num_equals": url.count("="),

        "num_ampersand": url.count("&"),

        "num_percent": url.count("%"),

        "num_colon": url.count(":"),

        "num_semicolon": url.count(";"),


        # ----------------------------------------------------
        # Character statistics
        # ----------------------------------------------------

        "num_digits": num_digits,

        "num_letters": num_letters,

        "num_special": num_special,


        # ----------------------------------------------------
        # Ratios
        # ----------------------------------------------------

        "digit_ratio":
            num_digits /
            max(len(url), 1),

        "letter_ratio":
            num_letters /
            max(len(url), 1),

        "special_ratio":
            num_special /
            max(len(url), 1),


        # ----------------------------------------------------
        # Domain structure
        # ----------------------------------------------------

        "domain_dots":
            domain.count("."),

        "subdomain_count":
            max(
                domain.count(".") - 1,
                0
            ),


        # ----------------------------------------------------
        # Security features
        # ----------------------------------------------------

        "has_https":
            int(
                parsed_url.scheme.lower()
                == "https"
            ),

        "has_ip":
            has_ip,

        "has_port":
            int(
                parsed_url.port is not None
            )
            if parsed_url.hostname
            else 0,

        "has_userinfo":
            int(
                parsed_url.username is not None
                or
                parsed_url.password is not None
            ),

        "has_punycode":
            int(
                "xn--" in domain
            ),

        "has_double_slash_path":
            int(
                "//" in path
            ),

        "has_hex_encoding":
            int(
                bool(
                    re.search(
                        r"%[0-9a-fA-F]{2}",
                        url
                    )
                )
            ),


        # ----------------------------------------------------
        # Suspicious content
        # ----------------------------------------------------

        "has_suspicious_keyword":
            int(
                suspicious_count > 0
            ),

        "suspicious_keyword_count":
            suspicious_count,


        # ----------------------------------------------------
        # URL shortener
        # ----------------------------------------------------

        "uses_shortener":
            int(
                domain in SHORTENERS
                or
                any(
                    domain.endswith(
                        "." + shortener
                    )
                    for shortener
                    in SHORTENERS
                )
            ),


        # ----------------------------------------------------
        # Query parameters
        # ----------------------------------------------------

        "num_query_params":
            0
            if not query
            else query.count("&") + 1,


        # ----------------------------------------------------
        # Token statistics
        # ----------------------------------------------------

        "token_count":
            len(tokens),

        "max_token_length":
            max(
                [len(token) for token in tokens]
                or [0]
            ),

        "avg_token_length":
            float(
                np.mean(
                    [
                        len(token)
                        for token in tokens
                    ]
                )
            )
            if tokens
            else 0.0,


        # ----------------------------------------------------
        # Entropy
        # ----------------------------------------------------

        "url_entropy":
            calculate_entropy(url),

        "domain_entropy":
            calculate_entropy(domain)
    }


    return features


# ============================================================
# LOAD TRAINED MODELS
# ============================================================

def load_models():

    required_files = [
        "decision_tree.joblib",
        "gradient_boosting.joblib",
        "xgboost.joblib",
        "feature_names.joblib"
    ]

    # Check model directory
    if not os.path.exists(MODEL_DIR):

        raise FileNotFoundError(
            f"URL model directory not found: "
            f"{MODEL_DIR}"
        )


    # Check individual files
    for filename in required_files:

        filepath = os.path.join(
            MODEL_DIR,
            filename
        )

        if not os.path.exists(filepath):

            raise FileNotFoundError(
                f"Required model file not found: "
                f"{filepath}"
            )


    print("Loading URL detection models...")


    decision_tree = joblib.load(
        os.path.join(
            MODEL_DIR,
            "decision_tree.joblib"
        )
    )


    gradient_boosting = joblib.load(
        os.path.join(
            MODEL_DIR,
            "gradient_boosting.joblib"
        )
    )


    xgboost_model = joblib.load(
        os.path.join(
            MODEL_DIR,
            "xgboost.joblib"
        )
    )


    feature_names = joblib.load(
        os.path.join(
            MODEL_DIR,
            "feature_names.joblib"
        )
    )


    print("✓ Decision Tree loaded")

    print("✓ Gradient Boosting loaded")

    print("✓ XGBoost loaded")

    print(
        f"✓ {len(feature_names)} URL features loaded"
    )


    return (
        decision_tree,
        gradient_boosting,
        xgboost_model,
        feature_names
    )


# ============================================================
# LOAD MODELS ONCE
# ============================================================

try:

    (
        decision_tree,
        gradient_boosting,
        xgboost_model,
        FEATURE_NAMES
    ) = load_models()

    MODELS_LOADED = True

except Exception as error:

    print(
        "ERROR loading URL models:"
    )

    print(error)

    MODELS_LOADED = False

    decision_tree = None
    gradient_boosting = None
    xgboost_model = None
    FEATURE_NAMES = []


# ============================================================
# URL PREDICTION
# ============================================================

def predict_url(url):

    # --------------------------------------------------------
    # Validate URL
    # --------------------------------------------------------

    if not url:

        return {
            "error": "URL cannot be empty"
        }


    if not MODELS_LOADED:

        return {
            "error":
                "URL detection models "
                "could not be loaded"
        }


    # --------------------------------------------------------
    # Extract features
    # --------------------------------------------------------

    features = extract_url_features(url)


    # --------------------------------------------------------
    # Convert features to model input
    # --------------------------------------------------------

    try:

        X = pd.DataFrame(
    [
        [
            features[feature]
            for feature in FEATURE_NAMES
        ]
    ],
    columns=FEATURE_NAMES
)

    except KeyError as error:

        return {
            "error":
                f"Feature mismatch: {error}"
        }


    # ========================================================
    # INDIVIDUAL MODEL PREDICTIONS
    # ========================================================

    dt_probability = float(
        decision_tree.predict_proba(X)[0][1]
    )

    gb_probability = float(
        gradient_boosting.predict_proba(X)[0][1]
    )

    xgb_probability = float(
        xgboost_model.predict_proba(X)[0][1]
    )


    # --------------------------------------------------------
    # Individual predictions
    # --------------------------------------------------------

    dt_prediction = (
        "MALICIOUS"
        if dt_probability >= 0.5
        else "SAFE"
    )

    gb_prediction = (
        "MALICIOUS"
        if gb_probability >= 0.5
        else "SAFE"
    )

    xgb_prediction = (
        "MALICIOUS"
        if xgb_probability >= 0.5
        else "SAFE"
    )


    # ========================================================
    # ENSEMBLE
    # ========================================================

    probabilities = [
        dt_probability,
        gb_probability,
        xgb_probability
    ]


    ensemble_probability = float(
        np.mean(probabilities)
    )


    # --------------------------------------------------------
    # Majority voting
    # --------------------------------------------------------

    malicious_votes = sum(
        [
            dt_prediction == "MALICIOUS",
            gb_prediction == "MALICIOUS",
            xgb_prediction == "MALICIOUS"
        ]
    )


    if malicious_votes >= 2:

        final_prediction = "MALICIOUS"

    else:

        final_prediction = "SAFE"


    # ========================================================
    # RISK LEVEL
    # ========================================================

    if ensemble_probability >= 0.80:

        risk = "HIGH"

    elif ensemble_probability >= 0.50:

        risk = "MEDIUM"

    else:

        risk = "LOW"


    # ========================================================
    # RESULT
    # ========================================================

    result = {

        "url": url,

        "prediction":
            final_prediction,

        "risk":
            risk,

        "risk_score":
            round(
                ensemble_probability * 100,
                2
            ),

        "models": {

            "decision_tree":
                dt_prediction,

            "gradient_boosting":
                gb_prediction,

            "xgboost":
                xgb_prediction
        },

        "probabilities": {

            "decision_tree":
                round(
                    dt_probability * 100,
                    2
                ),

            "gradient_boosting":
                round(
                    gb_probability * 100,
                    2
                ),

            "xgboost":
                round(
                    xgb_probability * 100,
                    2
                )
        },

        "malicious_votes":
            malicious_votes,

        "total_models":
            3
    }


    return result


# ============================================================
# TEST WHEN RUN DIRECTLY
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 60)
    print("DEEPMOCK URL DETECTION TEST")
    print("=" * 60)


    test_urls = [

        "https://www.google.com",

        "https://www.microsoft.com",

        "http://secure-login-verify-account-example.com/login"

    ]


    for test_url in test_urls:

        print("\nURL:")
        print(test_url)

        result = predict_url(
            test_url
        )

        print("\nResult:")

        for key, value in result.items():

            print(
                f"{key}: {value}"
            )