"""
WikiArt Painting Classifier — Web App

Flask server that loads a trained model and predicts the artist
from uploaded painting images.

Usage:
    python app/server.py
    python app/server.py --model results/models/vit.keras --port 5000
"""

import argparse
import io
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Add project root to path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, PROJECT_ROOT)

from flask import Flask, jsonify, request, send_from_directory

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

IMG_SIZE = (224, 224)
ARTIST_NAMES = [
    "Albrecht_Durer", "Boris_Kustodiev", "Camille_Pissarro", "Childe_Hassam",
    "Claude_Monet", "Edgar_Degas", "Eugene_Boudin", "Gustave_Dore",
    "Ilya_Repin", "Ivan_Aivazovsky", "Ivan_Shishkin", "John_Singer_Sargent",
    "Marc_Chagall", "Martiros_Saryan", "Nicholas_Roerich", "Pablo_Picasso",
    "Paul_Cezanne", "Pierre_Auguste_Renoir", "Pyotr_Konchalovsky",
    "Raphael_Kirchner", "Rembrandt", "Salvador_Dali", "Vincent_van_Gogh",
]

# Readable display names
DISPLAY_NAMES = {name: name.replace("_", " ") for name in ARTIST_NAMES}

MODELS_DIR = os.path.join(PROJECT_ROOT, "results", "models")

app = Flask(__name__, static_folder="static")
model = None
current_model_name = None


# ---------------------------------------------------------------------------
# Model loading & prediction
# ---------------------------------------------------------------------------

def list_available_models():
    """Scan results/models/ for .keras files and return a list of model names."""
    models = []
    if os.path.isdir(MODELS_DIR):
        for f in sorted(os.listdir(MODELS_DIR)):
            if f.endswith(".keras"):
                models.append(f)
    return models


def load_model(model_path):
    global model, current_model_name
    print(f"Loading model from {model_path}...")
    tf.keras.backend.clear_session()

    # Register custom objects needed to load saved models:
    # - transfer model: Lambda wrapping resnet50.preprocess_input
    # - vit model: CosineWarmup LR schedule (registered via import)
    from src.models.vit import CosineWarmup

    custom_objects = {
        "preprocess_input": tf.keras.applications.resnet50.preprocess_input,
        "CosineWarmup": CosineWarmup,
    }
    with tf.keras.utils.custom_object_scope(custom_objects):
        model = tf.keras.models.load_model(model_path)

    current_model_name = os.path.basename(model_path)
    print(f"Model loaded — {current_model_name} ({model.count_params():,} parameters)")


def preprocess_image(image_bytes):
    """Load image from bytes, resize to 224x224, normalise to [0, 1]."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE, Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr


def predict_artist(img_array):
    """Run prediction and return top-5 results."""
    batch = np.expand_dims(img_array, axis=0)
    probs = model.predict(batch, verbose=0)[0]

    top_indices = np.argsort(probs)[::-1][:5]
    results = []
    for idx in top_indices:
        results.append({
            "artist": DISPLAY_NAMES[ARTIST_NAMES[idx]],
            "artist_id": ARTIST_NAMES[idx],
            "confidence": float(probs[idx]),
        })
    return results


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "images" not in request.files:
        return jsonify({"error": "No images uploaded"}), 400

    files = request.files.getlist("images")
    if not files:
        return jsonify({"error": "No images uploaded"}), 400

    all_predictions = []
    for f in files:
        try:
            image_bytes = f.read()
            img_array = preprocess_image(image_bytes)
            results = predict_artist(img_array)
            all_predictions.append({
                "filename": f.filename,
                "predictions": results,
            })
        except Exception as e:
            all_predictions.append({
                "filename": f.filename,
                "error": str(e),
            })

    return jsonify({"results": all_predictions})


@app.route("/artists")
def artists():
    return jsonify({"artists": [DISPLAY_NAMES[a] for a in ARTIST_NAMES]})


@app.route("/models")
def models_list():
    available = list_available_models()
    return jsonify({
        "models": available,
        "current": current_model_name,
    })


@app.route("/models/switch", methods=["POST"])
def switch_model():
    data = request.get_json()
    if not data or "model" not in data:
        return jsonify({"error": "Missing 'model' field"}), 400

    filename = data["model"]
    available = list_available_models()
    if filename not in available:
        return jsonify({"error": f"Model '{filename}' not found"}), 404

    model_path = os.path.join(MODELS_DIR, filename)
    try:
        load_model(model_path)
        return jsonify({"current": current_model_name, "params": model.count_params()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="WikiArt Classifier Web App")
    parser.add_argument(
        "--model", type=str,
        default=os.path.join(PROJECT_ROOT, "results", "models", "vit.keras"),
        help="Path to .keras model checkpoint",
    )
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    args = parser.parse_args()

    load_model(args.model)
    print(f"\nStarting server at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
