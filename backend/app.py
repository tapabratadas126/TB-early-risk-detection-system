from flask import Flask, request, jsonify
import joblib
import numpy as np
import csv
import logging

# =========================
# App Setup
# =========================
app = Flask(__name__)

logging.basicConfig(level=logging.INFO)

# NOTE:
# The /predict response schema is frozen.
# Do NOT modify response keys without frontend update.

# =========================
# Load ML Model (Fail-safe)
# =========================
MODEL_PATH = "model/tb_risk_new_logic_model.joblib"

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError("Failed to load ML model") from e

# =========================
# Constants
# =========================
FEATURE_ORDER = [
    "symptom_1",
    "symptom_2",
    "symptom_3",
    "symptom_4",
    "symptom_5",
    "symptom_6",
    "symptom_7",
    "symptom_8",
    "symptom_9",
    "symptom_10",
    "symptom_11",
    "symptom_12"
]

RISK_MAP = {
    0: "Low",
    1: "Medium",
    2: "High"
}

DISCLAIMER = (
    "This system provides early TB risk screening only. "
    "It does not diagnose tuberculosis. "
    "Please consult a qualified healthcare professional."
)

# =========================
# Helpers
# =========================
def clean_text(value, max_len=50):
    if not isinstance(value, str):
        return ""
    return value.strip().lower()[:max_len]

# =========================
# Load Hospital Dataset
# =========================
HOSPITALS = []

with open("data/hospitals.csv", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        HOSPITALS.append({
            "hospital_name": row["hospital_name"],
            "district": row["district"].strip().lower(),
            "state": row["state"].strip().lower()
        })

def recommend_hospitals(district, state, limit=5):
    district_matches = [h for h in HOSPITALS if h["district"] == district]
    if district_matches:
        return district_matches[:limit]

    state_matches = [h for h in HOSPITALS if h["state"] == state]
    return state_matches[:limit]

# =========================
# Routes
# =========================
@app.route("/")
def home():
    return {"status": "Backend running"}

@app.route("/health")
def health():
    return {"status": "ok"}

@app.route("/predict", methods=["POST"])
def predict():
    logging.info("Predict endpoint hit")

    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400

    try:
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"error": "Invalid or empty JSON"}), 400

        # Validate symptom inputs
        features = []
        for key in FEATURE_ORDER:
            if key not in data:
                return jsonify({"error": f"Missing feature: {key}"}), 400
            if data[key] not in (0, 1):
                return jsonify({"error": f"{key} must be 0 or 1"}), 400
            features.append(data[key])

        X = np.array(features).reshape(1, -1)

        pred_class = int(model.predict(X)[0])
        proba = model.predict_proba(X)[0]
        confidence = float(np.max(proba) * 100)

        risk_level = RISK_MAP[pred_class]

        # Location-based hospital recommendation
        district = clean_text(data.get("district", ""))
        state = clean_text(data.get("state", ""))

        hospitals = []
        if risk_level in ("Medium", "High") and district and state:
            hospitals = recommend_hospitals(district, state)

        return jsonify({
            "risk_level": risk_level,
            "confidence_percent": round(confidence, 2),
            "hospitals": hospitals,
            "disclaimer": DISCLAIMER
        })

    except Exception:
        logging.exception("Unhandled server error")
        return jsonify({
            "error": "Server error. Please try again later.",
            "disclaimer": DISCLAIMER
        }), 500

# =========================
# Run App
# =========================
if __name__ == "__main__":
    app.run(debug=False)

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
