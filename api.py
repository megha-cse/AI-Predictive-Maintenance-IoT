"""Simple Flask REST API for machine-failure prediction."""
from flask import Flask, request, jsonify
from src.predict import predict_reading

app = Flask(__name__)

@app.get("/")
def home():
    return jsonify({
        "project": "AI Predictive Maintenance IoT",
        "message": "Send a POST request to /predict with machine sensor values."
    })

@app.post("/predict")
def predict():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Send a JSON object containing sensor readings."}), 400
    try:
        return jsonify(predict_reading(payload))
    except (ValueError, KeyError, FileNotFoundError) as exc:
        return jsonify({"error": str(exc)}), 400

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
