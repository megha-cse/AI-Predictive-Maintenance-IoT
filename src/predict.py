"""Load the saved model and predict failure risk for a machine reading."""
from pathlib import Path
import joblib
import pandas as pd
from src.preprocessing import prepare_features

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "predictive_maintenance_model.pkl"

def predict_reading(reading):
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model missing. Run 'python main.py' first.")
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    columns = bundle["feature_columns"]
    row = pd.DataFrame([reading])
    X = prepare_features(row, training=False).reindex(columns=columns, fill_value=0)
    probability = float(model.predict_proba(X)[0, 1])
    return {
        "failure_probability": probability,
        "prediction": int(probability >= 0.5),
        "status": "HIGH RISK" if probability >= 0.5 else "NORMAL"
    }

if __name__ == "__main__":
    sample = {
        "Type": "M", "Air temperature [K]": 300.0,
        "Process temperature [K]": 310.0, "Rotational speed [rpm]": 1500,
        "Torque [Nm]": 40.0, "Tool wear [min]": 120
    }
    print(predict_reading(sample))
