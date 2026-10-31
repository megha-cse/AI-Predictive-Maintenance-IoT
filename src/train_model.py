"""Train and save a Random Forest failure classifier."""
from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from src.data_loader import load_data
from src.preprocessing import prepare_features

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "predictive_maintenance_model.pkl"

def train():
    df = load_data()
    X, y = prepare_features(df, training=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    model = RandomForestClassifier(
        n_estimators=300, max_depth=14, min_samples_split=4,
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities) if y_test.nunique() > 1 else float("nan"),
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_columns": list(X.columns)}, MODEL_PATH)
    print(f"Model saved: {MODEL_PATH}")
    print("Evaluation metrics:")
    for key, value in metrics.items():
        print(f"{key:>10}: {value:.4f}")
    return model, X_test, y_test, predictions, probabilities, metrics

if __name__ == "__main__":
    train()
