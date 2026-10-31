"""Create evaluation plots and prediction output files."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from src.train_model import train, ROOT

OUT = ROOT / "outputs"

def evaluate():
    OUT.mkdir(parents=True, exist_ok=True)
    model, X_test, y_test, pred, prob, metrics = train()
    cm = confusion_matrix(y_test, pred, labels=[0, 1])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal", "Failure"], yticklabels=["Normal", "Failure"])
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Confusion Matrix")
    plt.tight_layout(); plt.savefig(OUT / "confusion_matrix.png", dpi=160); plt.close()

    importance = pd.Series(model.feature_importances_, index=X_test.columns).sort_values()
    plt.figure(figsize=(9, 6))
    importance.tail(12).plot(kind="barh")
    plt.xlabel("Importance"); plt.title("Top Feature Importance")
    plt.tight_layout(); plt.savefig(OUT / "feature_importance.png", dpi=160); plt.close()

    counts = y_test.value_counts().sort_index()
    plt.figure(figsize=(6, 4))
    plt.bar(["Normal", "Failure"], [counts.get(0, 0), counts.get(1, 0)])
    plt.ylabel("Number of records"); plt.title("Failure Distribution in Test Set")
    plt.tight_layout(); plt.savefig(OUT / "failure_distribution.png", dpi=160); plt.close()

    pd.DataFrame({
        "actual_failure": y_test.to_numpy(),
        "predicted_failure": pred,
        "failure_probability": prob
    }).to_csv(OUT / "prediction_results.csv", index=False)
    print("\nClassification report:")
    print(classification_report(y_test, pred, zero_division=0))
    print(f"Plots and predictions saved in: {OUT}")
    return metrics

if __name__ == "__main__":
    evaluate()
