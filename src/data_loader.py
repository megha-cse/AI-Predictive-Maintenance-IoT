"""Download and load the UCI AI4I 2020 Predictive Maintenance dataset."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
DATA_PATH = RAW_DIR / "ai4i2020.csv"

def load_data(download_if_missing=True):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    if not download_if_missing:
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    try:
        from ucimlrepo import fetch_ucirepo
        dataset = fetch_ucirepo(id=601)
        X = dataset.data.features.copy()
        y = dataset.data.targets.copy()
        df = pd.concat([X, y], axis=1)
        df.to_csv(DATA_PATH, index=False)
        print(f"Dataset downloaded and saved to {DATA_PATH}")
        return df
    except Exception as exc:
        raise RuntimeError(
            "Could not download the dataset. Check internet access, then run "
            "'python main.py' again. You can also download the UCI AI4I 2020 "
            "CSV manually and save it as data/raw/ai4i2020.csv."
        ) from exc
