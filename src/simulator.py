"""Simulate incoming sensor readings using rows from the public dataset."""
import numpy as np
import pandas as pd
from src.data_loader import load_data

def simulate_sensor_stream(number_of_readings=10, seed=42):
    df = load_data()
    sensor_columns = [
        "Air temperature [K]", "Process temperature [K]",
        "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"
    ]
    rng = np.random.default_rng(seed)
    sample = df.sample(n=min(number_of_readings, len(df)), random_state=seed).copy()
    for col in sensor_columns:
        if col in sample:
            sample[col] = sample[col].astype(float) + rng.normal(0, 0.2, len(sample))
    return sample.reset_index(drop=True)
