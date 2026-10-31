
"""Feature preparation for AI4I predictive-maintenance data."""
import numpy as np
import pandas as pd

TARGET = "Machine failure"

def prepare_features(df, training=True):
    data = df.copy()

    # Standardize dataset column names
    rename = {
        "Air temperature": "Air temperature [K]",
        "Process temperature": "Process temperature [K]",
        "Rotational speed": "Rotational speed [rpm]",
        "Torque": "Torque [Nm]",
        "Tool wear": "Tool wear [min]",
        "TWF": "TWF",
        "HDF": "HDF",
        "PWF": "PWF",
        "OSF": "OSF",
        "RNF": "RNF"
    }
    data = data.rename(columns=rename)

    # Feature engineering
    data["Temperature difference [K]"] = (
        data["Process temperature [K]"] -
        data["Air temperature [K]"]
    )

    data["Mechanical power [W]"] = (
        data["Torque [Nm]"] *
        data["Rotational speed [rpm]"] *
        (2 * np.pi / 60)
    )

    # Remove target-leakage and identifier columns
    remove = [
        "UDI", "Product ID", "TWF", "HDF",
        "PWF", "OSF", "RNF", TARGET
    ]

    X = data.drop(columns=remove, errors="ignore")

    # Convert product type into numerical columns
    if "Type" in X.columns:
        X = pd.get_dummies(X, columns=["Type"], dtype=int)

    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    if training:
        y = pd.to_numeric(data[TARGET]).astype(int)
        return X, y

    return X
