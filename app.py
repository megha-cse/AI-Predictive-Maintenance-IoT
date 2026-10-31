"""Streamlit dashboard for predictive maintenance."""
import streamlit as st
import pandas as pd
from pathlib import Path
from src.predict import predict_reading
from src.data_loader import load_data

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Predictive Maintenance AI", page_icon="⚙️", layout="wide")
st.title("⚙️ AI-Powered Predictive Maintenance System")
st.caption("Educational prototype using the public UCI AI4I 2020 dataset. No physical IoT hardware is connected.")

try:
    data = load_data()
    c1, c2, c3 = st.columns(3)
    c1.metric("Dataset records", f"{len(data):,}")
    c2.metric("Machine features", "5 sensor readings")
    c3.metric("Target", "Machine failure")
    st.subheader("Dataset preview")
    st.dataframe(data.head(10), use_container_width=True)
except Exception as exc:
    st.error(str(exc))
    st.stop()

st.divider()
st.subheader("Machine sensor simulator")
st.write("Enter a set of machine operating readings to estimate failure risk.")
with st.form("prediction_form"):
    a, b = st.columns(2)
    machine_type = a.selectbox("Product type", ["L", "M", "H"], index=1)
    air = a.number_input("Air temperature (K)", 295.0, 305.0, 300.0)
    process = a.number_input("Process temperature (K)", 300.0, 315.0, 310.0)
    rpm = b.number_input("Rotational speed (rpm)", 1000, 3000, 1500)
    torque = b.number_input("Torque (Nm)", 3.0, 80.0, 40.0)
    wear = b.number_input("Tool wear (min)", 0, 300, 120)
    submitted = st.form_submit_button("Predict machine condition", type="primary")

if submitted:
    reading = {
        "Type": machine_type, "Air temperature [K]": air,
        "Process temperature [K]": process, "Rotational speed [rpm]": rpm,
        "Torque [Nm]": torque, "Tool wear [min]": wear
    }
    try:
        result = predict_reading(reading)
        st.metric("Estimated failure probability", f"{result['failure_probability']:.1%}")
        if result["prediction"] == 1:
            st.error("HIGH RISK — schedule a human inspection.")
        else:
            st.success("NORMAL — no failure flagged by this model.")
        st.caption("This is a model estimate, not a safety guarantee or an automatic shutdown instruction.")
    except FileNotFoundError:
        st.warning("Model not trained yet. Run `python main.py` in the project terminal first.")
    except Exception as exc:
        st.error(str(exc))

st.divider()
st.subheader("Generated evaluation charts")
for filename, title in [
    ("failure_distribution.png", "Failure distribution"),
    ("confusion_matrix.png", "Confusion matrix"),
    ("feature_importance.png", "Feature importance")
]:
    path = ROOT / "outputs" / filename
    if path.exists():
        st.image(str(path), caption=title, use_container_width=True)
    else:
        st.info(f"{filename} will appear after you run `python main.py`.")
