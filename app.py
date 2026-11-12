# Block 1: Imports and Page Configuration
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Import the exact feature engineering function used when training the pipeline
from features import add_engineered_features

st.set_page_config(
    page_title="AI4I 2020 Predictive Maintenance",
    page_icon="⚙️",
    layout="centered"
)

# Block 2: Load the Saved Pipeline with Caching
# @st.cache_resource loads the model once into memory instead of reloading on every button click
@st.cache_resource
def load_pipeline():
    model_path = os.path.join("models", "final_model.joblib")
    return joblib.load(model_path)

pipeline = load_pipeline()

# Block 3: Title and Engineer Instructions
st.title("⚙️ Milling Machine Failure Predictor")
st.markdown(
    """
    **Target Users:** Maintenance Engineers & Plant Reliability Supervisors  
    **Dataset:** UCI AI4I 2020 Predictive Maintenance Dataset  
    Enter the 6 real-time machine telemetry readings below to estimate breakdown risk and inspect physical stress indicators.
    """
)
st.divider()

# Block 4: User Input Controls (Bounded to AI4I 2020 Dataset Ranges)
st.subheader("1. Enter Machine Telemetry Readings")

col1, col2 = st.columns(2)

with col1:
    prod_type = st.selectbox(
        "Product Quality Variant (Type)",
        options=["L", "M", "H"],
        index=0,
        help="L = Low (60% of data), M = Medium (30%), H = High (10%)"
    )
    air_temp = st.number_input(
        "Air temperature [K]",
        min_value=290.0,
        max_value=310.0,
        value=300.0,
        step=0.1,
        help="Dataset range: 295.3 K to 304.5 K (Mean: 300.0 K)"
    )
    process_temp = st.number_input(
        "Process temperature [K]",
        min_value=300.0,
        max_value=320.0,
        value=310.0,
        step=0.1,
        help="Dataset range: 305.7 K to 313.8 K (Mean: 310.0 K)"
    )

with col2:
    rot_speed = st.number_input(
        "Rotational speed [rpm]",
        min_value=1000,
        max_value=3000,
        value=1503,
        step=10,
        help="Dataset range: 1168 rpm to 2886 rpm (Median: 1503 rpm)"
    )
    torque = st.number_input(
        "Torque [Nm]",
        min_value=1.0,
        max_value=90.0,
        value=40.0,
        step=0.5,
        help="Dataset range: 3.8 Nm to 76.6 Nm (Mean: 40.0 Nm)"
    )
    tool_wear = st.number_input(
        "Tool wear [min]",
        min_value=0,
        max_value=300,
        value=108,
        step=1,
        help="Dataset range: 0 min to 253 min (Median: 108 min)"
    )

# Optional Decision Threshold Slider (Demonstrates Recall vs Precision Trade-off)
threshold = st.slider(
    "Decision Probability Threshold (Lower = Higher Recall / Earlier Warning)",
    min_value=0.10,
    max_value=0.90,
    value=0.50,
    step=0.05
)

# Block 5: Prediction & Plain-Language Diagnostic Explanation
if st.button("Predict Failure Risk", type="primary", use_container_width=True):
    # Construct a 1-row DataFrame with the exact raw column names expected by the Pipeline
    input_df = pd.DataFrame([{
        "Type": prod_type,
        "Air temperature [K]": float(air_temp),
        "Process temperature [K]": float(process_temp),
        "Rotational speed [rpm]": int(rot_speed),
        "Torque [Nm]": float(torque),
        "Tool wear [min]": int(tool_wear)
    }])

    # Compute engineered features for display in the UI
    fe_df = add_engineered_features(input_df)
    temp_diff_val = fe_df["temp_diff"].iloc[0]
    power_val = fe_df["power"].iloc[0]
    overstrain_val = tool_wear * torque

    # Get model probability and apply threshold
    failure_prob = float(pipeline.predict_proba(input_df)[0, 1])
    pred_label = 1 if failure_prob >= threshold else 0

    st.divider()
    st.subheader("2. Prediction & Diagnostic Summary")

    # Display KPI Metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Failure Probability", f"{failure_prob * 100:.1f}%")
    m2.metric("Temp Difference (temp_diff)", f"{temp_diff_val:.2f} K")
    m3.metric("Mechanical Power (power)", f"{power_val:.0f} W")

    if pred_label == 1:
        st.error(
            f"⚠️ **ALERT: High Failure Risk Detected** "
            f"(Predicted Class = `1`, Probability = `{failure_prob*100:.1f}%` >= `{threshold*100:.0f}%` threshold)."
        )
    else:
        st.success(
            f"✅ **NORMAL: Machine Operating Within Safe Parameters** "
            f"(Predicted Class = `0`, Probability = `{failure_prob*100:.1f}%` < `{threshold*100:.0f}%` threshold)."
        )

    # Plain-language diagnostic notes based on physical operating zones
    reasons = []
    if temp_diff_val < 8.6 and rot_speed < 1380:
        reasons.append(
            f"**Heat Dissipation Risk:** `temp_diff` ({temp_diff_val:.2f} K) is below 8.6 K while spindle speed ({rot_speed} rpm) is below 1380 rpm."
        )
    if power_val < 3500 or power_val > 9000:
        reasons.append(
            f"**Mechanical Power Risk:** Calculated `power` ({power_val:.0f} W) is outside the standard 3500 W – 9000 W operating window."
        )
    osf_limits = {"L": 11000, "M": 12000, "H": 13000}
    if overstrain_val > osf_limits[prod_type]:
        reasons.append(
            f"**Overstrain Risk:** Tool wear × Torque ({overstrain_val:.0f} minNm) exceeds the Type `{prod_type}` limit of {osf_limits[prod_type]} minNm."
        )
    if tool_wear >= 200:
        reasons.append(
            f"**Tool Wear Risk:** `Tool wear` ({tool_wear} min) has entered the 200–240 min critical wear zone."
        )

    st.markdown("**Plain-Language Engineering Explanation:**")
    if reasons:
        for r in reasons:
            st.markdown(f"- {r}")
    else:
        st.markdown(
            "- All physical stress indicators (temperature gradient, mechanical power, tool wear, and overstrain product) are currently inside nominal operating bounds."
        )