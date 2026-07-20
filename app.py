import pickle
import numpy as np
import pandas as pd
import statsmodels.api as sm
import streamlit as st

st.set_page_config(page_title="Healthy Meals Churn Predictor", page_icon="📊", layout="centered")

MODEL_FILE = "churn_probit_model.pkl"
TRANSFORM_FILE = "churn_probit_transformations.pkl"

@st.cache_resource
def load_artifacts():
    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)
    with open(TRANSFORM_FILE, "rb") as f:
        transformations = pickle.load(f)
    return model, transformations

model, transformations = load_artifacts()
scaler = transformations["scaler"]
income_mapping = transformations["income_level_mapping"]

st.title("Healthy Meals Renewal Probability Predictor")
st.write(
    "Enter customer demographic and prior-year activity information to estimate "
    "the probability that the customer will renew their Healthy Meals subscription."
)

st.subheader("Customer demographics")
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=35, step=1)
    education = st.selectbox(
        "Education",
        ["Graduate", "High School", "Other", "Post-Graduate"],
    )
    income_level = st.selectbox(
        "Income Level",
        ["Low", "Medium", "High", "Very High"],
    )
with col2:
    device_type = st.selectbox(
        "Device Type",
        ["Desktop-only", "Mobile-only", "Multi-device"],
    )
    tech_comfort_score = st.number_input(
        "Tech Comfort Score", min_value=1, max_value=10, value=5, step=1
    )

st.subheader("Prior-year Healthy Meals activity")
col3, col4 = st.columns(2)
with col3:
    total_num_sessions = st.number_input(
        "Total Number of Sessions", min_value=0.0, value=47.0, step=1.0
    )
    active_days = st.number_input(
        "Active Days", min_value=0.0, value=2.0, step=1.0
    )
with col4:
    gross_total_session_length = st.number_input(
        "Gross Total Session Length", min_value=0.0, value=2049.0, step=1.0,
        help="Use the same session-length units used in the training data."
    )
    active_quarters = st.number_input(
        "Active Quarters", min_value=0, max_value=4, value=2, step=1
    )

sessions_per_active_quarter = (
    total_num_sessions / active_quarters if active_quarters > 0 else 0.0
)
st.caption(
    f"Sessions per active quarter is calculated automatically: "
    f"{sessions_per_active_quarter:.2f}"
)

if st.button("Predict Renewal Probability", type="primary"):
    # Match pd.get_dummies(..., drop_first=True) used during training.
    education_high_school = int(education == "High School")
    education_other = int(education == "Other")
    education_post_graduate = int(education == "Post-Graduate")
    device_mobile = int(device_type == "Mobile-only")
    device_multi = int(device_type == "Multi-device")
    income_ordinal = income_mapping[income_level]

    # The saved StandardScaler was fitted on these exact 12 columns.
    # ACTIVE_QUARTERS and SUBSCRIPTION_DURATION were later excluded from the
    # final Probit model, but placeholder values are still supplied so the
    # saved scaler can reproduce the training transformation exactly.
    scaler_input = pd.DataFrame([{
        "TOTAL_NUM_SESSIONS": float(total_num_sessions),
        "GROSS_TOTAL_SESSION_LENGTH": float(gross_total_session_length),
        "ACTIVE_DAYS": float(active_days),
        "ACTIVE_QUARTERS": float(active_quarters),
        "SESSIONS_PER_ACTIVE_QUARTER": float(sessions_per_active_quarter),
        "EDUCATION_High School": education_high_school,
        "EDUCATION_Other": education_other,
        "EDUCATION_Post-Graduate": education_post_graduate,
        "DEVICE_TYPE_Mobile-only": device_mobile,
        "DEVICE_TYPE_Multi-device": device_multi,
        "INCOME_LEVEL_ORDINAL": income_ordinal,
        "SUBSCRIPTION_DURATION": 365.0,
    }], columns=scaler.feature_names_in_)

    scaled_array = scaler.transform(scaler_input)
    scaled_df = pd.DataFrame(
        scaled_array,
        columns=scaler.feature_names_in_,
        index=scaler_input.index,
    )

    # AGE and TECH_COMFORT_SCORE were not included in the fitted scaler in the
    # saved training pipeline, so they enter the final model in their raw form.
    model_input = pd.DataFrame([{
        "TOTAL_NUM_SESSIONS": scaled_df.loc[0, "TOTAL_NUM_SESSIONS"],
        "GROSS_TOTAL_SESSION_LENGTH": scaled_df.loc[0, "GROSS_TOTAL_SESSION_LENGTH"],
        "ACTIVE_DAYS": scaled_df.loc[0, "ACTIVE_DAYS"],
        "SESSIONS_PER_ACTIVE_QUARTER": scaled_df.loc[0, "SESSIONS_PER_ACTIVE_QUARTER"],
        "AGE": float(age),
        "TECH_COMFORT_SCORE": float(tech_comfort_score),
        "EDUCATION_High School": scaled_df.loc[0, "EDUCATION_High School"],
        "EDUCATION_Other": scaled_df.loc[0, "EDUCATION_Other"],
        "EDUCATION_Post-Graduate": scaled_df.loc[0, "EDUCATION_Post-Graduate"],
        "DEVICE_TYPE_Mobile-only": scaled_df.loc[0, "DEVICE_TYPE_Mobile-only"],
        "DEVICE_TYPE_Multi-device": scaled_df.loc[0, "DEVICE_TYPE_Multi-device"],
        "INCOME_LEVEL_ORDINAL": scaled_df.loc[0, "INCOME_LEVEL_ORDINAL"],
    }])

    # Add intercept and force the exact feature order expected by statsmodels.
    model_input = sm.add_constant(model_input, has_constant="add").astype(np.float64)
    expected_columns = model.params.index.tolist()
    model_input = model_input[expected_columns]

    renewal_probability = float(model.predict(model_input).iloc[0])
    churn_probability = 1.0 - renewal_probability

    if renewal_probability >= 0.60:
        churn_risk = "Low"
    elif renewal_probability >= 0.40:
        churn_risk = "Medium"
    else:
        churn_risk = "High"

    st.divider()
    metric1, metric2 = st.columns(2)
    metric1.metric("Renewal Probability", f"{renewal_probability:.1%}")
    metric2.metric("Churn Probability", f"{churn_probability:.1%}")

    if churn_risk == "High":
        st.error(f"Churn Risk: {churn_risk}")
    elif churn_risk == "Medium":
        st.warning(f"Churn Risk: {churn_risk}")
    else:
        st.success(f"Churn Risk: {churn_risk}")

    st.caption(
        "This prediction is generated by the Probit model trained in Assignment 2 Part 1. "
        "The displayed risk bands are presentation thresholds, not separate model outputs."
    )
