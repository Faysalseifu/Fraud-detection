from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

PREPROCESSOR_CANDIDATES = [
    Path("data/processed/preprocessor_fraud.pkl"),
    Path("notebooks/models/preprocessor_fraud.pkl"),
]
MODEL_CANDIDATES = [
    Path("models/xgb_fraud_best.pkl"),
    Path("notebooks/models/xgb_fraud_best.pkl"),
]

DEFAULT_FIELDS = [
    "purchase_value",
    "age",
    "time_since_signup_hours",
    "hour_of_day",
    "day_of_week",
    "device_count",
    "ip_count",
    "country",
]

COUNTRY_OPTIONS = [
    "United States",
    "Nigeria",
    "United Kingdom",
    "Germany",
    "France",
    "Ethiopia",
    "Unknown",
    "Other",
]


def _resolve_path(candidates: list[Path]) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Missing required artifact. Checked: "
        + ", ".join(str(p) for p in candidates)
    )


@st.cache_resource
def _load_artifacts():
    preprocessor_path = _resolve_path(PREPROCESSOR_CANDIDATES)
    model_path = _resolve_path(MODEL_CANDIDATES)
    preprocessor = joblib.load(preprocessor_path)
    model = joblib.load(model_path)
    return preprocessor, model, preprocessor_path, model_path


def _get_schema(preprocessor):
    columns = []
    cat_cols = set()
    num_cols = set()

    if hasattr(preprocessor, "feature_names_in_"):
        columns = list(preprocessor.feature_names_in_)

    if hasattr(preprocessor, "transformers_"):
        for name, _transformer, cols in preprocessor.transformers_:
            if cols in ("drop", "passthrough"):
                continue
            if isinstance(cols, (list, tuple, np.ndarray, pd.Index)):
                col_list = list(cols)
            else:
                continue
            name_lower = str(name).lower()
            if name_lower.startswith("cat"):
                cat_cols.update(col_list)
            elif name_lower.startswith("num"):
                num_cols.update(col_list)

    if not columns:
        columns = DEFAULT_FIELDS.copy()

    return columns, cat_cols, num_cols


def _build_input_frame(columns, cat_cols, num_cols, overrides):
    data = {col: 0.0 for col in columns}
    for col in cat_cols:
        data[col] = "Unknown"

    for key, value in overrides.items():
        if key in data:
            data[key] = value

    return pd.DataFrame([data])


def _get_transformed_feature_names(preprocessor, X_transformed):
    if hasattr(preprocessor, "get_feature_names_out"):
        return preprocessor.get_feature_names_out().tolist()
    return [f"f{i}" for i in range(X_transformed.shape[1])]


def _row_array(row):
    if hasattr(row, "toarray"):
        return row.toarray()[0]
    return np.asarray(row).ravel()


st.set_page_config(page_title="Fraud Risk Detector", layout="wide")

st.title("Fraud Risk Detector")
st.markdown(
    "Enter a transaction to get a fraud risk score and an explanation based on SHAP."
)

with st.sidebar:
    st.header("Project Info")
    st.markdown(
        "This dashboard loads the saved preprocessor and XGBoost model to score a "
        "single transaction and explain the drivers of the prediction."
    )

    try:
        preprocessor, model, pre_path, model_path = _load_artifacts()
        st.caption(f"Preprocessor: {pre_path}")
        st.caption(f"Model: {model_path}")
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

columns, cat_cols, num_cols = _get_schema(preprocessor)

st.subheader("Transaction Input")
st.caption("Use the form or upload a single-row CSV/JSON file.")

with st.form("transaction_form"):
    left, right = st.columns(2)
    with left:
        purchase_value = st.number_input("Purchase Value", min_value=0.0, value=50.0)
        age = st.number_input("User Age", min_value=18, max_value=100, value=35)
        time_since_signup_hours = st.number_input(
            "Hours Since Signup", min_value=0.0, value=24.0
        )
        hour_of_day = st.slider("Hour of Day", 0, 23, 14)
        day_of_week = st.slider("Day of Week (0=Mon)", 0, 6, 2)
    with right:
        device_count = st.number_input("Users on Device", min_value=1, value=1)
        ip_count = st.number_input("Users on IP", min_value=1, value=1)
        country = st.selectbox("Country", COUNTRY_OPTIONS)

    submitted = st.form_submit_button("Check Risk")

uploaded_file = st.file_uploader("Upload a CSV or JSON file", type=["csv", "json"])

input_df = None

if uploaded_file is not None:
    try:
        if uploaded_file.name.lower().endswith(".json"):
            raw_df = pd.read_json(uploaded_file)
        else:
            raw_df = pd.read_csv(uploaded_file)
        if raw_df.empty:
            st.warning("Uploaded file has no rows.")
        else:
            input_df = raw_df.iloc[[0]].copy()
    except Exception as exc:
        st.error(f"Failed to read file: {exc}")

if submitted and input_df is None:
    overrides = {
        "purchase_value": purchase_value,
        "age": age,
        "time_since_signup_hours": time_since_signup_hours,
        "hour_of_day": hour_of_day,
        "day_of_week": day_of_week,
        "device_count": device_count,
        "ip_count": ip_count,
        "country": country,
    }
    input_df = _build_input_frame(columns, cat_cols, num_cols, overrides)

if input_df is not None:
    input_df = _build_input_frame(columns, cat_cols, num_cols, input_df.iloc[0].to_dict())

    X_transformed = preprocessor.transform(input_df)
    proba = model.predict_proba(X_transformed)[0][1]

    if proba > 0.7:
        risk_level = "High"
        risk_color = "red"
    elif proba > 0.3:
        risk_level = "Medium"
        risk_color = "orange"
    else:
        risk_level = "Low"
        risk_color = "green"

    st.subheader("Risk Assessment")
    st.metric("Fraud Probability", f"{proba:.1%}")
    st.markdown(
        f"Risk Level: <span style='color:{risk_color};'><strong>{risk_level}</strong></span>",
        unsafe_allow_html=True,
    )

    st.subheader("Why this prediction?")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_transformed)
    if isinstance(shap_values, list):
        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, np.ndarray)):
        expected_value = expected_value[1] if len(expected_value) > 1 else expected_value[0]

    feature_names = _get_transformed_feature_names(preprocessor, X_transformed)
    row = _row_array(X_transformed[0])

    fig, _ax = plt.subplots(figsize=(10, 2))
    shap.force_plot(
        expected_value,
        shap_values[0],
        row,
        feature_names=feature_names,
        matplotlib=True,
        show=False,
    )
    st.pyplot(fig, clear_figure=True)

    with st.expander("How to interpret this"):
        st.markdown(
            "The plot shows which features push the score higher (toward fraud) "
            "or lower (toward legitimate). Larger bars indicate stronger influence."
        )

else:
    st.info("Submit the form or upload a file to generate a prediction.")
