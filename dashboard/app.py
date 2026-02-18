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

RULE_FORCE_SCORE = 0.92
RULE_BOOST = 0.08

HIGH_RISK_COUNTRIES = {
    "Nigeria",
    "Ethiopia",
}


def _resolve_path(candidates: list[Path]) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Missing required artifact. Checked: " + ", ".join(str(p) for p in candidates)
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


def _clean_input_frame(df, cat_cols, num_cols):
    cleaned = df.copy()
    for col in num_cols:
        if col in cleaned.columns:
            cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce").fillna(0.0)
    for col in cat_cols:
        if col in cleaned.columns:
            cleaned[col] = cleaned[col].fillna("Unknown")

    for col in ("device_count", "ip_count"):
        if col in cleaned.columns:
            cleaned[col] = cleaned[col].clip(lower=1)
    if "time_since_signup_hours" in cleaned.columns:
        cleaned["time_since_signup_hours"] = cleaned["time_since_signup_hours"].clip(
            lower=0.0
        )

    return cleaned


def _get_transformed_feature_names(preprocessor, X_transformed):
    if hasattr(preprocessor, "get_feature_names_out"):
        return preprocessor.get_feature_names_out().tolist()
    return [f"f{i}" for i in range(X_transformed.shape[1])]


def _row_array(row):
    if hasattr(row, "toarray"):
        return row.toarray()[0]
    return np.asarray(row).ravel()


st.set_page_config(page_title="Fraud Risk Detector", layout="wide")

st.markdown(
    """
    <style>
    .app-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1d4ed8;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #334155;
        margin-bottom: 1rem;
    }
    .summary-card {
        border-radius: 12px;
        padding: 0.9rem 1rem;
        margin: 0.4rem 0;
        border: 1px solid #dbeafe;
        background: #eff6ff;
        color: #1e3a8a;
    }
    .badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.85rem;
        margin-left: 0.35rem;
    }
    .badge-low {
        background: #dcfce7;
        color: #166534;
    }
    .badge-medium {
        background: #fef3c7;
        color: #92400e;
    }
    .badge-high {
        background: #fee2e2;
        color: #991b1b;
    }
    .section-label {
        color: #475569;
        font-size: 0.95rem;
        margin-top: 0.2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='app-title'>Fraud Risk Detector</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>Enter a transaction to get a fraud risk score and a SHAP explanation.</div>",
    unsafe_allow_html=True,
)
st.caption(
    "Use manual entry or upload a single-row file. Business rule toggles in the sidebar "
    "can simulate stricter fraud controls."
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

    st.divider()
    st.subheader("Model Performance")
    st.metric("Fraud Recall", "87%+")
    st.metric("Precision", "89%+")
    st.metric("PR-AUC", "0.91+")

    st.divider()
    st.subheader("Rule Toggles")
    rule_fast_signup = st.checkbox(
        "Force high risk if time_since_signup_hours < 2",
        value=False,
    )
    rule_many_devices = st.checkbox(
        "Force high risk if device_count > 3",
        value=False,
    )
    rule_high_risk_country = st.checkbox(
        "Force high risk if country is high-risk",
        value=False,
    )

columns, cat_cols, num_cols = _get_schema(preprocessor)

st.subheader("Transaction Input")
st.caption("Choose a mode below, then score the transaction.")

manual_tab, upload_tab = st.tabs(["Manual Entry", "File Upload"])

with manual_tab:
    with st.form("transaction_form"):
        left, right = st.columns(2)
        with left:
            purchase_value = st.number_input(
                "Purchase Value", min_value=0.0, value=50.0
            )
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

        submitted = st.form_submit_button("Check Risk", use_container_width=True)

with upload_tab:
    uploaded_file = st.file_uploader(
        "Upload a single-row CSV or JSON file", type=["csv", "json"]
    )

if "uploaded_file" not in locals():
    uploaded_file = None

input_df = None
input_source = None
raw_upload = None

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
            raw_upload = input_df.iloc[0].to_dict()
            input_source = "upload"
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
    input_source = "form"

if input_df is not None:
    input_df = _build_input_frame(
        columns, cat_cols, num_cols, input_df.iloc[0].to_dict()
    )
    input_df = _clean_input_frame(input_df, cat_cols, num_cols)

    if "country" in input_df.columns:
        input_df["country"] = input_df["country"].where(
            input_df["country"].isin(COUNTRY_OPTIONS), "Other"
        )

    if input_source == "upload" and raw_upload is not None:
        required_fields = [
            "purchase_value",
            "age",
            "time_since_signup_hours",
            "device_count",
            "ip_count",
            "country",
        ]
        missing_fields = [
            field
            for field in required_fields
            if field in raw_upload
            and (raw_upload[field] is None or str(raw_upload[field]).strip() == "")
        ]
        if missing_fields:
            st.error(f"Required fields missing values: {', '.join(missing_fields)}")
            st.stop()

    with st.spinner("Scoring transaction..."):
        X_transformed = preprocessor.transform(input_df)
        proba = model.predict_proba(X_transformed)[0][1]

    rule_hits = []
    if rule_fast_signup and input_df.loc[0, "time_since_signup_hours"] < 2:
        rule_hits.append("fast signup")
    if rule_many_devices and input_df.loc[0, "device_count"] > 3:
        rule_hits.append("many devices")
    if rule_high_risk_country and input_df.loc[0, "country"] in HIGH_RISK_COUNTRIES:
        rule_hits.append("high-risk country")

    adjusted_proba = proba
    if rule_hits:
        adjusted_proba = min(
            1.0, max(RULE_FORCE_SCORE, proba + RULE_BOOST * len(rule_hits))
        )

    if adjusted_proba > 0.7:
        risk_level = "High"
        risk_badge_class = "badge-high"
    elif adjusted_proba > 0.3:
        risk_level = "Medium"
        risk_badge_class = "badge-medium"
    else:
        risk_level = "Low"
        risk_badge_class = "badge-low"

    st.subheader("Risk Assessment")
    metric_left, metric_mid, metric_right = st.columns(3)
    metric_left.metric("Model Probability", f"{proba:.1%}")
    metric_mid.metric("Adjusted Probability", f"{adjusted_proba:.1%}")
    metric_right.metric("Risk Level", risk_level)

    st.markdown(
        f"""
        <div class='summary-card'>
            Final Decision
            <span class='badge {risk_badge_class}'>{risk_level} Risk</span>
            <div class='section-label'>Adjusted probability after business rules: {adjusted_proba:.1%}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(float(adjusted_proba))

    if risk_level == "High":
        st.error("High risk: escalate review and block if needed.")
    elif risk_level == "Medium":
        st.warning("Medium risk: verify with extra checks.")
    else:
        st.success("Low risk: approve with standard monitoring.")

    st.caption("Thresholds: Low <= 30% | Medium 30-70% | High > 70%")
    if rule_hits:
        st.info(f"Rule overrides applied: {', '.join(rule_hits)}")

    with st.expander("Input snapshot used for scoring"):
        st.dataframe(input_df.T, use_container_width=True)

    st.divider()

    st.subheader("Why this prediction?")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_transformed)
    if isinstance(shap_values, list):
        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, np.ndarray)):
        expected_value = (
            expected_value[1] if len(expected_value) > 1 else expected_value[0]
        )

    feature_names = _get_transformed_feature_names(preprocessor, X_transformed)
    row = _row_array(X_transformed[0])

    st.markdown("**Waterfall view (recommended)**")
    st.caption("Red = pushes toward fraud, Blue = pushes toward legitimate.")
    explanation = shap.Explanation(
        values=shap_values[0],
        base_values=expected_value,
        data=row,
        feature_names=feature_names,
    )
    fig, _ax = plt.subplots(figsize=(10, 4))
    try:
        shap.plots.waterfall(explanation, max_display=12, show=False)
        st.pyplot(fig, clear_figure=True)
    except Exception as exc:
        st.warning(f"Waterfall plot failed: {exc}")
        shap.plots.bar(explanation, max_display=12, show=False)
        st.pyplot(fig, clear_figure=True)

    with st.expander("Force view (optional)"):
        force_plot = shap.force_plot(
            expected_value,
            shap_values[0],
            row,
            feature_names=feature_names,
        )
        force_html = f"<head>{shap.getjs()}</head>{force_plot.html()}"
        st.components.v1.html(force_html, height=220)

    with st.expander("How to interpret this"):
        st.markdown(
            "The plot shows which features push the score higher (toward fraud) "
            "or lower (toward legitimate). Larger bars indicate stronger influence."
        )

else:
    st.info("Submit the form or upload a file to generate a prediction.")
