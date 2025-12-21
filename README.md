# Improved Detection of Fraud for E‑Commerce and Bank Transactions

This project builds an end‑to‑end fraud detection pipeline for two real‑world datasets: an e‑commerce clickstream/transactions dataset (Fraud_Data) and the popular credit card transactions dataset (creditcard). It covers data understanding, cleaning, feature engineering (including geolocation), model training, evaluation, and explainability.

## Business Overview — What’s Happening
- Fraud causes direct financial loss and customer friction. The goal is to maximize fraud catch rate while minimizing false positives that block good users.
- We analyze behavioral and transactional signals — e.g., signup→purchase time, device/IP velocity, country risk, and purchase patterns — to build robust models that generalize to new fraud tactics.
- Primary success metric is PR‑AUC (Precision‑Recall AUC) due to extreme class imbalance; F1 and confusion matrix provide operational insight.

## Solution Overview
- Data Cleaning and EDA: Validate schema, quantify imbalance, visualize risk patterns and feature distributions.
- Feature Engineering: Time since signup, hour/day features, IP‑to‑country mapping via range join, velocity features per device/IP.
- Modeling: Baselines and ensembles (Logistic Regression, XGBoost) with stratified splits, class weights/`scale_pos_weight`, and optional SMOTE on the training set only.
- Explainability: SHAP for global and local explanations to support trust and operations.

## Language and Tech Stack
- Language: Python (3.10+ recommended)
- Core: pandas, numpy, scikit‑learn, xgboost, imbalanced‑learn, matplotlib, seaborn, shap, jupyter
- Testing: pytest

## Folder Structure
```
README.md
requirements.txt
data/
	raw/
		creditcard.csv
		Fraud_Data.csv
		IpAddress_to_Country.csv
	processed/
		creditcard_processed.csv
		fraud_features.csv
		fraud_processed.csv
models/
notebooks/
	eda-creditcard.ipynb
	eda-fraud-data.ipynb
	feature-engineering.ipynb
	modeling.ipynb
	shap-explainability.ipynb
	README.md
scripts/
src/
	feature_utils.py
tests/
	test_feature_utils.py
reports/
	interim1_report.md
```

## Setup and Installation (Windows)
1) Create and activate a virtual environment:
```bash
python -m venv .venv
.venv\Scripts\activate
```
2) Install dependencies:
```bash
pip install -r requirements.txt
```

## Data
- Place raw CSVs under `data/raw/` (see structure above).
- Processed artifacts and engineered features are saved under `data/processed/` by the notebooks.

## Workflow (Notebooks Order)
1. EDA (Fraud_Data): `notebooks/eda-fraud-data.ipynb` — schema checks, distributions, country risk.
2. EDA (creditcard): `notebooks/eda-creditcard.ipynb` — distributions, imbalance quantification.
3. Feature Engineering: `notebooks/feature-engineering.ipynb` — time features, IP→country mapping, velocity.
4. Modeling: `notebooks/modeling.ipynb` — stratified splits, class weights/SMOTE (train only), PR‑AUC/F1 evaluation.
5. Explainability: `notebooks/shap-explainability.ipynb` — SHAP plots for chosen model(s).

## Key Features Implemented
- `time_since_signup` (hours): fast signup→purchase is often suspicious.
- Time‑based: `hour_of_day`, `day_of_week`.
- IP→country: range merge using `IpAddress_to_Country.csv` to derive country risk.
- Velocity: rolling counts per device/IP within 1h/24h windows.

## Testing
- Unit tests for feature utilities:
```bash
pytest -q
```
- See `tests/test_feature_utils.py`.

## Model Artifacts
- Trained models and checkpoints saved under `models/` (e.g., `xgb_fraud_best.pkl`, `xgb_credit_best.pkl`).

## Reports
- Interim‑1 report: `reports/interim1_report.md` summarizes cleaning, EDA insights, features, and imbalance strategy.

## Contributing and Branches
- Default branch: `main`. Active work branch may be `task-2`.
- PRs and issues welcome. Keep notebooks and data paths consistent.

## Getting Help
- Ensure virtualenv is activated before running notebooks.
- If GPU is available for XGBoost, configure accordingly; otherwise CPU defaults are fine.

