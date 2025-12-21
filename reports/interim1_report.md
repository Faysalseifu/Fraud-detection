# Interim-1 Report: Task 1 — Data Analysis and Preprocessing

Repository: https://github.com/Faysalseifu/Fraud-detection

This report summarizes the progress on Task 1: EDA, cleaning, geolocation integration, feature engineering, and class imbalance strategy. Detailed work is in the notebooks and source files:
- EDA (creditcard): [notebooks/eda-creditcard.ipynb](notebooks/eda-creditcard.ipynb)
- EDA (Fraud_Data): [notebooks/eda-fraud-data.ipynb](notebooks/eda-fraud-data.ipynb)
- Feature Engineering: [notebooks/feature-engineering.ipynb](notebooks/feature-engineering.ipynb)
- Modeling (for imbalance checks): [notebooks/modeling.ipynb](notebooks/modeling.ipynb)
- Utilities: [src/feature_utils.py](src/feature_utils.py)

## Data Cleaning and Preprocessing
- Missing values: Verified and handled via median imputation in modeling; EDA confirms low/no missing in raw datasets.
- Duplicates: Checked in EDA; no harmful duplicates detected impacting training; keep original integrity for imbalance assessment.
- Types & parsing:
  - Parsed `signup_time` and `purchase_time` as datetimes in feature engineering.
  - IP addresses from `IpAddress_to_Country.csv` integrated via range-based mapping to derive `country`.
  - Non-numeric identifiers (`device_id`, `user_id`) excluded or one-hot encoded where appropriate.
- Sanitization: Infinite values coerced to NaN, columns entirely NaN dropped; remaining NaNs imputed (see modeling for pipeline).

## EDA: Key Insights and Visualizations
- Univariate:
  - Distributions of `purchase_value`, `age`, and time-of-day visualized in EDA notebooks.
- Bivariate:
  - Fraud rate by `source`, `browser`, `sex`, and age groups.
  - Time-based patterns around `hour_of_day` and `day_of_week`.
- Class distribution:
  - Severe imbalance in both datasets, visualized as bar charts.
- Country-level fraud patterns (Fraud_Data):
  - Top-risk countries identified post IP-to-country mapping.

Screenshots to include in submission (paste images into `reports/images/` and link here):
- Fraud rate by country — `fraud_by_country.png`
- Class distribution bars — `class_distribution_fraud.png`, `class_distribution_credit.png`
- Univariate histograms — `purchase_value_hist.png`, `age_hist.png`
- Bivariate plots — `fraud_by_source.png`, `fraud_by_browser.png`

## Feature Engineering Choices
- `time_since_signup` (hours): Critical signal; rapid signup→purchase often correlates with fraud. Implemented via `time_since_signup_hours()` in [src/feature_utils.py](src/feature_utils.py).
- IP→Country mapping: Merged `Fraud_Data.csv` with `IpAddress_to_Country.csv` using IP integer ranges to derive `country`. Challenges: ensuring correct range joins and performance; verified samples and edge-cases.
- Time-based features: `hour_of_day`, `day_of_week` derived from purchase timestamps for behavior profiling.
- Velocity features: Device/IP transaction velocity within rolling windows using `count_prev_within()` in [src/feature_utils.py](src/feature_utils.py).

## Class Imbalance: Analysis and Strategy
- Quantification:
  - Fraud_Data ≈ 9% fraud rate (binary target).
  - creditcard ≈ 0.17% fraud rate.
  - Confirmed via EDA and modeling outputs.
- Strategy:
  - Use stratified train/test splits to preserve target distribution.
  - Apply SMOTE oversampling only on the training set to avoid leakage.
  - Use model class weights (e.g., Logistic Regression/XGBoost `scale_pos_weight`) for extreme imbalance.
  - Primary metric: Precision-Recall AUC; secondary: F1, confusion matrix.
- Optional: Show before/after distributions post-SMOTE to illustrate balancing (plots can be added to EDA notebooks).

## Reproducibility
- Environment: see [requirements.txt](requirements.txt). Install with:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

- Data: processed files are in [data/processed/](data/processed/). Raw files in [data/raw/](data/raw/).
- Run notebooks in order:
  1. [notebooks/eda-fraud-data.ipynb](notebooks/eda-fraud-data.ipynb)
  2. [notebooks/eda-creditcard.ipynb](notebooks/eda-creditcard.ipynb)
  3. [notebooks/feature-engineering.ipynb](notebooks/feature-engineering.ipynb)
  4. [notebooks/modeling.ipynb](notebooks/modeling.ipynb)

## Submission Checklist
- Push latest changes to GitHub and share repo link.
- Ensure this report is present at [reports/interim1_report.md](reports/interim1_report.md).
- Include key plot screenshots in `reports/images/` and reference them in this report.
- Verify notebooks run end‑to‑end without errors.

## Notes & Next Steps
- If any country-level plot is missing, add a cell to aggregate fraud rate by `country` and visualize the top N.
- Proceed to Task 2 modeling with class-weighted estimators; evaluate using PR‑AUC and F1.
