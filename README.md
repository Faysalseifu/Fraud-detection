# Improved-detection-of-fraud-cases-for-e-commerce-and-bank-transactions
This project involves analyzing and preprocessing transaction data, engineering features that help identify fraud patterns, building and training models, evaluating them, and interpreting decisions.

## Current progress (Task 1)
- Data cleaning: duplicates removed; dtypes corrected; missing values inspected (no imputation applied yet).
- EDA (fraud & creditcard): univariate/bivariate plots, class imbalance quantified, key feature distributions explored.
- Geolocation: IP range merge adds country; fraud patterns by country plotted.
- Fraud_Data feature engineering: time-based features (hour_of_day, day_of_week, time_since_signup) and per-user velocity (tx_count_1h/24h); saved to data/processed/fraud_features.csv.
- Data transformation (fraud): stratified split; One-Hot Encode categoricals; scale numerics; SMOTE on train only with class balance logged pre/post.
- Data transformation (creditcard): stratified split; scale numerics; SMOTE on train only with class balance logged pre/post.

## Outstanding
- Decide and document missing-value handling (impute vs. drop) and apply if needed.
- Modeling notebook: train/evaluate models with the transformed datasets and report metrics.
- Explainability notebook: add SHAP/global + local explanations for the chosen model.
