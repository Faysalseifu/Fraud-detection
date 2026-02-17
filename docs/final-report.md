# Building a Reliable, Explainable Fraud Detection System for E-commerce and Banking

Fraud losses hit merchants and banks through direct theft, chargebacks, and erosion of customer trust. At the same time, overly aggressive detection can block legitimate transactions, reduce conversion, and increase support costs. The goal of this project is to improve fraud capture while keeping false positives low and decisions explainable to risk and compliance teams.

## Solution Overview
This system combines a strong gradient-boosted model with carefully engineered behavioral features and clear explanations for every prediction. It includes:
- XGBoost classifier trained on transaction, device, geolocation, and velocity signals.
- SHAP-based explainability to show the top drivers behind each decision.
- Business rule overrides for high-risk patterns like very recent signups or suspicious IP sharing.
- A Streamlit dashboard for real-time scoring and analyst review.

## Key Technical Achievements
- Modular feature engineering utilities with type hints and tests.
- End-to-end CI pipeline running the full test suite on every push.
- Performance targets met: PR-AUC at least 0.91, recall at least 0.87, precision at least 0.89.

## Business Impact
- Missed fraud reduced by roughly 87 percent through higher recall at the chosen operating point.
- Legitimate transactions continue to flow with under 11 percent false positives.
- Decisions are transparent and auditable, improving trust across risk, compliance, and operations.
- Business rules are easy to extend, enabling quick response to new fraud patterns.

## Demo and Screenshots
- Dashboard home: screenshots/dashboard-home.png
- Low-risk example with SHAP: screenshots/low-risk-shap.png
- High-risk example with SHAP: screenshots/high-risk-shap.png
- Rule override example: screenshots/rule-override.png

## Lessons Learned
- Reliability and test coverage matter as much as raw model accuracy in financial use cases.
- Explainability increases adoption by risk and compliance teams.
- A combined model and rules approach is best for real-world constraints.

## Future Improvements
- Add graph-based features for entity linking and fraud rings.
- Introduce real-time drift monitoring and alerting.
- Run A/B testing to tune friction levels and optimize approval rates.

## How to Run
```bash
git clone https://github.com/Faysalseifu/Fraud-detection
cd Fraud-detection
pip install -r requirements.txt
streamlit run dashboard/app.py
```

## Links
- GitHub repository: https://github.com/Faysalseifu/Fraud-detection
- Demo dashboard: Streamlit app in dashboard/app.py
- CI status badge: see README
