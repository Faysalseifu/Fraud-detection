# Fraud Risk Detector - AI-Powered Fraud Prevention Capstone

[![CI](https://github.com/Faysalseifu/Fraud-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/Faysalseifu/Fraud-detection/actions)

Interactive real-time fraud scoring dashboard for e-commerce and credit card transactions.

## Business Problem
Fraud causes massive financial losses and chargebacks for merchants and banks while false positives frustrate legitimate customers and increase support costs.

## Solution
- XGBoost model trained on behavioral, geolocation, and velocity features
- SHAP explainability for every prediction
- Business rule overrides (time since signup, device/IP sharing, country risk)
- Catches ~87% of fraud cases with precision >=89% (PR-AUC >=0.91)

## Quick Start
```bash
git clone https://github.com/Faysalseifu/Fraud-detection
cd Fraud-detection
pip install -r requirements.txt
streamlit run dashboard/app.py
```

## Demo
![Dashboard home](screenshots/dashboard-home.png)
![Low-risk prediction with SHAP](screenshots/low-risk-shap.png)
![High-risk prediction with SHAP](screenshots/high-risk-shap.png)

## Final Report
Read the write-up in [docs/final-report.md](docs/final-report.md).

## Key Results
- Fraud recall: >=87%
- Precision at operating point: >=89%
- Very low false positive rate -> minimal customer friction
- Transparent decisions via SHAP -> trusted by risk teams

## Project Structure
```
├── dashboard/              # Streamlit app
├── data/                   # Raw and processed datasets
├── models/                 # Trained model artifacts
├── notebooks/              # EDA, feature engineering, modeling
├── reports/                # Interim reports
├── src/                    # Feature utilities
├── tests/                  # pytest suite
├── .github/workflows/      # CI pipeline
├── screenshots/            # Demo images
└── requirements.txt
```

## Features
- Real-time fraud probability and risk level
- SHAP waterfall and force plot explanations
- Interactive business rules (override model with domain logic)
- Test coverage and automated CI

## Author
Faysal
LinkedIn: https://www.linkedin.com/in/your-link
Contact: your.email@example.com

## Built for
Finance and risk teams who need reliable, explainable fraud detection with minimal customer impact.

