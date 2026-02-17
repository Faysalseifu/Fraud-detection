# Building a Reliable & Explainable Fraud Detection System for Merchants and Banks

## 1) The Business Problem
Fraud prevention in digital payments is a risk-management balancing act. If institutions miss fraud, losses appear immediately through chargebacks, reimbursement, operational rework, and downstream trust damage. If they overreact, they block good customers, create checkout friction, and reduce lifetime value. In both e-commerce and banking environments, this becomes a measurable tradeoff between security and growth.

The practical challenge is that fraud behavior changes fast: new accounts are spun up quickly, device and IP patterns evolve, and attacks adapt to static rules. A model that only optimizes for overall accuracy can look good on paper while still underperforming in production where class imbalance is severe and false negatives are expensive. At the same time, finance teams cannot rely on opaque black-box decisions. They need clear, auditable reasons for each flag so compliance, operations, and customer support can act with confidence.

This capstone targets that real-world gap by building a system that is both high-performing and operationally trustworthy: strong fraud detection power, low unnecessary customer friction, and transparent explanations for every scored transaction.

## 2) Approach & Solution
The project combines machine learning, engineered risk features, and explainability into a practical scoring workflow. Two datasets were used to capture complementary fraud patterns: e-commerce behavior and credit-card transaction dynamics. For the e-commerce stream, geolocation enrichment was included through IP-to-country mapping, which improves context around regional activity and suspicious origin patterns.

Feature engineering focused on signals that risk teams can interpret and operationalize. Examples include:
- **Signup velocity** (`time_since_signup_hours`) to identify very new accounts making high-value attempts.
- **Shared infrastructure** (`device_count`, `ip_count`) to detect account clustering and potentially coordinated activity.
- **Temporal behavior** (`hour_of_day`, `day_of_week`) to capture unusual transaction timing patterns.
- **Country signal** using known risk differentiation and fallback handling for unseen categories.

For modeling, the core classifier is **XGBoost**, selected for robust non-linear learning, strong tabular performance, and stable behavior under imbalanced conditions. The pipeline uses preprocessing plus model artifacts saved for reproducible inference. Production-minded quality controls were added through unit/integration tests and CI execution.

To address trust and governance, the system integrates **SHAP** explanations for each prediction. Instead of outputting only a risk probability, the dashboard surfaces feature contributions so analysts can answer “why” a transaction was flagged. This converts model output into decision support.

Finally, domain rules are layered on top of model probabilities as optional overrides (for example, very fast signup or unusually shared devices/IPs). This hybrid architecture mirrors how finance risk teams operate in practice: ML for pattern detection, rules for policy control and rapid adaptation.

## 3) Key Results
The final system met the project’s target performance profile for high-stakes fraud screening:
- **Recall ≥ 87%** to capture the large majority of fraudulent activity.
- **Precision ≥ 89%** to minimize false alarms and reduce unnecessary investigation cost.
- **PR-AUC ≥ 0.91** for strong ranking quality under class imbalance.

Beyond model metrics, engineering reliability was validated through automated testing and CI. End-to-end behavior checks cover feature utilities, model save/load consistency, SHAP output shape compatibility, and resilience to unusual inputs. This ensures the solution is not only accurate in notebooks, but dependable when packaged for real usage.

## 4) Demo Highlights
The Streamlit dashboard demonstrates an analyst-friendly risk workflow:
- **Home screen**: business summary, model context, and input methods.
- **Low-risk example**: normal transaction receives low score with SHAP factors supporting approval.
- **High-risk example**: suspicious profile gets elevated score with clear feature-level explanation.
- **Rule-toggle scenario**: policy switches can force or boost risk when business constraints require conservative handling.

Suggested screenshots for submission package:
- `README` with CI badge visible.
- Dashboard landing page.
- Low-risk prediction with SHAP panel.
- High-risk prediction with SHAP panel.
- Rule-override demonstration.
- `pytest` terminal showing all tests passing.
- GitHub Actions run showing green CI.

## 5) Business Value
The most important outcome of this capstone is business practicality. In finance, model quality matters only if it translates into fewer losses and smoother customer operations. This solution does both.

First, stronger fraud capture directly lowers avoidable loss exposure. Catching more fraudulent attempts earlier reduces chargeback pressure, manual rework, and potential reputational risk. Second, high precision helps preserve legitimate customer flow. That means fewer unnecessary declines, less customer frustration, and better conversion retention for merchants.

Third, explainability creates organizational trust. SHAP-based reasoning allows risk, compliance, and audit functions to understand the basis of each decision, which is critical for regulated or policy-sensitive environments. Support teams also benefit because they can communicate clear rationale when customer cases are reviewed.

Fourth, the rule + ML design improves governance. Risk owners can apply business controls immediately without retraining the entire model, while still benefiting from learned fraud patterns. This shortens response time when threat behavior changes.

Overall, the system supports a core finance objective: reduce fraud cost while maintaining customer experience and decision transparency.

## 6) Lessons Learned
Three lessons stood out during implementation. The first is that **reliability is as important as accuracy**. In financial systems, repeatability, tests, and CI checks are part of model performance because unstable pipelines create operational risk. The second is that **explainability accelerates adoption**. Teams are far more comfortable acting on model output when they can inspect feature-level drivers. The third is that **hybrid decisioning wins in practice**. Pure ML can miss policy nuance, while pure rules struggle with evolving fraud tactics; combining both provides stronger control and flexibility.

## 7) Next Steps
- Add graph-based entity features to detect fraud rings and shared identities.
- Introduce real-time data/score drift monitoring with alert thresholds.
- Run A/B tests on review friction to optimize approval rate vs. fraud loss.

## Reproducibility
```bash
git clone https://github.com/Faysalseifu/Fraud-detection
cd Fraud-detection
pip install -r requirements.txt
pytest -v
streamlit run dashboard/app.py
```
