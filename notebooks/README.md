# Notebooks Directory

This directory contains all Jupyter notebooks for the fraud detection project, organized by task.

## Notebook Descriptions

- **eda-fraud-data.ipynb**  
	Exploratory Data Analysis for the e-commerce dataset (`Fraud_Data.csv` + IP-to-country mapping).  
	Includes: data loading, cleaning, class imbalance visualization, geolocation merge, and initial insights.

- **eda-creditcard.ipynb**  
	Exploratory Data Analysis for the credit card transaction dataset (`creditcard.csv`).  
	Focuses on distributions of Amount, Time, PCA features (V1-V28), and extreme class imbalance.

- **feature-engineering.ipynb**  
	Feature creation and preprocessing for both datasets.  
	Includes:  
	- Time-based features (hour_of_day, day_of_week, time_since_signup)  
	- Device/IP velocity features  
	- Country encoding  
	- Scaling and encoding pipelines  
	- Saving processed datasets

- **modeling.ipynb**  
	Model building, training, and evaluation.  
	Includes:  
	- Train-test split (stratified)  
	- Baseline Logistic Regression  
	- Ensemble model (XGBoost/LightGBM/RandomForest)  
	- Cross-validation  
	- Comparison using PR-AUC, F1, confusion matrix  
	- Final model selection

- **shap-explainability.ipynb**  
	Model interpretability using SHAP.  
	Includes:  
	- Global feature importance (summary plot)  
	- Built-in importance comparison  
	- Force plots for TP, FP, FN examples  
	- Business recommendations derived from insights

## Usage Order
Run notebooks in this order:
1. eda-fraud-data.ipynb
2. eda-creditcard.ipynb
3. feature-engineering.ipynb
4. modeling.ipynb
5. shap-explainability.ipynb
# Notebooks

Notebooks for exploratory data analysis, feature engineering, modeling, and explainability. Each notebook is kept lightweight and can be run independently.
