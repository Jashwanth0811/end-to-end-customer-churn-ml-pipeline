# 🔮 Customer Churn Analysis & Prediction

A full-stack Python ML project for telecom customer churn analysis.

## Project Structure
```
churn_project/
├── main.py            ← Orchestration (run this!)
├── eda.py             ← Exploratory Data Analysis + 7 visualizations
├── features.py        ← Feature engineering & preprocessing
├── models_train.py    ← Train & compare 8 ML models
├── report.py          ← Risk scoring & executive report
├── data/
│   └── generate_data.py  ← Synthetic dataset generator
├── visualizations/    ← All 12 output charts (auto-generated)
├── reports/           ← Scored customers CSV + text report
├── models/            ← Saved best model + comparison CSV
└── requirements.txt
```

## Quickstart
```bash
pip install -r requirements.txt
python main.py
```

## Models Compared
Logistic Regression, Decision Tree, Random Forest,
Gradient Boosting, AdaBoost, KNN, Naive Bayes, SVM

## Visualizations Generated (12 total)
```
01 Overview Dashboard     07 Revenue Impact
02 Correlation Heatmap    08 Model Comparison
03 Demographics           09 ROC Curves (all models)
04 Payment & Billing      10 Best Model Deep-dive
05 Scatter Tenure/Charges 11 Threshold Tuning
06 Support Calls          12 Risk Dashboard
```
<img width="2114" height="1580" alt="01_overview_dashboard" src="https://github.com/user-attachments/assets/695f571c-f906-4398-b92d-10957d6d6839" />
<img width="1360" height="1225" alt="02_correlation_heatmap" src="https://github.com/user-attachments/assets/45fcbcdf-a9d6-4df2-9ca0-06e878c0f5fd" />
<img width="2069" height="731" alt="06_support_calls" src="https://github.com/user-attachments/assets/ac7d8252-7782-48a1-93fc-3dab9a3e8b68" />
<img width="1920" height="732" alt="07_revenue_impact" src="https://github.com/user-attachments/assets/bd0b9051-0fe2-445c-b573-bde87b586ab5" />
<img width="2070" height="1019" alt="08_model_comparison" src="https://github.com/user-attachments/assets/a7e1fe3f-d799-4ef5-b1e8-40fc1b583408" />
<img width="1286" height="1066" alt="09_roc_curves" src="https://github.com/user-attachments/assets/b79f8a16-e718-4ff6-82db-a81db6b271fd" />
<img width="1976" height="1728" alt="10_best_model_detail" src="https://github.com/user-attachments/assets/0f7beef3-ebab-442c-b0fa-3d4695cd854b" />
<img width="1620" height="866" alt="11_threshold_tuning" src="https://github.com/user-attachments/assets/c890c5fb-db73-4e81-9717-9b88fb8b1525" />
<img width="1979" height="1483" alt="12_risk_dashboard" src="https://github.com/user-attachments/assets/eb9ac6c7-74fe-4740-a687-1991cedce894" />
