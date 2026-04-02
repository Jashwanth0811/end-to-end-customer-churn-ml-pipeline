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
01 Overview Dashboard     07 Revenue Impact
02 Correlation Heatmap    08 Model Comparison
03 Demographics           09 ROC Curves (all models)
04 Payment & Billing      10 Best Model Deep-dive
05 Scatter Tenure/Charges 11 Threshold Tuning
06 Support Calls          12 Risk Dashboard
