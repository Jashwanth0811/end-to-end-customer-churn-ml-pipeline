"""
generate_data.py
Generates a realistic synthetic telecom customer churn dataset.
"""
import numpy as np
import pandas as pd

np.random.seed(42)

def generate_churn_data(n=5000):
    tenure         = np.random.exponential(scale=24, size=n).clip(1, 72).astype(int)
    monthly_charge = np.random.normal(65, 20, n).clip(20, 120).round(2)
    num_products   = np.random.choice([1, 2, 3, 4], n, p=[0.3, 0.4, 0.2, 0.1])
    support_calls  = np.random.poisson(1.5, n).clip(0, 10)
    contract       = np.random.choice(["Month-to-month", "One year", "Two year"],
                                      n, p=[0.55, 0.25, 0.20])
    payment        = np.random.choice(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
        n, p=[0.35, 0.22, 0.22, 0.21])
    internet       = np.random.choice(["DSL", "Fiber optic", "No"], n, p=[0.35, 0.45, 0.20])
    senior         = np.random.choice([0, 1], n, p=[0.84, 0.16])
    partner        = np.random.choice([0, 1], n, p=[0.52, 0.48])
    dependents     = np.random.choice([0, 1], n, p=[0.70, 0.30])
    paperless      = np.random.choice([0, 1], n, p=[0.41, 0.59])
    tech_support   = np.random.choice([0, 1], n, p=[0.50, 0.50])
    online_backup  = np.random.choice([0, 1], n, p=[0.47, 0.53])
    streaming_tv   = np.random.choice([0, 1], n, p=[0.45, 0.55])
    gender         = np.random.choice(["Male", "Female"], n)

    total_charges  = (monthly_charge * tenure + np.random.normal(0, 50, n)).clip(0).round(2)

    # Churn probability model (realistic correlations)
    logit = (
        -2.5
        - 0.04 * tenure
        + 0.03 * monthly_charge
        + 0.20 * support_calls
        + 0.40 * senior
        - 0.30 * partner
        - 0.20 * num_products
        + 1.20 * (contract == "Month-to-month").astype(int)
        - 0.60 * (contract == "Two year").astype(int)
        + 0.50 * (internet == "Fiber optic").astype(int)
        + 0.30 * (payment == "Electronic check").astype(int)
        - 0.20 * tech_support
        - 0.15 * online_backup
        + 0.10 * paperless
        + np.random.normal(0, 0.5, n)
    )
    prob  = 1 / (1 + np.exp(-logit))
    churn = (np.random.rand(n) < prob).astype(int)

    df = pd.DataFrame({
        "CustomerID"    : [f"CUST-{i:05d}" for i in range(1, n + 1)],
        "Gender"        : gender,
        "SeniorCitizen" : senior,
        "Partner"       : partner,
        "Dependents"    : dependents,
        "Tenure"        : tenure,
        "PhoneService"  : np.random.choice([0, 1], n, p=[0.10, 0.90]),
        "MultipleLines" : np.random.choice([0, 1], n, p=[0.53, 0.47]),
        "InternetService": internet,
        "OnlineSecurity": np.random.choice([0, 1], n, p=[0.52, 0.48]),
        "OnlineBackup"  : online_backup,
        "DeviceProtection": np.random.choice([0, 1], n, p=[0.51, 0.49]),
        "TechSupport"   : tech_support,
        "StreamingTV"   : streaming_tv,
        "StreamingMovies": np.random.choice([0, 1], n, p=[0.50, 0.50]),
        "Contract"      : contract,
        "PaperlessBilling": paperless,
        "PaymentMethod" : payment,
        "NumProducts"   : num_products,
        "SupportCalls"  : support_calls,
        "MonthlyCharges": monthly_charge,
        "TotalCharges"  : total_charges,
        "Churn"         : churn,
    })
    return df

if __name__ == "__main__":
    df = generate_churn_data()
    df.to_csv("data/telecom_churn.csv", index=False)
    print(f"Dataset saved: {df.shape[0]} rows, churn rate={df.Churn.mean():.1%}")
