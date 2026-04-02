"""
features.py
Feature engineering and preprocessing pipeline for churn prediction.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


BINARY_COLS = ["SeniorCitizen", "Partner", "Dependents", "PhoneService",
               "MultipleLines", "OnlineSecurity", "OnlineBackup",
               "DeviceProtection", "TechSupport", "StreamingTV",
               "StreamingMovies", "PaperlessBilling"]

CAT_COLS    = ["Gender", "InternetService", "Contract", "PaymentMethod"]

NUM_COLS    = ["Tenure", "MonthlyCharges", "TotalCharges",
               "SupportCalls", "NumProducts"]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Derived numeric features
    df["ChargesPerMonth"]      = (df["TotalCharges"] / df["Tenure"]).replace([np.inf], 0)
    df["HighValueCustomer"]    = (df["MonthlyCharges"] > df["MonthlyCharges"].quantile(0.75)).astype(int)
    df["LongTenure"]           = (df["Tenure"] > 24).astype(int)
    df["FrequentCaller"]       = (df["SupportCalls"] >= 3).astype(int)
    df["TenureChargeInteract"] = df["Tenure"] * df["MonthlyCharges"]
    df["ServicesCount"]        = (df[["OnlineSecurity","OnlineBackup","DeviceProtection",
                                      "TechSupport","StreamingTV","StreamingMovies"]].sum(axis=1))

    # Encode categoricals
    for col in CAT_COLS:
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=False)
        dummies.columns = [c.replace(" ", "_").replace("-", "_") for c in dummies.columns]
        df = pd.concat([df, dummies], axis=1)

    df.drop(columns=["CustomerID"] + CAT_COLS, inplace=True)
    return df


def get_feature_matrix(df: pd.DataFrame):
    """Returns (X_train, X_test, y_train, y_test, feature_names, scaler)."""
    df_feat = engineer_features(df)
    X = df_feat.drop(columns=["Churn"])
    y = df_feat["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y)

    scaler  = StandardScaler()
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    X_test_s  = pd.DataFrame(scaler.transform(X_test),      columns=X.columns)

    return X_train_s, X_test_s, y_train, y_test, list(X.columns), scaler


if __name__ == "__main__":
    df = pd.read_csv("data/telecom_churn.csv")
    X_tr, X_te, y_tr, y_te, feats, _ = get_feature_matrix(df)
    print(f"Train: {X_tr.shape}, Test: {X_te.shape}")
    print(f"Features ({len(feats)}): {feats[:8]} ...")
