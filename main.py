"""
main.py
═══════════════════════════════════════════════════════════════════
Customer Churn Analysis & Prediction Pipeline
═══════════════════════════════════════════════════════════════════
Steps:
  1. Generate / load dataset
  2. Exploratory Data Analysis + Visualizations
  3. Feature Engineering & Preprocessing
  4. Train & Compare 8 ML Models
  5. Deep-dive on Best Model
  6. Customer Risk Scoring + Executive Report

Usage:
    python main.py                  # full pipeline (generates data)
    python main.py --no-eda         # skip EDA plots (faster)
    python main.py --data path.csv  # use your own CSV
"""

import argparse, sys, time
from pathlib import Path

import pandas as pd

# ── Banner ─────────────────────────────────────────────────────────────────────
BANNER = """
╔══════════════════════════════════════════════════════════════╗
║   🔮  Customer Churn Analysis & Prediction System  🔮       ║
║        Powered by Scikit-learn  |  Python 3                 ║
╚══════════════════════════════════════════════════════════════╝
"""

def main():
    print(BANNER)
    t_start = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("--data",   default=None, help="Path to customer CSV")
    parser.add_argument("--no-eda", action="store_true", help="Skip EDA plots")
    args = parser.parse_args()

    # ── Step 1: Data ──────────────────────────────────────────────────────────
    print("=" * 62)
    print("  STEP 1: Data Loading")
    print("=" * 62)
    if args.data and Path(args.data).exists():
        df = pd.read_csv(args.data)
        print(f"  Loaded: {args.data}  ({df.shape[0]:,} rows)")
    else:
        print("  Generating synthetic telecom churn dataset...")
        from data.generate_data import generate_churn_data
        df = generate_churn_data(n=5000)
        df.to_csv("data/telecom_churn.csv", index=False)
        print(f"  Generated: 5,000 customers, churn rate={df.Churn.mean():.1%}")

    print(f"\n  Shape : {df.shape}")
    print(f"  Cols  : {list(df.columns)}")
    print(f"  Churn : {df.Churn.sum():,} ({df.Churn.mean():.1%})\n")

    # ── Step 2: EDA ───────────────────────────────────────────────────────────
    print("=" * 62)
    print("  STEP 2: Exploratory Data Analysis")
    print("=" * 62)
    if not args.no_eda:
        from eda import run_eda
        run_eda(df)
    else:
        print("  (skipped via --no-eda flag)\n")

    # ── Step 3: Feature Engineering ───────────────────────────────────────────
    print("=" * 62)
    print("  STEP 3: Feature Engineering & Preprocessing")
    print("=" * 62)
    from features import get_feature_matrix
    X_train, X_test, y_train, y_test, feature_names, scaler = get_feature_matrix(df)
    print(f"  Train  : {X_train.shape[0]:,} samples, {X_train.shape[1]} features")
    print(f"  Test   : {X_test.shape[0]:,}  samples")
    print(f"  Target : {y_train.sum()} churned in train | {y_test.sum()} in test\n")

    # ── Step 4 & 5: Modeling ──────────────────────────────────────────────────
    print("=" * 62)
    print("  STEP 4 & 5: Model Training, Evaluation & Visualization")
    print("=" * 62)
    from models_train import run_models
    trained, results_df, best_name, best_clf, best_threshold = \
        run_models(X_train, X_test, y_train, y_test, feature_names)

    # ── Step 6: Report ────────────────────────────────────────────────────────
    print("=" * 62)
    print("  STEP 6: Risk Scoring & Executive Report")
    print("=" * 62)
    from report import run_report
    scored_df = run_report(df, results_df, best_name, best_clf,
                           scaler, feature_names, best_threshold)

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    print("\n" + "=" * 62)
    print(f"  ✅  Pipeline complete in {elapsed:.1f}s")
    print("=" * 62)
    print("\n  📁  Output files:")
    for p in sorted(Path("visualizations").glob("*.png")):
        print(f"      {p}")
    for p in sorted(Path("reports").glob("*")):
        print(f"      {p}")
    for p in sorted(Path("models").glob("*")):
        print(f"      {p}")
    print()

if __name__ == "__main__":
    main()
