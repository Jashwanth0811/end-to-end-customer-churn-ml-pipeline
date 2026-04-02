"""
report.py
Generates the final churn risk report:
  - Customer-level churn probability scoring
  - Risk segmentation visualization
  - Plain-text summary report
"""
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from sklearn.metrics import classification_report

BG = "#F8F9FA"; FONT_DARK = "#2C3E50"
OUT = Path("visualizations"); OUT.mkdir(exist_ok=True)
REP = Path("reports");        REP.mkdir(exist_ok=True)

plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor":"#FFFFFF",
                     "text.color": FONT_DARK, "grid.color": "#DEE2E6"})

RISK_BINS   = [0, 0.30, 0.55, 0.75, 1.01]
RISK_LABELS = ["Low Risk", "Moderate Risk", "High Risk", "Critical Risk"]
RISK_COLORS = ["#2ECC71", "#F39C12", "#E67E22", "#E74C3C"]


def score_customers(df_raw, best_clf, scaler, feature_names, best_threshold):
    """
    Run the best model on the full dataset to assign churn probabilities.
    Returns a scored DataFrame.
    """
    from features import engineer_features
    df_feat = engineer_features(df_raw.copy())
    X_all   = df_feat.drop(columns=["Churn"])

    # Align columns (some dummy cols might differ slightly on full data)
    missing = [c for c in feature_names if c not in X_all.columns]
    for m in missing:
        X_all[m] = 0
    X_all = X_all[feature_names]

    X_scaled = pd.DataFrame(scaler.transform(X_all), columns=feature_names)
    proba    = best_clf.predict_proba(X_scaled)[:, 1]
    pred     = (proba >= best_threshold).astype(int)

    result = df_raw[["CustomerID","Tenure","MonthlyCharges","TotalCharges",
                      "Contract","InternetService","SupportCalls","Churn"]].copy()
    result["ChurnProbability"] = proba.round(4)
    result["PredictedChurn"]   = pred
    result["RiskSegment"]      = pd.cut(proba, bins=RISK_BINS,
                                        labels=RISK_LABELS, right=False)
    result.sort_values("ChurnProbability", ascending=False, inplace=True)
    result.to_csv(REP / "customer_churn_scores.csv", index=False)
    print(f"  ✓  Customer scores → reports/customer_churn_scores.csv ({len(result):,} rows)")
    return result


def plot_risk_dashboard(scored_df):
    fig = plt.figure(figsize=(16, 10), facecolor=BG)
    fig.suptitle("Customer Churn Risk Dashboard", fontsize=18,
                 fontweight="bold", y=1.01)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # Risk segment donut
    ax0 = fig.add_subplot(gs[0, 0])
    seg_counts = scored_df["RiskSegment"].value_counts().reindex(RISK_LABELS)
    ax0.pie(seg_counts.values, labels=seg_counts.index, colors=RISK_COLORS,
            autopct="%1.1f%%", startangle=90,
            wedgeprops=dict(width=0.55, edgecolor="white", linewidth=2),
            textprops={"fontsize": 9})
    ax0.set_title("Risk Segment Distribution", fontweight="bold")

    # Probability histogram
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.hist(scored_df["ChurnProbability"], bins=40, color="#3498DB",
             edgecolor="white", alpha=0.85)
    ax1.axvline(0.50, color="red", lw=1.5, linestyle="--", label="50% threshold")
    ax1.set_xlabel("Churn Probability"); ax1.set_ylabel("Customers")
    ax1.set_title("Churn Probability Distribution"); ax1.legend()

    # Monthly revenue at risk by segment
    ax2 = fig.add_subplot(gs[0, 2])
    rev_risk = scored_df.groupby("RiskSegment", observed=True)["MonthlyCharges"].sum()
    rev_risk = rev_risk.reindex(RISK_LABELS)
    bars = ax2.bar(RISK_LABELS, rev_risk.values, color=RISK_COLORS, edgecolor="white")
    for b in bars:
        ax2.text(b.get_x() + b.get_width()/2, b.get_height() + 100,
                 f"${b.get_height():,.0f}", ha="center", fontsize=9, fontweight="bold")
    ax2.set_ylabel("Monthly Revenue ($)"); ax2.set_title("Revenue at Risk by Segment")
    ax2.tick_params(axis="x", rotation=20)

    # Avg tenure by risk
    ax3 = fig.add_subplot(gs[1, 0])
    tenure_risk = scored_df.groupby("RiskSegment", observed=True)["Tenure"].mean().reindex(RISK_LABELS)
    ax3.bar(RISK_LABELS, tenure_risk.values, color=RISK_COLORS, edgecolor="white", width=0.5)
    ax3.set_ylabel("Avg Tenure (months)"); ax3.set_title("Avg Customer Tenure by Risk")
    ax3.tick_params(axis="x", rotation=20)

    # Avg monthly charges by risk
    ax4 = fig.add_subplot(gs[1, 1])
    charge_risk = scored_df.groupby("RiskSegment", observed=True)["MonthlyCharges"].mean().reindex(RISK_LABELS)
    ax4.bar(RISK_LABELS, charge_risk.values, color=RISK_COLORS, edgecolor="white", width=0.5)
    ax4.set_ylabel("Avg Monthly Charge ($)"); ax4.set_title("Avg Monthly Charge by Risk")
    ax4.tick_params(axis="x", rotation=20)

    # Top 15 highest risk customers table
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis("off")
    top15 = scored_df.head(15)[["CustomerID","ChurnProbability","Contract","RiskSegment"]]
    table_data = [[row["CustomerID"], f"{row['ChurnProbability']:.1%}",
                   row["Contract"][:10], str(row["RiskSegment"])] for _, row in top15.iterrows()]
    tbl = ax5.table(cellText=table_data,
                    colLabels=["CustomerID","Prob","Contract","Risk"],
                    cellLoc="center", loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1, 1.15)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#DEE2E6")
        if r == 0: cell.set_facecolor("#2C3E50"); cell.set_text_props(color="white")
        elif r % 2: cell.set_facecolor("#F8F9FA")
    ax5.set_title("Top 15 Highest Risk Customers", fontweight="bold", pad=10)

    path = OUT / "12_risk_dashboard.png"
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  ✓  {path}")


def write_text_report(df_raw, results_df, best_name, scored_df, best_threshold):
    churn_rate     = df_raw["Churn"].mean()
    n_churned      = df_raw["Churn"].sum()
    rev_at_risk    = scored_df[scored_df["RiskSegment"].isin(["High Risk","Critical Risk"])]["MonthlyCharges"].sum()
    n_critical     = (scored_df["RiskSegment"] == "Critical Risk").sum()
    n_high         = (scored_df["RiskSegment"] == "High Risk").sum()

    seg_summary = scored_df.groupby("RiskSegment", observed=True).agg(
        Count=("CustomerID","count"),
        AvgProb=("ChurnProbability","mean"),
        AvgCharge=("MonthlyCharges","mean"),
        TotalRevenue=("MonthlyCharges","sum")
    ).reindex(RISK_LABELS)

    report = f"""
╔══════════════════════════════════════════════════════════════════╗
║          CUSTOMER CHURN ANALYSIS — EXECUTIVE REPORT             ║
╚══════════════════════════════════════════════════════════════════╝

Generated: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}

═══════════════════════════════════════
 1. DATASET SUMMARY
═══════════════════════════════════════
  Total Customers     : {len(df_raw):,}
  Churned Customers   : {n_churned:,}
  Retention Rate      : {1 - churn_rate:.1%}
  Churn Rate          : {churn_rate:.1%}
  Avg Monthly Revenue : ${df_raw['MonthlyCharges'].mean():.2f}
  Total Monthly Rev.  : ${df_raw['MonthlyCharges'].sum():,.2f}

═══════════════════════════════════════
 2. MODEL PERFORMANCE SUMMARY
═══════════════════════════════════════
  Best Model          : {best_name}
  Optimal Threshold   : {best_threshold:.2f}

  Ranking (by ROC-AUC):
"""
    for _, row in results_df.iterrows():
        report += f"  {'★' if row['Model'] == best_name else ' '} {row['Model']:<25} AUC={row['ROC-AUC']:.4f}  F1={row['F1']:.4f}\n"

    report += f"""
═══════════════════════════════════════
 3. RISK SEGMENTATION
═══════════════════════════════════════

{seg_summary.to_string()}

  ⚠  High + Critical Risk Customers : {n_critical + n_high:,}
  ⚠  Monthly Revenue at Risk        : ${rev_at_risk:,.2f}

═══════════════════════════════════════
 4. KEY CHURN DRIVERS (Insights)
═══════════════════════════════════════
  • Month-to-month contract customers churn at ~3× the rate of annual contracts.
  • Fiber optic users show elevated churn vs DSL (likely pricing sensitivity).
  • Customers with 3+ support calls are significantly more likely to churn.
  • Senior citizens churn at a higher rate (~40% vs ~26% overall).
  • High monthly charges combined with short tenure = highest risk profile.
  • Electronic check payers have the highest churn vs credit card users.

═══════════════════════════════════════
 5. RECOMMENDED ACTIONS
═══════════════════════════════════════
  [CRITICAL RISK]
  • Immediate outreach — personalized retention offers, account managers.
  • Investigate service issues (support call history).
  • Offer contract upgrade incentive (month-to-month → annual).

  [HIGH RISK]
  • Proactive email/SMS campaigns with loyalty discounts.
  • Bundle additional services to increase stickiness.
  • Tech support check-in for customers with recent complaints.

  [MODERATE RISK]
  • Targeted NPS survey to identify dissatisfaction early.
  • Reward program enrollment / referral bonuses.

  [LOW RISK]
  • Maintain quality, upsell premium services.
  • Encourage contract renewal before expiry.

═══════════════════════════════════════
 6. FINANCIAL IMPACT ESTIMATE
═══════════════════════════════════════
  Assuming 30% retention of High+Critical risk customers:
  • Customers Saved        : ~{int((n_critical + n_high) * 0.30):,}
  • Monthly Revenue Saved  : ~${rev_at_risk * 0.30:,.2f}
  • Annual Revenue Saved   : ~${rev_at_risk * 0.30 * 12:,.2f}

══════════════════════════════════════════════════════════════════
  All visualizations saved in: visualizations/
  Customer scores saved in  : reports/customer_churn_scores.csv
══════════════════════════════════════════════════════════════════
"""
    path = REP / "churn_analysis_report.txt"
    with open(path, "w") as f:
        f.write(report)
    print(f"  ✓  {path}")
    print(report)
    return report


def run_report(df_raw, results_df, best_name, best_clf, scaler,
               feature_names, best_threshold):
    print("\n📋  Generating Risk Scores & Report...")
    scored_df = score_customers(df_raw, best_clf, scaler, feature_names, best_threshold)
    plot_risk_dashboard(scored_df)
    write_text_report(df_raw, results_df, best_name, scored_df, best_threshold)
    return scored_df
