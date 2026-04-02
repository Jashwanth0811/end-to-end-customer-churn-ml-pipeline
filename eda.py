"""
eda.py
Exploratory Data Analysis and rich visualizations for churn analysis.
Saves all plots to visualizations/ directory.
"""
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path

# ── Style ──────────────────────────────────────────────────────────────────────
PALETTE   = {"No Churn": "#2ECC71", "Churn": "#E74C3C"}
COLORS    = ["#2ECC71", "#E74C3C"]
BG        = "#F8F9FA"
FONT_DARK = "#2C3E50"
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": "#FFFFFF",
                     "axes.edgecolor": "#DEE2E6", "grid.color": "#DEE2E6",
                     "text.color": FONT_DARK})

OUT = Path("visualizations")
OUT.mkdir(exist_ok=True)


def save(fig, name):
    path = OUT / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {path}")


# ── 1. Overview Dashboard ──────────────────────────────────────────────────────
def plot_overview(df):
    churn_rate = df.Churn.mean()
    counts     = df.Churn.value_counts()

    fig = plt.figure(figsize=(16, 10), facecolor=BG)
    fig.suptitle("Customer Churn — Overview Dashboard", fontsize=18,
                 fontweight="bold", color=FONT_DARK, y=1.01)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # Donut
    ax0 = fig.add_subplot(gs[0, 0])
    wedges, texts, autotexts = ax0.pie(
        counts, labels=["No Churn", "Churn"], colors=COLORS,
        autopct="%1.1f%%", startangle=90,
        wedgeprops=dict(width=0.55, edgecolor="white", linewidth=2),
        textprops={"fontsize": 11})
    for at in autotexts: at.set_fontweight("bold")
    ax0.set_title("Churn Distribution", fontweight="bold")

    # Tenure histogram
    ax1 = fig.add_subplot(gs[0, 1])
    for val, label, col in zip([0, 1], ["No Churn", "Churn"], COLORS):
        ax1.hist(df[df.Churn == val]["Tenure"], bins=24, alpha=0.75,
                 color=col, label=label, edgecolor="white")
    ax1.set_xlabel("Tenure (months)"); ax1.set_ylabel("Count")
    ax1.set_title("Tenure Distribution by Churn"); ax1.legend()

    # Monthly charges KDE
    ax2 = fig.add_subplot(gs[0, 2])
    for val, label, col in zip([0, 1], ["No Churn", "Churn"], COLORS):
        subset = df[df.Churn == val]["MonthlyCharges"]
        ax2.hist(subset, bins=24, alpha=0.65, color=col,
                 label=label, edgecolor="white", density=True)
        subset.plot.kde(ax=ax2, color=col, linewidth=2)
    ax2.set_xlabel("Monthly Charges ($)"); ax2.set_ylabel("Density")
    ax2.set_title("Monthly Charges Distribution"); ax2.legend()

    # Churn by contract
    ax3 = fig.add_subplot(gs[1, 0])
    ct = df.groupby("Contract")["Churn"].mean().sort_values(ascending=False)
    bars = ax3.barh(ct.index, ct.values * 100, color=["#E74C3C","#F39C12","#2ECC71"],
                    edgecolor="white", height=0.5)
    for b in bars:
        ax3.text(b.get_width() + 0.5, b.get_y() + b.get_height()/2,
                 f"{b.get_width():.1f}%", va="center", fontsize=10)
    ax3.set_xlabel("Churn Rate (%)"); ax3.set_title("Churn Rate by Contract Type")
    ax3.set_xlim(0, 60)

    # Churn by internet service
    ax4 = fig.add_subplot(gs[1, 1])
    ct2 = df.groupby("InternetService")["Churn"].mean().sort_values(ascending=False)
    ax4.bar(ct2.index, ct2.values * 100, color=["#E74C3C","#3498DB","#2ECC71"],
            edgecolor="white", width=0.5)
    ax4.set_ylabel("Churn Rate (%)"); ax4.set_title("Churn Rate by Internet Service")
    for i, v in enumerate(ct2.values):
        ax4.text(i, v * 100 + 0.5, f"{v:.1%}", ha="center", fontsize=10, fontweight="bold")

    # Support calls boxplot
    ax5 = fig.add_subplot(gs[1, 2])
    data_box = [df[df.Churn == 0]["SupportCalls"], df[df.Churn == 1]["SupportCalls"]]
    bp = ax5.boxplot(data_box, patch_artist=True, widths=0.5,
                     medianprops=dict(color="white", linewidth=2))
    for patch, col in zip(bp["boxes"], COLORS):
        patch.set_facecolor(col)
    ax5.set_xticklabels(["No Churn", "Churn"])
    ax5.set_ylabel("Support Calls"); ax5.set_title("Support Calls by Churn")

    fig.text(0.5, -0.02,
             f"Dataset: {len(df):,} customers  |  Overall Churn Rate: {churn_rate:.1%}",
             ha="center", fontsize=11, color="#7F8C8D")
    save(fig, "01_overview_dashboard.png")


# ── 2. Correlation Heatmap ─────────────────────────────────────────────────────
def plot_correlation(df):
    num_cols = ["Tenure","MonthlyCharges","TotalCharges","SupportCalls",
                "NumProducts","SeniorCitizen","Partner","Dependents","Churn"]
    corr = df[num_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
                center=0, linewidths=0.5, ax=ax,
                cbar_kws={"shrink": 0.8}, annot_kws={"size": 9})
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold", pad=12)
    save(fig, "02_correlation_heatmap.png")


# ── 3. Churn by Demographics ───────────────────────────────────────────────────
def plot_demographics(df):
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle("Churn Rate by Demographics", fontsize=15, fontweight="bold")

    cats = [("Gender", "Gender"), ("SeniorCitizen","Senior Citizen"),
            ("Partner", "Has Partner"), ("Dependents","Has Dependents")]
    for ax, (col, title) in zip(axes, cats):
        rates = df.groupby(col)["Churn"].mean() * 100
        bars  = ax.bar(rates.index.astype(str), rates.values,
                       color=["#3498DB","#E67E22","#9B59B6","#1ABC9C"][:len(rates)],
                       edgecolor="white", width=0.5)
        for b in bars:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.5,
                    f"{b.get_height():.1f}%", ha="center", fontsize=11, fontweight="bold")
        ax.set_title(title, fontweight="bold"); ax.set_ylabel("Churn Rate (%)")
        ax.set_ylim(0, max(rates.values) * 1.25)
    plt.tight_layout()
    save(fig, "03_demographics_churn.png")


# ── 4. Payment & Billing ───────────────────────────────────────────────────────
def plot_payment(df):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Churn by Payment & Billing", fontsize=15, fontweight="bold")

    # Payment method
    ax = axes[0]
    pm = df.groupby("PaymentMethod")["Churn"].mean().sort_values(ascending=True) * 100
    colors = sns.color_palette("flare", len(pm))
    bars = ax.barh(pm.index, pm.values, color=colors, edgecolor="white", height=0.5)
    for b in bars:
        ax.text(b.get_width() + 0.3, b.get_y() + b.get_height()/2,
                f"{b.get_width():.1f}%", va="center", fontsize=10)
    ax.set_xlabel("Churn Rate (%)"); ax.set_title("By Payment Method")

    # Paperless billing
    ax2 = axes[1]
    pb = df.groupby("PaperlessBilling")["Churn"].mean() * 100
    ax2.bar(["Traditional", "Paperless"], pb.values,
            color=["#2ECC71", "#E74C3C"], edgecolor="white", width=0.4)
    for i, v in enumerate(pb.values):
        ax2.text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Churn Rate (%)"); ax2.set_title("By Billing Type")

    plt.tight_layout()
    save(fig, "04_payment_billing.png")


# ── 5. Charges vs Tenure Scatter ──────────────────────────────────────────────
def plot_scatter(df):
    fig, ax = plt.subplots(figsize=(10, 7))
    for val, label, col, marker in [(0,"No Churn","#2ECC71","o"), (1,"Churn","#E74C3C","X")]:
        sub = df[df.Churn == val]
        ax.scatter(sub["Tenure"], sub["MonthlyCharges"],
                   c=col, label=label, alpha=0.35, s=20, marker=marker)
    ax.set_xlabel("Tenure (months)"); ax.set_ylabel("Monthly Charges ($)")
    ax.set_title("Tenure vs Monthly Charges — Churn Overlay",
                 fontsize=14, fontweight="bold")
    ax.legend(markerscale=2)
    save(fig, "05_scatter_tenure_charges.png")


# ── 6. Support Calls Deep-Dive ─────────────────────────────────────────────────
def plot_support(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Support Calls Analysis", fontsize=15, fontweight="bold")

    ax = axes[0]
    calls_churn = df.groupby("SupportCalls")["Churn"].mean() * 100
    ax.plot(calls_churn.index, calls_churn.values, marker="o",
            color="#E74C3C", linewidth=2.5, markersize=7)
    ax.fill_between(calls_churn.index, calls_churn.values, alpha=0.15, color="#E74C3C")
    ax.set_xlabel("Number of Support Calls"); ax.set_ylabel("Churn Rate (%)")
    ax.set_title("Churn Rate by Support Calls Count")

    ax2 = axes[1]
    vol = df.groupby(["SupportCalls", "Churn"]).size().unstack(fill_value=0)
    vol.plot(kind="bar", ax=ax2, color=COLORS, edgecolor="white", width=0.7)
    ax2.set_xlabel("Support Calls"); ax2.set_ylabel("Customers")
    ax2.set_title("Customer Volume by Support Calls"); ax2.legend(["No Churn","Churn"])
    ax2.tick_params(axis="x", rotation=0)

    plt.tight_layout()
    save(fig, "06_support_calls.png")


# ── 7. Revenue Impact ─────────────────────────────────────────────────────────
def plot_revenue(df):
    churned_rev    = df[df.Churn == 1]["MonthlyCharges"].sum()
    retained_rev   = df[df.Churn == 0]["MonthlyCharges"].sum()
    avg_churn_rev  = df[df.Churn == 1]["MonthlyCharges"].mean()
    avg_retain_rev = df[df.Churn == 0]["MonthlyCharges"].mean()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Revenue Impact Analysis", fontsize=15, fontweight="bold")

    ax = axes[0]
    ax.bar(["Retained","Churned"], [retained_rev, churned_rev],
           color=["#2ECC71","#E74C3C"], edgecolor="white", width=0.5)
    ax.set_ylabel("Total Monthly Revenue ($)")
    ax.set_title("Monthly Revenue: Retained vs Churned")
    for i, v in enumerate([retained_rev, churned_rev]):
        ax.text(i, v + 500, f"${v:,.0f}", ha="center", fontsize=11, fontweight="bold")

    ax2 = axes[1]
    ax2.bar(["Retained","Churned"], [avg_retain_rev, avg_churn_rev],
            color=["#2ECC71","#E74C3C"], edgecolor="white", width=0.5)
    ax2.set_ylabel("Avg Monthly Charge ($)")
    ax2.set_title("Avg Monthly Charge per Customer")
    for i, v in enumerate([avg_retain_rev, avg_churn_rev]):
        ax2.text(i, v + 0.5, f"${v:.2f}", ha="center", fontsize=11, fontweight="bold")

    plt.tight_layout()
    save(fig, "07_revenue_impact.png")


def run_eda(df):
    print("\n📊  Running EDA & Generating Visualizations...")
    plot_overview(df)
    plot_correlation(df)
    plot_demographics(df)
    plot_payment(df)
    plot_scatter(df)
    plot_support(df)
    plot_revenue(df)
    print("  All EDA plots saved.\n")


if __name__ == "__main__":
    df = pd.read_csv("data/telecom_churn.csv")
    run_eda(df)
