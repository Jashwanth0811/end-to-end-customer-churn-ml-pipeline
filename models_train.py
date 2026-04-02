"""
models.py
Trains multiple classifiers, compares them, and generates model evaluation plots.
"""
import warnings
warnings.filterwarnings("ignore")

import pickle, time
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from pathlib import Path
from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import (RandomForestClassifier,
                                     GradientBoostingClassifier,
                                     AdaBoostClassifier)
from sklearn.tree            import DecisionTreeClassifier
from sklearn.svm             import SVC
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.naive_bayes     import GaussianNB
from sklearn.metrics         import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score,
    confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score)

BG        = "#F8F9FA"
FONT_DARK = "#2C3E50"
OUT       = Path("visualizations")
OUT.mkdir(exist_ok=True)

MODEL_PATH = Path("models")
MODEL_PATH.mkdir(exist_ok=True)

plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor":"#FFFFFF",
                     "text.color": FONT_DARK, "grid.color": "#DEE2E6"})


# ── Define candidate models ───────────────────────────────────────────────────
def get_models():
    return {
        "Logistic Regression" : LogisticRegression(max_iter=500, C=1.0, random_state=42),
        "Decision Tree"       : DecisionTreeClassifier(max_depth=6, random_state=42),
        "Random Forest"       : RandomForestClassifier(n_estimators=200, max_depth=10,
                                                       random_state=42, n_jobs=-1),
        "Gradient Boosting"   : GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                                            learning_rate=0.05, random_state=42),
        "AdaBoost"            : AdaBoostClassifier(n_estimators=100, random_state=42),
        "KNN"                 : KNeighborsClassifier(n_neighbors=7),
        "Naive Bayes"         : GaussianNB(),
        "SVM"                 : SVC(probability=True, kernel="rbf", C=1.0, random_state=42),
    }


# ── Train & evaluate all models ───────────────────────────────────────────────
def train_all(X_train, X_test, y_train, y_test):
    models  = get_models()
    results = []

    print("\n🤖  Training & Evaluating Models...")
    print(f"{'Model':<25} {'Acc':>6} {'AUC':>6} {'F1':>6} {'Prec':>6} {'Rec':>6} {'Time':>7}")
    print("-" * 70)

    trained = {}
    for name, clf in models.items():
        t0 = time.time()
        clf.fit(X_train, y_train)
        elapsed = time.time() - t0

        y_pred  = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1]

        acc  = accuracy_score(y_test, y_pred)
        auc  = roc_auc_score(y_test, y_proba)
        f1   = f1_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec  = recall_score(y_test, y_pred)

        print(f"{name:<25} {acc:>6.3f} {auc:>6.3f} {f1:>6.3f} {prec:>6.3f} {rec:>6.3f} {elapsed:>6.2f}s")
        results.append({"Model": name, "Accuracy": acc, "ROC-AUC": auc,
                        "F1": f1, "Precision": prec, "Recall": rec})
        trained[name] = clf

    results_df = pd.DataFrame(results).sort_values("ROC-AUC", ascending=False)
    best_name  = results_df.iloc[0]["Model"]
    best_model = trained[best_name]

    # Save best model
    with open(MODEL_PATH / "best_model.pkl", "wb") as f:
        pickle.dump({"model": best_model, "name": best_name}, f)
    results_df.to_csv(MODEL_PATH / "model_comparison.csv", index=False)

    print(f"\n  🏆  Best model: {best_name} (AUC={results_df.iloc[0]['ROC-AUC']:.4f})\n")
    return trained, results_df, best_name


# ── Visualization 1: Model Comparison Bar Chart ───────────────────────────────
def plot_model_comparison(results_df):
    metrics = ["Accuracy", "ROC-AUC", "F1", "Precision", "Recall"]
    df_melt = results_df.melt(id_vars="Model", value_vars=metrics,
                               var_name="Metric", value_name="Score")

    fig, ax = plt.subplots(figsize=(14, 7))
    bar_w   = 0.15
    x       = np.arange(len(results_df))
    palette = ["#3498DB","#E74C3C","#2ECC71","#F39C12","#9B59B6"]

    for i, (metric, col) in enumerate(zip(metrics, palette)):
        vals = results_df[metric].values
        bars = ax.bar(x + i * bar_w, vals, bar_w, label=metric, color=col,
                      edgecolor="white", alpha=0.9)

    ax.set_xticks(x + bar_w * 2)
    ax.set_xticklabels(results_df["Model"], rotation=25, ha="right", fontsize=10)
    ax.set_ylim(0.5, 1.02); ax.set_ylabel("Score")
    ax.set_title("Model Comparison — All Metrics", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right"); ax.grid(axis="y", alpha=0.5)
    plt.tight_layout()
    path = OUT / "08_model_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  ✓  {path}")


# ── Visualization 2: ROC Curves ───────────────────────────────────────────────
def plot_roc_curves(trained, X_test, y_test):
    fig, ax = plt.subplots(figsize=(10, 8))
    palette = plt.cm.tab10(np.linspace(0, 1, len(trained)))

    for (name, clf), col in zip(trained.items(), palette):
        y_proba = clf.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        ax.plot(fpr, tpr, lw=2, color=col, label=f"{name} (AUC={auc:.3f})")

    ax.plot([0,1],[0,1], "k--", lw=1, alpha=0.5, label="Random")
    ax.fill_between([0,1],[0,1], alpha=0.05, color="gray")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Models", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9); ax.grid(alpha=0.4)
    path = OUT / "09_roc_curves.png"
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  ✓  {path}")


# ── Visualization 3: Best Model Detailed Report ───────────────────────────────
def plot_best_model_detail(best_name, best_clf, X_test, y_test, feature_names):
    y_pred  = best_clf.predict(X_test)
    y_proba = best_clf.predict_proba(X_test)[:, 1]

    fig = plt.figure(figsize=(16, 12), facecolor=BG)
    fig.suptitle(f"Best Model Deep-Dive: {best_name}",
                 fontsize=16, fontweight="bold", y=1.01)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.38)

    # Confusion Matrix
    ax0 = fig.add_subplot(gs[0, 0])
    cm  = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="RdYlGn", ax=ax0,
                linewidths=1, linecolor="white",
                xticklabels=["No Churn","Churn"],
                yticklabels=["No Churn","Churn"],
                annot_kws={"size": 14, "weight":"bold"})
    ax0.set_xlabel("Predicted"); ax0.set_ylabel("Actual")
    ax0.set_title("Confusion Matrix")

    # ROC
    ax1 = fig.add_subplot(gs[0, 1])
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    ax1.plot(fpr, tpr, color="#E74C3C", lw=2.5, label=f"AUC={auc:.4f}")
    ax1.fill_between(fpr, tpr, alpha=0.12, color="#E74C3C")
    ax1.plot([0,1],[0,1],"k--",lw=1)
    ax1.set_xlabel("FPR"); ax1.set_ylabel("TPR")
    ax1.set_title("ROC Curve"); ax1.legend()

    # Precision-Recall
    ax2 = fig.add_subplot(gs[0, 2])
    prec_c, rec_c, _ = precision_recall_curve(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)
    ax2.plot(rec_c, prec_c, color="#3498DB", lw=2.5, label=f"AP={ap:.4f}")
    ax2.fill_between(rec_c, prec_c, alpha=0.12, color="#3498DB")
    ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall Curve"); ax2.legend()

    # Score distribution
    ax3 = fig.add_subplot(gs[1, 0])
    for val, label, col in [(0,"No Churn","#2ECC71"),(1,"Churn","#E74C3C")]:
        scores = y_proba[y_test == val]
        ax3.hist(scores, bins=30, alpha=0.65, color=col, label=label, edgecolor="white")
    ax3.axvline(0.5, color="black", lw=1.5, linestyle="--", label="Threshold 0.5")
    ax3.set_xlabel("Predicted Probability"); ax3.set_ylabel("Count")
    ax3.set_title("Prediction Score Distribution"); ax3.legend()

    # Feature importances (tree-based) or coefficients
    ax4 = fig.add_subplot(gs[1, 1:])
    if hasattr(best_clf, "feature_importances_"):
        imp = pd.Series(best_clf.feature_importances_, index=feature_names)
    elif hasattr(best_clf, "coef_"):
        imp = pd.Series(np.abs(best_clf.coef_[0]), index=feature_names)
    else:
        imp = pd.Series(np.ones(len(feature_names)), index=feature_names)

    top_imp = imp.sort_values(ascending=True).tail(20)
    colors  = ["#E74C3C" if v > top_imp.median() else "#3498DB" for v in top_imp.values]
    ax4.barh(top_imp.index, top_imp.values, color=colors, edgecolor="white", height=0.7)
    ax4.set_xlabel("Importance Score")
    ax4.set_title("Top 20 Feature Importances")
    ax4.axvline(top_imp.median(), color="black", lw=1.2, linestyle="--", alpha=0.5)

    path = OUT / "10_best_model_detail.png"
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  ✓  {path}")


# ── Visualization 4: Threshold Tuning ─────────────────────────────────────────
def plot_threshold_analysis(best_clf, X_test, y_test):
    y_proba    = best_clf.predict_proba(X_test)[:, 1]
    thresholds = np.arange(0.05, 0.96, 0.02)
    f1s, precs, recs, accs = [], [], [], []

    for th in thresholds:
        yp = (y_proba >= th).astype(int)
        f1s.append(f1_score(y_test, yp, zero_division=0))
        precs.append(precision_score(y_test, yp, zero_division=0))
        recs.append(recall_score(y_test, yp, zero_division=0))
        accs.append(accuracy_score(y_test, yp))

    best_th = thresholds[np.argmax(f1s)]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(thresholds, f1s,   label="F1",        lw=2.5, color="#E74C3C")
    ax.plot(thresholds, precs, label="Precision",  lw=2,   color="#3498DB")
    ax.plot(thresholds, recs,  label="Recall",     lw=2,   color="#2ECC71")
    ax.plot(thresholds, accs,  label="Accuracy",   lw=2,   color="#F39C12", linestyle="--")
    ax.axvline(best_th, color="black", lw=1.5, linestyle=":", label=f"Best F1 threshold={best_th:.2f}")
    ax.axvline(0.50,   color="gray",  lw=1,   linestyle="--", alpha=0.6, label="Default 0.50")
    ax.set_xlabel("Threshold"); ax.set_ylabel("Score")
    ax.set_title("Threshold Tuning — Precision / Recall / F1 Trade-off",
                 fontsize=14, fontweight="bold")
    ax.legend(); ax.grid(alpha=0.4)
    plt.tight_layout()
    path = OUT / "11_threshold_tuning.png"
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  ✓  {path}")
    return float(best_th)


def run_models(X_train, X_test, y_train, y_test, feature_names):
    trained, results_df, best_name = train_all(X_train, X_test, y_train, y_test)
    best_clf = trained[best_name]

    print("📈  Generating Model Visualizations...")
    plot_model_comparison(results_df)
    plot_roc_curves(trained, X_test, y_test)
    plot_best_model_detail(best_name, best_clf, X_test, y_test, feature_names)
    best_threshold = plot_threshold_analysis(best_clf, X_test, y_test)
    print()

    return trained, results_df, best_name, best_clf, best_threshold


if __name__ == "__main__":
    from features import get_feature_matrix
    df = pd.read_csv("data/telecom_churn.csv")
    X_tr, X_te, y_tr, y_te, feats, _ = get_feature_matrix(df)
    run_models(X_tr, X_te, y_tr, y_te, feats)
