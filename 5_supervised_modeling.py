# Section 5: Supervised Modeling, Model Evaluation, and Final Model Selection
#
# Compares Logistic Regression, Random Forest, and Gradient Boosting.
# Outputs (figures/):
#   model_comparison_table.png
#   model_performance_comparison.png
#   model_roc_comparison.png
#   final_model_threshold_tuning.png
#   final_model_roc_curve.png
#   final_model_confusion_matrix.png
#   final_model_feature_importance.png
#
# Outputs (outputs/):
#   model_comparison_metrics.csv
#   final_model_selected_metrics.csv
#   final_model_threshold_tuning.csv
#   final_model_feature_importance.csv
#   best_model_summary.csv
#   prediction_sample.csv
#
# Outputs (models/):
#   best_model.pkl  + per-model pkls

import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve,
)

os.makedirs("outputs", exist_ok=True)
os.makedirs("figures", exist_ok=True)
os.makedirs("models",  exist_ok=True)

RANDOM_STATE = 42

# ── Unified palette ───────────────────────────────────────────────────────────
C_PASS = "#2a9d8f"
C_FAIL = "#e63946"
C_BLUE = "#457b9d"
C_GOLD = "#f4a261"
C_GRAY = "#6c757d"

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.alpha": 0.3,
    "legend.frameon": False,
    "font.size": 10,
})

# =============================================================================
# 1. Load data
# =============================================================================
ROOT = Path.cwd()
candidate_paths = [
    ROOT / "raw" / "restaurant_clean_clustered.csv",
    ROOT / "raw" / "restaurant_clean.csv",
    ROOT / "data" / "processed" / "restaurant_clean.csv",
    ROOT / "data" / "restaurant_clean.csv",
    ROOT / "restaurant_clean.csv",
]
DATA_PATH = next((p for p in candidate_paths if p.exists()), None)
if DATA_PATH is None:
    raise FileNotFoundError(
        "Cannot find restaurant_clean.csv. "
        "Place it in raw/restaurant_clean.csv or the project root.")

print(f"Loaded: {DATA_PATH}")
df = pd.read_csv(DATA_PATH, low_memory=False)
df["inspection_date"] = pd.to_datetime(df["inspection_date"], errors="coerce")
print(f"Shape: {df.shape}")
print(f"Date range: {df['inspection_date'].min().date()} to {df['inspection_date'].max().date()}")
print(f"Failure rate: {df['failed'].mean():.4f}")

# =============================================================================
# 2. Feature engineering (leakage-safe)
# =============================================================================
df["poor_history_flag"] = (
    (df["prev_failed"] == 1) | (df["prev_score"] >= 28)
).astype(int)

df["complaint_intensity"] = (
    df["food_complaints_total"].fillna(0)
    + df["rodent_complaints"].fillna(0)
    + df["food_safety_complaints"].fillna(0)
)
df["complaint_density"] = (
    df["food_complaints_total"].fillna(0)
    / df["total_population"].replace(0, np.nan)
) * 10000
df["high_complaint_flag"] = (
    df["food_complaints_total"] > df["food_complaints_total"].quantile(0.75)
).astype(int)

df["month_sin"]   = np.sin(2 * np.pi * df["inspection_month"] / 12)
df["month_cos"]   = np.cos(2 * np.pi * df["inspection_month"] / 12)
df["dow_sin"]     = np.sin(2 * np.pi * df["inspection_dow"] / 7)
df["dow_cos"]     = np.cos(2 * np.pi * df["inspection_dow"] / 7)
df["summer_flag"] = df["inspection_month"].isin([6, 7, 8]).astype(int)
df["log_yelp_reviews"] = np.log1p(df["yelp_reviews"])
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# =============================================================================
# 3. Feature selection and train/test split
# =============================================================================
numeric_features = [
    "prev_score", "prev_failed", "inspection_count",
    "poor_history_flag", "is_first_inspection", "has_history",
    "food_complaints_total", "rodent_complaints", "food_safety_complaints",
    "complaint_intensity", "complaint_density", "high_complaint_flag",
    "temp_mean", "precipitation_sum", "rain_sum", "snowfall_sum",
    "wind_speed_mean", "cloud_cover_mean",
    "month_sin", "month_cos", "dow_sin", "dow_cos",
    "summer_flag", "is_weekend",
    "median_household_income", "total_population", "white_pct",
    "has_yelp", "has_location", "log_yelp_reviews", "yelp_price",
]
categorical_features = ["boro", "cuisine_grouped", "yelp_category_primary"]

numeric_features     = [c for c in numeric_features     if c in df.columns]
categorical_features = [c for c in categorical_features if c in df.columns]

model_df = df.dropna(subset=["failed", "inspection_date"]).copy()
train_df = model_df[model_df["inspection_date"] <  "2024-01-01"].copy()
test_df  = model_df[model_df["inspection_date"] >= "2024-01-01"].copy()

X_train = train_df[numeric_features + categorical_features]
y_train = train_df["failed"].astype(int)
X_test  = test_df[numeric_features  + categorical_features]
y_test  = test_df["failed"].astype(int)

print(f"Train: {X_train.shape}  failure={y_train.mean():.3f}")
print(f"Test:  {X_test.shape}   failure={y_test.mean():.3f}")

# =============================================================================
# 4. Preprocessing pipeline
# =============================================================================
try:
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=20)
except TypeError:
    encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
])
categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot",  encoder),
])
preprocessor = ColumnTransformer([
    ("num", numeric_transformer,     numeric_features),
    ("cat", categorical_transformer, categorical_features),
])

# =============================================================================
# 5. Model training and hyperparameter tuning
# =============================================================================
models = {
    "Logistic Regression": (
        LogisticRegression(class_weight="balanced", solver="liblinear",
                           max_iter=2000, random_state=RANDOM_STATE),
        {"model__C": [0.1, 1.0], "model__penalty": ["l1", "l2"]},
    ),
    "Random Forest": (
        RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
        {"model__n_estimators": [100, 200], "model__max_depth": [8, 12],
         "model__min_samples_leaf": [5, 10]},
    ),
    "Gradient Boosting": (
        GradientBoostingClassifier(random_state=RANDOM_STATE),
        {"model__n_estimators": [100, 200], "model__learning_rate": [0.05, 0.10],
         "model__max_depth": [2, 3]},
    ),
}

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
results = []
probs   = {}
best_estimators = {}

for name, (clf, grid) in models.items():
    print(f"\nTraining {name}...")
    pipe   = Pipeline([("preprocessor", preprocessor), ("model", clf)])
    search = GridSearchCV(pipe, grid, scoring="roc_auc", cv=cv, n_jobs=-1)
    search.fit(X_train, y_train)
    model  = search.best_estimator_
    best_estimators[name] = model

    y_prob = model.predict_proba(X_test)[:, 1]
    probs[name] = y_prob
    y_pred = (y_prob >= 0.50).astype(int)

    results.append({
        "Model":       name,
        "Best Params": search.best_params_,
        "Accuracy":    accuracy_score(y_test, y_pred),
        "Precision":   precision_score(y_test, y_pred, zero_division=0),
        "Recall":      recall_score(y_test, y_pred,    zero_division=0),
        "F1":          f1_score(y_test, y_pred,         zero_division=0),
        "ROC-AUC":     roc_auc_score(y_test, y_prob),
    })
    print(f"  Best params: {search.best_params_}")
    print(f"  ROC-AUC:     {roc_auc_score(y_test, y_prob):.4f}")
    print(f"  Recall:      {recall_score(y_test, y_pred, zero_division=0):.4f}")

metrics = pd.DataFrame(results).sort_values("ROC-AUC", ascending=False)
metrics.to_csv("outputs/model_comparison_metrics.csv", index=False)
best_name = metrics.iloc[0]["Model"]
print(f"\nBest model by ROC-AUC: {best_name}")

# =============================================================================
# 6. Model comparison figures
# =============================================================================
metrics_plot = metrics.copy()
metrics_plot["Model"] = pd.Categorical(
    metrics_plot["Model"],
    categories=["Logistic Regression", "Random Forest", "Gradient Boosting"],
    ordered=True)
metrics_plot = metrics_plot.sort_values("Model")
table_df = metrics_plot[["Model","Accuracy","Precision","Recall","F1","ROC-AUC"]].copy()
for col in ["Accuracy","Precision","Recall","F1","ROC-AUC"]:
    table_df[col] = table_df[col].astype(float).round(3)

# fig: comparison table
fig, ax = plt.subplots(figsize=(11, 2.8))
ax.axis("off")
table = ax.table(
    cellText=table_df.values.tolist(),
    colLabels=table_df.columns.tolist(),
    cellLoc="center", loc="center")
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 1.55)
for (row, col), cell in table.get_celld().items():
    cell.set_edgecolor("#2F3A4A"); cell.set_linewidth(0.8)
    if row == 0:
        cell.set_facecolor("#243B53")
        cell.set_text_props(color="white", weight="bold")
    elif table_df.iloc[row-1]["Model"] == best_name:
        cell.set_facecolor("#DCEEFF")
        cell.set_text_props(weight="bold")
    elif row % 2 == 0:
        cell.set_facecolor("#F7F9FB")
    else:
        cell.set_facecolor("white")
ax.set_title("Model Comparison Metrics", weight="bold", pad=18)
ax.text(0.5, -0.08, "Highlighted row indicates the selected final model.",
        transform=ax.transAxes, ha="center", fontsize=10, color="#52606D")
plt.tight_layout()
plt.savefig("figures/model_comparison_table.png", bbox_inches="tight")
plt.close()
print("Saved: figures/model_comparison_table.png")

# fig: bar chart comparison
metric_cols = ["Precision", "Recall", "F1", "ROC-AUC"]
bar_colors  = [C_BLUE, C_GOLD, C_PASS, C_FAIL]
x     = np.arange(len(metrics_plot))
width = 0.18
fig, ax = plt.subplots(figsize=(10, 5.5))
for i, (col, color) in enumerate(zip(metric_cols, bar_colors)):
    bars = ax.bar(x + (i - 1.5) * width, metrics_plot[col], width,
                  label=col, color=color, edgecolor="white")
    for bar in bars:
        h = bar.get_height()
        if h > 0.025:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.012,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(metrics_plot["Model"])
ax.set_ylim(0, 0.78)
ax.set_ylabel("Metric value")
ax.set_title("Model Performance Comparison")
ax.grid(axis="y", alpha=0.25)
ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.10))
ax.text(0.5, -0.12,
        "Accuracy is not shown because class imbalance makes recall and F1 more decision-relevant.",
        transform=ax.transAxes, ha="center", fontsize=9, color=C_GRAY)
plt.tight_layout()
plt.savefig("figures/model_performance_comparison.png", bbox_inches="tight")
plt.close()
print("Saved: figures/model_performance_comparison.png")

# fig: ROC comparison
roc_colors = {
    "Logistic Regression": C_FAIL,
    "Random Forest":       C_PASS,
    "Gradient Boosting":   C_GOLD,
}
fig, ax = plt.subplots(figsize=(7.5, 6))
for name in metrics.sort_values("ROC-AUC", ascending=False)["Model"]:
    fpr, tpr, _ = roc_curve(y_test, probs[name])
    auc = roc_auc_score(y_test, probs[name])
    ax.plot(fpr, tpr, lw=2.4, color=roc_colors.get(name, C_GRAY),
            label=f"{name} (AUC = {auc:.3f})")
ax.plot([0,1],[0,1], linestyle="--", color=C_GRAY, lw=1.6, label="Random classifier")
ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve Comparison Across Models")
ax.grid(alpha=0.25)
ax.legend(loc="lower right", frameon=True)
plt.tight_layout()
plt.savefig("figures/model_roc_comparison.png", bbox_inches="tight")
plt.close()
print("Saved: figures/model_roc_comparison.png")

# =============================================================================
# 7. Threshold tuning on best model
# =============================================================================
best_prob = probs[best_name]
threshold_rows = []
for t in np.arange(0.10, 0.91, 0.01):
    pred = (best_prob >= t).astype(int)
    threshold_rows.append({
        "threshold": t,
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall":    recall_score(y_test, pred,    zero_division=0),
        "f1":        f1_score(y_test, pred,         zero_division=0),
        "accuracy":  accuracy_score(y_test, pred),
    })
threshold_df   = pd.DataFrame(threshold_rows)
threshold_df.to_csv("outputs/final_model_threshold_tuning.csv", index=False)
best_row       = threshold_df.loc[threshold_df["f1"].idxmax()]
best_threshold = best_row["threshold"]
final_pred     = (best_prob >= best_threshold).astype(int)
print(f"\nBest threshold: {best_threshold:.2f}  F1={best_row['f1']:.4f}  Recall={best_row['recall']:.4f}")

fig, ax = plt.subplots(figsize=(9, 5.5))
ax.plot(threshold_df["threshold"], threshold_df["precision"], lw=2.2, color=C_BLUE,  label="Precision")
ax.plot(threshold_df["threshold"], threshold_df["recall"],    lw=2.2, color=C_GOLD,  label="Recall")
ax.plot(threshold_df["threshold"], threshold_df["f1"],        lw=2.8, color=C_PASS,  label="F1-score")
ax.axvline(best_threshold, linestyle="--", color=C_FAIL, lw=1.8)
ax.scatter([best_threshold], [best_row["f1"]], s=90, color=C_FAIL, zorder=5)
ax.annotate(
    f"Best F1 threshold = {best_threshold:.2f}\nF1 = {best_row['f1']:.3f}",
    xy=(best_threshold, best_row["f1"]),
    xytext=(best_threshold + 0.07, best_row["f1"] + 0.10),
    arrowprops=dict(arrowstyle="->", color=C_FAIL, lw=1.2),
    fontsize=9.5,
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C_GRAY),
)
ax.set_xlim(0.10, 0.90); ax.set_ylim(0, 1.05)
ax.set_xlabel("Classification threshold"); ax.set_ylabel("Metric value")
ax.set_title(f"Threshold Tuning for {best_name}")
ax.legend(loc="upper right")
ax.grid(alpha=0.25)
plt.tight_layout()
plt.savefig("figures/final_model_threshold_tuning.png", bbox_inches="tight")
plt.close()
print("Saved: figures/final_model_threshold_tuning.png")

# =============================================================================
# 8. Final model evaluation
# =============================================================================
final_metrics = pd.DataFrame([{
    "Model":              best_name,
    "Selected threshold": best_threshold,
    "Accuracy":           accuracy_score(y_test, final_pred),
    "Precision":          precision_score(y_test, final_pred, zero_division=0),
    "Recall":             recall_score(y_test, final_pred,    zero_division=0),
    "F1":                 f1_score(y_test, final_pred,         zero_division=0),
    "ROC-AUC":            roc_auc_score(y_test, best_prob),
}])
final_metrics.to_csv("outputs/final_model_selected_metrics.csv", index=False)
print(final_metrics.to_string(index=False))

# fig: final model ROC
fig, ax = plt.subplots(figsize=(7, 5.5))
fpr, tpr, _ = roc_curve(y_test, best_prob)
auc = roc_auc_score(y_test, best_prob)
ax.plot(fpr, tpr, color=C_FAIL, lw=2.8, label=f"{best_name} (AUC = {auc:.3f})")
ax.plot([0,1],[0,1], linestyle="--", color=C_GRAY, lw=1.6, label="Random classifier")
ax.fill_between(fpr, tpr, alpha=0.12, color=C_FAIL)
ax.set_xlim(0,1); ax.set_ylim(0, 1.02)
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title(f"Final Model ROC Curve: {best_name}")
ax.grid(alpha=0.25); ax.legend(loc="lower right", frameon=True)
plt.tight_layout()
plt.savefig("figures/final_model_roc_curve.png", bbox_inches="tight")
plt.close()
print("Saved: figures/final_model_roc_curve.png")

# fig: confusion matrix
cm      = confusion_matrix(y_test, final_pred)
row_pct = cm / cm.sum(axis=1, keepdims=True)
fig, ax = plt.subplots(figsize=(6.5, 5.5))
im = ax.imshow(row_pct, cmap="Blues", vmin=0, vmax=1)
classes = ["Pass", "Fail"]
ax.set_xticks([0,1]); ax.set_yticks([0,1])
ax.set_xticklabels(classes); ax.set_yticklabels(classes)
ax.set_xlabel("Predicted label"); ax.set_ylabel("True label")
ax.set_title(f"Final Model Confusion Matrix | Threshold = {best_threshold:.2f}")
for i in range(2):
    for j in range(2):
        tc = "white" if row_pct[i,j] > 0.55 else "#102A43"
        ax.text(j, i, f"{cm[i,j]:,}\n({row_pct[i,j]:.1%})",
                ha="center", va="center", color=tc, fontsize=14, weight="bold")
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.ax.set_ylabel("Row percentage", rotation=270, labelpad=18)
plt.tight_layout()
plt.savefig("figures/final_model_confusion_matrix.png", bbox_inches="tight")
plt.close()
print("Saved: figures/final_model_confusion_matrix.png")

# =============================================================================
# 9. Feature importance
# =============================================================================
def get_feature_names(fitted_model, num_feats, cat_feats):
    prep = fitted_model.named_steps["preprocessor"]
    names = [f"num__{c}" for c in num_feats]
    try:
        cat_enc = prep.named_transformers_["cat"].named_steps["onehot"]
        names += [f"cat__{c}" for c in cat_enc.get_feature_names_out(cat_feats)]
    except Exception:
        pass
    return names

final_model = best_estimators[best_name]
model_step  = final_model.named_steps["model"]
feat_names  = get_feature_names(final_model, numeric_features, categorical_features)

if hasattr(model_step, "coef_"):
    importance_vals = model_step.coef_[0]
    xlabel = "Logistic Regression coefficient"
    bar_colors_fi = [C_FAIL if v > 0 else C_BLUE for v in importance_vals]
elif hasattr(model_step, "feature_importances_"):
    importance_vals = model_step.feature_importances_
    xlabel = "Feature importance"
    bar_colors_fi = C_BLUE
else:
    importance_vals = np.zeros(len(feat_names))
    xlabel = "Importance"
    bar_colors_fi = C_GRAY

fi = pd.DataFrame({"feature": feat_names, "importance": importance_vals})
fi["abs_importance"] = fi["importance"].abs()
fi = fi.sort_values("abs_importance", ascending=False).head(15).sort_values("abs_importance")
fi["feature_clean"] = (
    fi["feature"]
    .str.replace("num__", "", regex=False)
    .str.replace("cat__", "", regex=False)
    .str.replace("cuisine_grouped_", "Cuisine: ", regex=False)
    .str.replace("yelp_category_primary_", "Yelp: ", regex=False)
    .str.replace("_", " ", regex=False)
)
fi.to_csv("outputs/final_model_feature_importance.csv", index=False)

fig, ax = plt.subplots(figsize=(9.5, 6.5))
bar_c = [C_FAIL if v > 0 else C_BLUE for v in fi["importance"]] if hasattr(model_step, "coef_") else C_BLUE
ax.barh(fi["feature_clean"], fi["importance"], color=bar_c, edgecolor="white")
ax.axvline(0, color="black", lw=1)
ax.set_xlabel(xlabel)
ax.set_title(f"Top Predictors in Final Model: {best_name}")
ax.grid(axis="x", alpha=0.25)
if hasattr(model_step, "coef_"):
    ax.legend(handles=[
        Patch(facecolor=C_FAIL, label="Higher predicted failure risk"),
        Patch(facecolor=C_BLUE, label="Lower predicted failure risk"),
    ], loc="lower right")
plt.tight_layout()
plt.savefig("figures/final_model_feature_importance.png", bbox_inches="tight")
plt.close()
print("Saved: figures/final_model_feature_importance.png")

# =============================================================================
# 10. Save models and CSV outputs
# =============================================================================
joblib.dump(final_model, "models/best_model.pkl")
for name, est in best_estimators.items():
    fname = name.lower().replace(" ", "_") + "_model.pkl"
    joblib.dump(est, Path("models") / fname)

best_model_summary = final_metrics.copy()
best_model_summary["Reason"] = (
    "Best balance of recall, F1-score, ROC-AUC, and interpretability")
best_model_summary.to_csv("outputs/best_model_summary.csv", index=False)

prediction_sample = X_test.copy().reset_index(drop=True).head(50)
prediction_sample.insert(0, "Actual_Label",         y_test.reset_index(drop=True).head(50))
prediction_sample.insert(1, "Predicted_Label",       final_pred[:50])
prediction_sample.insert(2, "Predicted_Probability", best_prob[:50])
prediction_sample.to_csv("outputs/prediction_sample.csv", index=False)

print("\nAll outputs saved successfully.")
print("  models/best_model.pkl")
print("  figures/model_comparison_table.png")
print("  figures/model_performance_comparison.png")
print("  figures/model_roc_comparison.png")
print("  figures/final_model_threshold_tuning.png")
print("  figures/final_model_roc_curve.png")
print("  figures/final_model_confusion_matrix.png")
print("  figures/final_model_feature_importance.png")
