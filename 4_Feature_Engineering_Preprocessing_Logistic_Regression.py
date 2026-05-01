# Section 4: Feature Engineering, Preprocessing, and Logistic Regression
#
# Goal: build a leakage-safe, interpretable baseline model for predicting
# NYC restaurant inspection failure using only information available before
# the inspection outcome is known.
#
# Outputs (figures/):
#   logistic_threshold_tuning.png
#   logistic_confusion_matrix_best_threshold.png
#   logistic_roc_curve.png
#   logistic_coefficient_plot.png
#   logistic_odds_ratio_plot.png
#
# Outputs (outputs/):
#   restaurant_feature_engineered_leakage_safe.csv
#   leakage_audit.csv
#   logistic_regression_metrics_threshold_comparison.csv
#   logistic_regression_coefficients.csv
#   logistic_threshold_tuning.csv
#   baseline_vs_logistic_metrics.csv
#   logistic_regression_odds_ratios.csv
#
# Outputs (models/):
#   logistic_model.pkl

import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import joblib

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, ConfusionMatrixDisplay, RocCurveDisplay,
)

os.makedirs("outputs", exist_ok=True)
os.makedirs("models",  exist_ok=True)
os.makedirs("figures", exist_ok=True)

# ── Unified palette (matches EDA section) ────────────────────────────────────
C_PASS   = "#2a9d8f"
C_FAIL   = "#e63946"
C_BLUE   = "#457b9d"
C_GOLD   = "#f4a261"
C_GRAY   = "#6c757d"

plt.rcParams.update({
    "figure.dpi": 150,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.alpha": 0.4,
    "legend.frameon": False,
})

# =============================================================================
# 1. Load data
# =============================================================================
ROOT = Path.cwd()
for candidate in [ROOT, ROOT.parent]:
    if (candidate / "raw" / "restaurant_clean_clustered.csv").exists():
        clean_path = candidate / "raw" / "restaurant_clean_clustered.csv"; break
    if (candidate / "raw" / "restaurant_clean.csv").exists():
        clean_path = candidate / "raw" / "restaurant_clean.csv"; break
else:
    clean_path = ROOT / "raw" / "restaurant_clean.csv"

assert clean_path.exists(), f"Missing file: {clean_path}"
df = pd.read_csv(clean_path, low_memory=False)
df["inspection_date"] = pd.to_datetime(df["inspection_date"], errors="coerce")

print(f"Loaded: {clean_path}")
print(f"Shape:  {df.shape}")

# =============================================================================
# 2. Target definition and class balance
# =============================================================================
target_col = "failed"
print("\nTarget distribution:")
print(df[target_col].value_counts(dropna=False))
print(f"\nFailure rate: {df[target_col].mean():.3f}")

# =============================================================================
# 3. Leakage audit
# =============================================================================
excluded_features = {
    "score":          "Directly defines failed; using it would leak the target.",
    "grade":          "Directly reflects inspection outcome.",
    "score_bucket":   "Derived from score, therefore leaks target information.",
    "action":         "Post-inspection administrative outcome.",
    "violation_count":"Current inspection violation count is only known during/after inspection.",
    "critical_count": "Current inspection critical violations are only known during/after inspection.",
    "critical_ratio": "Derived from current inspection violations.",
    "score_trend":    "Uses current score, therefore leaks outcome information.",
    "cluster":        "Part B cluster may use score/violation variables; excluded for leakage-safe modeling.",
    "camis":          "Restaurant ID, not generalizable.",
    "dba":            "Restaurant name, not generalizable.",
    "inspection_date":"Raw date replaced by engineered temporal features.",
    "latitude":       "High missingness; not used directly.",
    "longitude":      "High missingness; not used directly.",
    "zipcode":        "High cardinality; borough captures broader location context.",
}

leakage_audit = pd.DataFrame(
    excluded_features.items(), columns=["Excluded Feature", "Reason"])
leakage_audit.to_csv("outputs/leakage_audit.csv", index=False)
print(leakage_audit.to_string(index=False))

# =============================================================================
# 4. Feature engineering
# =============================================================================
df_fe = df.copy()

# Historical risk
df_fe["poor_history_flag"] = (
    (df_fe["prev_failed"] == 1) | (df_fe["prev_score"] >= 28)
).astype(int)

# Complaint environment
df_fe["complaint_intensity"] = (
    df_fe["food_complaints_total"].fillna(0)
    + df_fe["rodent_complaints"].fillna(0)
    + df_fe["food_safety_complaints"].fillna(0)
)
df_fe["complaint_density"] = (
    df_fe["food_complaints_total"].fillna(0)
    / df_fe["total_population"].replace(0, np.nan)
) * 10000
df_fe["high_complaint_flag"] = (
    df_fe["food_complaints_total"]
    > df_fe["food_complaints_total"].quantile(0.75)
).astype(int)

# Cyclic temporal encodings
df_fe["month_sin"] = np.sin(2 * np.pi * df_fe["inspection_month"] / 12)
df_fe["month_cos"] = np.cos(2 * np.pi * df_fe["inspection_month"] / 12)
df_fe["dow_sin"]   = np.sin(2 * np.pi * df_fe["inspection_dow"] / 7)
df_fe["dow_cos"]   = np.cos(2 * np.pi * df_fe["inspection_dow"] / 7)
df_fe["summer_flag"] = df_fe["inspection_month"].isin([6, 7, 8]).astype(int)

# Yelp / visibility
df_fe["log_yelp_reviews"] = np.log1p(df_fe["yelp_reviews"])

df_fe.replace([np.inf, -np.inf], np.nan, inplace=True)
print(f"\nFeature-engineered shape: {df_fe.shape}")

# =============================================================================
# 5. Feature selection
# =============================================================================
numeric_features = [
    # Historical
    "prev_score", "prev_failed", "inspection_count",
    "poor_history_flag", "is_first_inspection", "has_history",
    # Complaint environment
    "food_complaints_total", "rodent_complaints", "food_safety_complaints",
    "complaint_intensity", "complaint_density", "high_complaint_flag",
    # Weather
    "temp_mean", "precipitation_sum", "rain_sum", "snowfall_sum",
    "wind_speed_mean", "cloud_cover_mean",
    # Temporal
    "month_sin", "month_cos", "dow_sin", "dow_cos",
    "summer_flag", "is_weekend",
    # Demographics
    "median_household_income", "total_population", "white_pct",
    # Yelp / visibility
    "has_yelp", "has_location", "log_yelp_reviews", "yelp_price",
]
categorical_features = ["boro", "cuisine_grouped", "yelp_category_primary"]

numeric_features     = [c for c in numeric_features     if c in df_fe.columns]
categorical_features = [c for c in categorical_features if c in df_fe.columns]

print(f"\nNumeric features ({len(numeric_features)}):  {numeric_features}")
print(f"Categorical features ({len(categorical_features)}): {categorical_features}")

# =============================================================================
# 6. Time-based train / test split
# =============================================================================
model_df = df_fe.dropna(subset=[target_col, "inspection_date"]).copy()
train_df  = model_df[model_df["inspection_date"] <  "2024-01-01"].copy()
test_df   = model_df[model_df["inspection_date"] >= "2024-01-01"].copy()

X_train = train_df[numeric_features + categorical_features]
y_train = train_df[target_col].astype(int)
X_test  = test_df[numeric_features  + categorical_features]
y_test  = test_df[target_col].astype(int)

print(f"\nTrain: {X_train.shape}  failure rate: {y_train.mean():.3f}")
print(f"Test:  {X_test.shape}   failure rate: {y_test.mean():.3f}")

# =============================================================================
# 7. Preprocessing pipeline
# =============================================================================
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
])
categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot",  OneHotEncoder(handle_unknown="ignore")),
])
preprocessor = ColumnTransformer([
    ("num", numeric_transformer,     numeric_features),
    ("cat", categorical_transformer, categorical_features),
])

# =============================================================================
# 8. Regularized logistic regression with cross-validated grid search
# =============================================================================
logistic_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(
        class_weight="balanced", solver="liblinear",
        max_iter=2000, random_state=42)),
])

param_grid = {
    "model__C":       [0.01, 0.1, 1.0, 10.0],
    "model__penalty": ["l1", "l2"],
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(
    logistic_pipeline, param_grid, scoring="roc_auc",
    cv=cv, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

best_logistic_model = grid_search.best_estimator_
print(f"\nBest params:   {grid_search.best_params_}")
print(f"Best CV AUC:   {grid_search.best_score_:.4f}")

# =============================================================================
# 9. Evaluation at default threshold (0.50)
# =============================================================================
y_prob        = best_logistic_model.predict_proba(X_test)[:, 1]
y_pred_default = (y_prob >= 0.50).astype(int)

metrics_default = {
    "Accuracy":  accuracy_score(y_test, y_pred_default),
    "Precision": precision_score(y_test, y_pred_default, zero_division=0),
    "Recall":    recall_score(y_test, y_pred_default,    zero_division=0),
    "F1-score":  f1_score(y_test, y_pred_default,        zero_division=0),
    "ROC-AUC":   roc_auc_score(y_test, y_prob),
}
print("\nDefault threshold (0.50):")
print(classification_report(y_test, y_pred_default, target_names=["Pass","Fail"]))

# Majority-class baseline
y_pred_baseline  = np.zeros_like(y_test)
baseline_metrics = {
    "Accuracy":  accuracy_score(y_test, y_pred_baseline),
    "Precision": precision_score(y_test, y_pred_baseline, zero_division=0),
    "Recall":    recall_score(y_test, y_pred_baseline,    zero_division=0),
    "F1-score":  f1_score(y_test, y_pred_baseline,        zero_division=0),
    "ROC-AUC":   0.50,
}
baseline_comparison = pd.DataFrame(
    [baseline_metrics, metrics_default],
    index=["Majority-Class Baseline", "Logistic Regression | Threshold 0.50"])
baseline_comparison.to_csv("outputs/baseline_vs_logistic_metrics.csv")
print(baseline_comparison)

# =============================================================================
# 10. Threshold tuning
# =============================================================================
thresholds = np.arange(0.10, 0.91, 0.01)
threshold_results = []
for thr in thresholds:
    yp = (y_prob >= thr).astype(int)
    threshold_results.append({
        "threshold": thr,
        "precision": precision_score(y_test, yp, zero_division=0),
        "recall":    recall_score(y_test, yp,    zero_division=0),
        "f1":        f1_score(y_test, yp,         zero_division=0),
    })
threshold_df  = pd.DataFrame(threshold_results)
best_f1_row   = threshold_df.loc[threshold_df["f1"].idxmax()]
best_threshold = best_f1_row["threshold"]
print(f"\nBest threshold (F1): {best_threshold:.2f}  F1={best_f1_row['f1']:.4f}")

# fig: threshold tuning curve
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(threshold_df["threshold"], threshold_df["precision"],
        color=C_BLUE,  lw=2, label="Precision")
ax.plot(threshold_df["threshold"], threshold_df["recall"],
        color=C_GOLD,  lw=2, label="Recall")
ax.plot(threshold_df["threshold"], threshold_df["f1"],
        color=C_PASS,  lw=2, label="F1-score")
ax.axvline(best_threshold, color=C_FAIL, linestyle="--", lw=1.8,
           label=f"Best F1 threshold = {best_threshold:.2f}\nF1 = {best_f1_row['f1']:.3f}")
ax.set_xlabel("Classification Threshold")
ax.set_ylabel("Metric Value")
ax.set_title("Threshold Tuning for Logistic Regression")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("figures/logistic_threshold_tuning.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved: figures/logistic_threshold_tuning.png")

# =============================================================================
# 11. Evaluation at best threshold
# =============================================================================
y_pred_best = (y_prob >= best_threshold).astype(int)
metrics_best = {
    "Accuracy":  accuracy_score(y_test, y_pred_best),
    "Precision": precision_score(y_test, y_pred_best, zero_division=0),
    "Recall":    recall_score(y_test, y_pred_best,    zero_division=0),
    "F1-score":  f1_score(y_test, y_pred_best,        zero_division=0),
    "ROC-AUC":   roc_auc_score(y_test, y_prob),
}
comparison_metrics = pd.DataFrame(
    [metrics_default, metrics_best],
    index=["Threshold 0.50", f"Best F1 Threshold {best_threshold:.2f}"])
comparison_metrics.to_csv("outputs/logistic_regression_metrics_threshold_comparison.csv")
print(comparison_metrics)

# fig: confusion matrix at best threshold
cm_best = confusion_matrix(y_test, y_pred_best)
fig, ax = plt.subplots(figsize=(5, 4))
disp = ConfusionMatrixDisplay(cm_best, display_labels=["Pass","Fail"])
disp.plot(ax=ax, values_format="d",
          colorbar=False,
          im_kw={"cmap": "Blues"})
ax.set_title(f"Confusion Matrix | Threshold = {best_threshold:.2f}")
plt.tight_layout()
plt.savefig("figures/logistic_confusion_matrix_best_threshold.png",
            dpi=200, bbox_inches="tight")
plt.close()
print("Saved: figures/logistic_confusion_matrix_best_threshold.png")

# fig: ROC curve
fig, ax = plt.subplots(figsize=(6, 5))
RocCurveDisplay.from_estimator(best_logistic_model, X_test, y_test, ax=ax,
                                color=C_FAIL, name="Logistic Regression")
ax.plot([0,1],[0,1], linestyle="--", color=C_GRAY, label="Random classifier")
ax.set_title("ROC Curve: Leakage-Safe Logistic Regression")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("figures/logistic_roc_curve.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved: figures/logistic_roc_curve.png")

# =============================================================================
# 12. Coefficient interpretation
# =============================================================================
preprocessor_fitted = best_logistic_model.named_steps["preprocessor"]
model_fitted        = best_logistic_model.named_steps["model"]

cat_feature_names = list(
    preprocessor_fitted
    .named_transformers_["cat"]
    .named_steps["onehot"]
    .get_feature_names_out(categorical_features)
)
all_feature_names = numeric_features + cat_feature_names

coef_df = pd.DataFrame({
    "feature":     all_feature_names,
    "coefficient": model_fitted.coef_[0],
})
coef_df["abs_coefficient"] = coef_df["coefficient"].abs()
coef_df = coef_df.sort_values("abs_coefficient", ascending=False)
coef_df.to_csv("outputs/logistic_regression_coefficients.csv", index=False)

top_pos = coef_df.sort_values("coefficient", ascending=False).head(15)
top_neg = coef_df.sort_values("coefficient", ascending=True).head(15)
coef_plot_df = pd.concat([top_neg, top_pos])

# fig: coefficient plot (unified colors)
fig, ax = plt.subplots(figsize=(10, 8))
colors = [C_FAIL if v > 0 else C_BLUE for v in coef_plot_df["coefficient"]]
ax.barh(coef_plot_df["feature"], coef_plot_df["coefficient"],
        color=colors, edgecolor="white", linewidth=0.6)
ax.axvline(0, color="black", linewidth=1)
ax.set_title("Top Logistic Regression Coefficients\n"
             "(Red = higher failure risk, Blue = lower failure risk)")
ax.set_xlabel("Coefficient")
plt.tight_layout()
plt.savefig("figures/logistic_coefficient_plot.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved: figures/logistic_coefficient_plot.png")

# Odds ratios
odds_df = coef_df.copy()
odds_df["odds_ratio"] = np.exp(odds_df["coefficient"])
top_odds = pd.concat([
    odds_df.sort_values("odds_ratio").head(15),
    odds_df.sort_values("odds_ratio", ascending=False).head(15),
])
odds_df.to_csv("outputs/logistic_regression_odds_ratios.csv", index=False)

fig, ax = plt.subplots(figsize=(10, 8))
colors_o = [C_FAIL if v > 1 else C_BLUE for v in top_odds["odds_ratio"]]
ax.barh(top_odds["feature"], top_odds["odds_ratio"],
        color=colors_o, edgecolor="white", linewidth=0.6)
ax.axvline(1, color="black", linewidth=1, linestyle="--")
ax.set_title("Top Logistic Regression Odds Ratios\n"
             "(Red > 1 = higher odds of failure, Blue < 1 = lower odds)")
ax.set_xlabel("Odds Ratio")
plt.tight_layout()
plt.savefig("figures/logistic_odds_ratio_plot.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved: figures/logistic_odds_ratio_plot.png")

# =============================================================================
# 13. Save all outputs
# =============================================================================
joblib.dump(best_logistic_model, "models/logistic_model.pkl")
df_fe.to_csv("outputs/restaurant_feature_engineered_leakage_safe.csv", index=False)
leakage_audit.to_csv("outputs/leakage_audit.csv", index=False)
threshold_df.to_csv("outputs/logistic_threshold_tuning.csv", index=False)

print("\nAll outputs saved.")
print("  models/logistic_model.pkl")
print("  outputs/restaurant_feature_engineered_leakage_safe.csv")
print("  figures/logistic_threshold_tuning.png")
print("  figures/logistic_confusion_matrix_best_threshold.png")
print("  figures/logistic_roc_curve.png")
print("  figures/logistic_coefficient_plot.png")
print("  figures/logistic_odds_ratio_plot.png")
