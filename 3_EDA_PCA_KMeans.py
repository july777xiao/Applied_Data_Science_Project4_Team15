# -*- coding: utf-8 -*-
# # Section 3: Exploratory Data Analysis, PCA, and Clustering
#
# This notebook uses Person A's cleaned data:
#
# - `data/processed/restaurant_clean.csv`
# - `data/processed/restaurant_yelp_subset.csv`
#
# Main outputs:
#
# - EDA/PCA/KMeans figures saved in `figures/person_b/`
# - Clustered handoff file saved as `data/processed/restaurant_clean_clustered.csv`
#
# Required structure:
#
# - 3.1 Univariate Analysis
# - 3.2 Bivariate Analysis
# - 3.3 Interaction Effects
# - 3.4 Unsupervised Learning: PCA
# - 3.5 Unsupervised Learning: KMeans

from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

try:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    try:
        from IPython.display import display
    except ImportError:
        display = print
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        f"Missing package: {exc.name}. Install with: "
        "pip install pandas numpy matplotlib seaborn scikit-learn"
    ) from exc

RANDOM_STATE = 42

ROOT = Path.cwd()
if not (ROOT / "data" / "processed" / "restaurant_clean.csv").exists():
    ROOT = ROOT.parent

DATA_DIR = ROOT / "data" / "processed"
FIG_DIR = ROOT / "figures" / "person_b"
FIG_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({
    "figure.figsize": (11, 6),
    "figure.dpi": 120,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
})

BOROUGH_ORDER = ["BRONX", "BROOKLYN", "MANHATTAN", "QUEENS", "STATEN ISLAND"]
MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
DOW_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def savefig(name):
    path = FIG_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    print(f"Saved figure: {path.relative_to(ROOT)}")


def clean_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax


def pct_axis(ax):
    ax.yaxis.set_major_formatter(lambda x, pos: f"{x:.0%}")
    return ax


def qbin(series, labels):
    # Ranking avoids qcut failures when many repeated values exist.
    return pd.qcut(series.rank(method="first"), q=len(labels), labels=labels)

clean_path = DATA_DIR / "restaurant_clean.csv"
yelp_path = DATA_DIR / "restaurant_yelp_subset.csv"

assert clean_path.exists(), f"Missing file: {clean_path}"
assert yelp_path.exists(), f"Missing file: {yelp_path}"

df = pd.read_csv(clean_path, parse_dates=["inspection_date"], low_memory=False)
yelp_df = pd.read_csv(yelp_path, parse_dates=["inspection_date"], low_memory=False)
base_columns = df.columns.tolist()

numeric_cols = [
    "score", "violation_count", "critical_count", "failed", "latitude", "longitude",
    "yelp_rating", "yelp_reviews", "yelp_price", "high_rating", "temp_mean",
    "temp_max", "temp_min", "precipitation_sum", "rain_sum", "snowfall_sum",
    "wind_speed_mean", "wind_gust_mean", "cloud_cover_mean", "food_complaints_total",
    "rodent_complaints", "food_safety_complaints", "median_household_income",
    "total_population", "white_population", "inspection_year", "inspection_month",
    "inspection_dow", "inspection_quarter", "is_weekend", "prev_score", "prev_failed",
    "inspection_count", "score_trend", "grade_available", "has_yelp",
    "is_first_inspection", "has_history", "white_pct", "has_location",
]
for frame in [df, yelp_df]:
    for col in numeric_cols:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")

df["failed_label"] = df["failed"].map({0: "Passed", 1: "Failed"})
df["grade_display"] = df["grade"].fillna("No grade")
df["month_name"] = pd.Categorical(
    df["inspection_month"].map(lambda x: MONTH_LABELS[int(x) - 1] if pd.notna(x) and 1 <= int(x) <= 12 else np.nan),
    categories=MONTH_LABELS,
    ordered=True,
)
df["dow_name"] = pd.Categorical(
    df["inspection_dow"].map(lambda x: DOW_LABELS[int(x)] if pd.notna(x) and 0 <= int(x) <= 6 else np.nan),
    categories=DOW_LABELS,
    ordered=True,
)
df["season"] = pd.Categorical(
    np.select(
        [
            df["inspection_month"].isin([12, 1, 2]),
            df["inspection_month"].isin([3, 4, 5]),
            df["inspection_month"].isin([6, 7, 8]),
            df["inspection_month"].isin([9, 10, 11]),
        ],
        ["Winter", "Spring", "Summer", "Fall"],
        default="Unknown",
    ),
    categories=["Winter", "Spring", "Summer", "Fall", "Unknown"],
    ordered=True,
)

print(f"Full cleaned input: {len(df):,} rows x {len(base_columns):,} columns")
print(f"Analysis frame after helper labels: {df.shape[0]:,} rows x {df.shape[1]:,} columns")
print(f"Yelp subset: {yelp_df.shape[0]:,} rows x {yelp_df.shape[1]:,} columns")
print(f"Date range: {df['inspection_date'].min().date()} to {df['inspection_date'].max().date()}")
print(f"Unique restaurants: {df['camis'].nunique():,}")
print(f"Failure rate: {df['failed'].mean():.1%}")
print(f"Yelp coverage: {df['has_yelp'].mean():.1%}")

display(df.head())

audit = pd.DataFrame({
    "metric": [
        "rows", "columns", "unique restaurants", "failure rate", "has Yelp rate",
        "has location rate", "mean score", "median score", "mean violations",
        "mean critical violations",
    ],
    "value": [
        f"{len(df):,}", f"{df.shape[1]:,}", f"{df['camis'].nunique():,}",
        f"{df['failed'].mean():.1%}", f"{df['has_yelp'].mean():.1%}",
        f"{df['has_location'].mean():.1%}", f"{df['score'].mean():.2f}",
        f"{df['score'].median():.2f}", f"{df['violation_count'].mean():.2f}",
        f"{df['critical_count'].mean():.2f}",
    ],
})
display(audit)

missing = (
    df.isna().mean()
    .sort_values(ascending=False)
    .rename("missing_rate")
    .reset_index()
    .rename(columns={"index": "feature"})
)
display(missing.head(20))

fig, ax = plt.subplots(figsize=(10, 5))
plot_missing = missing.head(15).sort_values("missing_rate")
sns.barplot(data=plot_missing, x="missing_rate", y="feature", ax=ax, color="#6c757d")
ax.set_title("Top Missingness Rates After Cleaning")
ax.set_xlabel("Missing rate")
ax.set_ylabel("Feature")
ax.xaxis.set_major_formatter(lambda x, pos: f"{x:.0%}")
clean_axis(ax)
savefig("00_missingness_audit.png")
plt.show()

# ## 3.1 Univariate Analysis
# This section summarizes the target, inspection scores, violations, boroughs, cuisine groups, Yelp variables, time patterns, weather, and complaint context.
# Important modeling note: `score`, `grade`, and `score_bucket` are useful for descriptive EDA, but they directly encode the inspection outcome and should be excluded from supervised failure prediction features.

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

failure_counts = df["failed_label"].value_counts().reindex(["Passed", "Failed"])
sns.barplot(x=failure_counts.index, y=failure_counts.values, ax=axes[0, 0], palette=["#2a9d8f", "#e76f51"])
axes[0, 0].set_title("Inspection Outcome Counts")
axes[0, 0].set_xlabel("Outcome")
axes[0, 0].set_ylabel("Number of inspections")
for i, v in enumerate(failure_counts.values):
    axes[0, 0].text(i, v + len(df) * 0.006, f"{v:,.0f}\n({v / len(df):.1%})", ha="center")
clean_axis(axes[0, 0])

score_df = df.dropna(subset=["score"])
sns.histplot(score_df["score"], bins=45, kde=True, ax=axes[0, 1], color="#457b9d")
axes[0, 1].axvline(28, color="#d62828", linestyle="--", linewidth=2, label="Failure threshold = 28")
axes[0, 1].axvline(score_df["score"].median(), color="#2a9d8f", linestyle=":", linewidth=2, label=f"Median = {score_df['score'].median():.1f}")
axes[0, 1].set_title("Score Distribution")
axes[0, 1].set_xlabel("Inspection score, lower is better")
axes[0, 1].legend()
clean_axis(axes[0, 1])

sns.boxplot(y=score_df["score"], ax=axes[0, 2], color="#a8dadc")
axes[0, 2].axhline(28, color="#d62828", linestyle="--", linewidth=2)
axes[0, 2].set_title("Score Boxplot")
axes[0, 2].set_ylabel("Inspection score")
clean_axis(axes[0, 2])

sns.histplot(df["violation_count"], bins=range(0, int(df["violation_count"].max()) + 2), ax=axes[1, 0], color="#264653")
axes[1, 0].set_xlim(-0.5, min(12.5, df["violation_count"].max() + 0.5))
axes[1, 0].set_title("Violation Count Distribution")
axes[1, 0].set_xlabel("Violation count")
clean_axis(axes[1, 0])

sns.histplot(df["critical_count"], bins=range(0, int(df["critical_count"].max()) + 2), ax=axes[1, 1], color="#e76f51")
axes[1, 1].set_xlim(-0.5, min(10.5, df["critical_count"].max() + 0.5))
axes[1, 1].set_title("Critical Violation Count Distribution")
axes[1, 1].set_xlabel("Critical violation count")
clean_axis(axes[1, 1])

grade_order = ["A", "B", "C", "No grade"]
grade_counts = df["grade_display"].value_counts().reindex(grade_order).fillna(0)
sns.barplot(x=grade_counts.index, y=grade_counts.values, ax=axes[1, 2], color="#7b2cbf")
axes[1, 2].set_title("Grade Distribution")
axes[1, 2].set_xlabel("Grade")
axes[1, 2].set_ylabel("Number of inspections")
clean_axis(axes[1, 2])

fig.suptitle("3.1 Core Univariate Distributions", y=1.02)
savefig("01_univariate_core_distributions.png")
plt.show()

print(
    "The decision boundary at score=28 clearly separates failing and non-failing inspections, "
    "motivating our binary classification target."
)

fig, axes = plt.subplots(3, 3, figsize=(20, 16))

borough_counts = df["boro"].value_counts().reindex(BOROUGH_ORDER)
sns.barplot(x=borough_counts.values, y=borough_counts.index, ax=axes[0, 0], color="#2a9d8f")
axes[0, 0].set_title("Inspection Count by Borough")
axes[0, 0].set_xlabel("Number of inspections")
axes[0, 0].set_ylabel("Borough")
clean_axis(axes[0, 0])

top_cuisine_counts = df["cuisine_grouped"].value_counts().head(15).sort_values()
sns.barplot(x=top_cuisine_counts.values, y=top_cuisine_counts.index, ax=axes[0, 1], color="#457b9d")
axes[0, 1].set_title("Top Cuisine Groups")
axes[0, 1].set_xlabel("Number of inspections")
axes[0, 1].set_ylabel("Cuisine group")
clean_axis(axes[0, 1])

sns.histplot(yelp_df["yelp_rating"].dropna(), bins=np.arange(1, 5.25, 0.25), kde=True, ax=axes[0, 2], color="#ffb703")
axes[0, 2].axvline(4.0, color="#d62828", linestyle="--", linewidth=2)
axes[0, 2].set_title("Yelp Rating Distribution")
axes[0, 2].set_xlabel("Yelp rating")
clean_axis(axes[0, 2])

rating_counts = yelp_df["high_rating"].map({0: "Below 4.0", 1: "At least 4.0"}).value_counts().reindex(["Below 4.0", "At least 4.0"])
sns.barplot(x=rating_counts.index, y=rating_counts.values, ax=axes[1, 0], palette=["#adb5bd", "#ffb703"])
axes[1, 0].set_title("High Yelp Rating Distribution")
axes[1, 0].set_xlabel("Yelp group")
axes[1, 0].set_ylabel("Number of inspections")
clean_axis(axes[1, 0])

sns.histplot(yelp_df["yelp_reviews"].dropna().clip(upper=yelp_df["yelp_reviews"].quantile(0.99)), bins=40, ax=axes[1, 1], color="#fb8500")
axes[1, 1].set_title("Yelp Review Counts, 99th Percentile Clipped")
axes[1, 1].set_xlabel("Review count")
clean_axis(axes[1, 1])

price_counts = yelp_df["yelp_price"].dropna().astype(int).value_counts().sort_index()
sns.barplot(x=price_counts.index.astype(str), y=price_counts.values, ax=axes[1, 2], color="#8ecae6")
axes[1, 2].set_title("Yelp Price Tier Distribution")
axes[1, 2].set_xlabel("Price tier")
clean_axis(axes[1, 2])

monthly_counts = df["month_name"].value_counts().reindex(MONTH_LABELS)
sns.lineplot(x=monthly_counts.index, y=monthly_counts.values, marker="o", ax=axes[2, 0], color="#1d3557")
axes[2, 0].set_title("Inspection Volume by Month")
axes[2, 0].set_xlabel("Month")
axes[2, 0].set_ylabel("Number of inspections")
axes[2, 0].tick_params(axis="x", rotation=45)
clean_axis(axes[2, 0])

dow_counts = df["dow_name"].value_counts().reindex(DOW_LABELS)
sns.barplot(x=dow_counts.index, y=dow_counts.values, ax=axes[2, 1], color="#2a9d8f")
axes[2, 1].set_title("Inspection Volume by Day of Week")
axes[2, 1].set_xlabel("Day of week")
clean_axis(axes[2, 1])

sns.histplot(df["food_complaints_total"].dropna(), bins=40, ax=axes[2, 2], color="#6d597a")
axes[2, 2].set_title("Food-Related 311 Complaints")
axes[2, 2].set_xlabel("Food complaints total")
clean_axis(axes[2, 2])

fig.suptitle("3.1 Geography, Cuisine, Yelp, Time, and Complaint Distributions", y=1.01)
savefig("02_univariate_context_distributions.png")
plt.show()

# ## 3.2 Bivariate Analysis
# This section links major feature families to the primary target (`failed`) and to inspection severity (`score`). The goal is to identify predictive signals, explain public-health patterns, and flag possible leakage for later modeling.

analysis = df.copy()
analysis["violation_bucket"] = pd.cut(
    analysis["violation_count"],
    bins=[-0.1, 0, 1, 2, 3, 4, np.inf],
    labels=["0", "1", "2", "3", "4", "5+"],
)
analysis["critical_bucket"] = pd.cut(
    analysis["critical_count"],
    bins=[-0.1, 0, 1, 2, 3, np.inf],
    labels=["0", "1", "2", "3", "4+"],
)

fig, axes = plt.subplots(2, 3, figsize=(20, 11))

sns.boxplot(data=df, x="failed_label", y="score", order=["Passed", "Failed"], ax=axes[0, 0], palette=["#2a9d8f", "#e76f51"])
axes[0, 0].axhline(28, color="#d62828", linestyle="--", linewidth=2)
axes[0, 0].set_title("Score by Outcome")
axes[0, 0].set_xlabel("Outcome")
axes[0, 0].set_ylabel("Inspection score")
clean_axis(axes[0, 0])

for ax, bucket, title in [
    (axes[0, 1], "violation_bucket", "Failure Rate by Violation Bucket"),
    (axes[0, 2], "critical_bucket", "Failure Rate by Critical Violation Bucket"),
]:
    rates = analysis.groupby(bucket, observed=False)["failed"].mean().reset_index()
    sns.barplot(data=rates, x=bucket, y="failed", ax=ax, color="#e76f51")
    ax.set_title(title)
    ax.set_xlabel(bucket.replace("_", " ").title())
    ax.set_ylabel("Failure rate")
    ax.yaxis.set_major_formatter(lambda x, pos: f"{x:.0%}")
    clean_axis(ax)

history_df = df[(df["has_history"] == 1) & (df["prev_score"] >= 0) & df["score"].notna()].copy()
history_plot = history_df.sample(n=min(12000, len(history_df)), random_state=RANDOM_STATE)
sns.regplot(data=history_plot, x="prev_score", y="score", scatter_kws={"alpha": 0.18, "s": 16}, line_kws={"color": "#d62828"}, ax=axes[1, 0])
axes[1, 0].axhline(28, color="#d62828", linestyle="--", linewidth=1)
axes[1, 0].axvline(28, color="#d62828", linestyle="--", linewidth=1)
axes[1, 0].set_title("Previous Score vs Current Score")
axes[1, 0].set_xlabel("Previous score")
axes[1, 0].set_ylabel("Current score")
clean_axis(axes[1, 0])

sns.boxplot(data=df, x="failed_label", y="score_trend", order=["Passed", "Failed"], ax=axes[1, 1], palette=["#2a9d8f", "#e76f51"])
axes[1, 1].axhline(0, color="black", linestyle=":")
axes[1, 1].set_title("Score Trend by Outcome")
axes[1, 1].set_xlabel("Outcome")
axes[1, 1].set_ylabel("Current score minus previous score")
clean_axis(axes[1, 1])

first_summary = (
    df.groupby("is_first_inspection")
    .agg(failure_rate=("failed", "mean"), avg_score=("score", "mean"), inspections=("failed", "size"))
    .rename(index={0: "Repeat", 1: "First"})
    .reset_index()
)
sns.barplot(data=first_summary, x="is_first_inspection", y="failure_rate", ax=axes[1, 2], color="#6c757d")
axes[1, 2].set_title("Failure Rate: First vs Repeat Inspections")
axes[1, 2].set_xlabel("Inspection history")
axes[1, 2].set_ylabel("Failure rate")
axes[1, 2].yaxis.set_major_formatter(lambda x, pos: f"{x:.0%}")
clean_axis(axes[1, 2])

savefig("03_bivariate_core_relationships.png")
plt.show()

print(f"Correlation between previous and current score: {history_df[['prev_score', 'score']].corr().iloc[0, 1]:.3f}")

yelp_plot = df[df["has_yelp"] == 1].dropna(subset=["yelp_rating", "score"]).copy()
yelp_plot = yelp_plot.sample(n=min(9000, len(yelp_plot)), random_state=RANDOM_STATE)
weather_plot = df.dropna(subset=["temp_mean", "score"]).sample(n=min(12000, df["temp_mean"].notna().sum()), random_state=RANDOM_STATE)

df["temp_bin"] = pd.cut(df["temp_mean"], bins=[-20, 0, 10, 20, 30, 45], labels=["<=0", "0-10", "10-20", "20-30", "30+"])
df["precip_bin"] = pd.cut(df["precipitation_sum"], bins=[-0.01, 0, 1, 5, 15, np.inf], labels=["0", "0-1", "1-5", "5-15", "15+"])

fig, axes = plt.subplots(2, 3, figsize=(20, 11))

sns.regplot(data=yelp_plot, x="yelp_rating", y="score", scatter_kws={"alpha": 0.22, "s": 18}, line_kws={"color": "#d62828"}, ax=axes[0, 0])
axes[0, 0].axhline(28, color="#d62828", linestyle="--", linewidth=1.5)
axes[0, 0].axvline(4.0, color="#ffb703", linestyle="--", linewidth=1.5)
axes[0, 0].set_title("Yelp Rating vs Inspection Score")
axes[0, 0].set_xlabel("Yelp rating")
axes[0, 0].set_ylabel("Inspection score")
clean_axis(axes[0, 0])

sns.boxplot(data=yelp_plot, x="failed_label", y="yelp_rating", order=["Passed", "Failed"], ax=axes[0, 1], palette=["#2a9d8f", "#e76f51"])
axes[0, 1].axhline(4.0, color="#ffb703", linestyle="--", linewidth=1.5)
axes[0, 1].set_title("Yelp Rating by Outcome")
axes[0, 1].set_xlabel("Outcome")
axes[0, 1].set_ylabel("Yelp rating")
clean_axis(axes[0, 1])

sns.scatterplot(data=weather_plot, x="temp_mean", y="score", hue="season", alpha=0.28, s=18, ax=axes[0, 2])
sns.regplot(data=weather_plot, x="temp_mean", y="score", scatter=False, ax=axes[0, 2], color="black")
axes[0, 2].axhline(28, color="#d62828", linestyle="--", linewidth=1.5)
axes[0, 2].set_title("Temperature vs Inspection Score")
axes[0, 2].set_xlabel("Mean temperature, C")
axes[0, 2].set_ylabel("Inspection score")
clean_axis(axes[0, 2])

temp_rates = df.groupby("temp_bin", observed=False)["failed"].mean().reset_index()
sns.barplot(data=temp_rates, x="temp_bin", y="failed", ax=axes[1, 0], color="#e76f51")
axes[1, 0].set_title("Failure Rate by Temperature Bin")
axes[1, 0].set_xlabel("Mean temperature bin, C")
axes[1, 0].set_ylabel("Failure rate")
axes[1, 0].yaxis.set_major_formatter(lambda x, pos: f"{x:.0%}")
clean_axis(axes[1, 0])

precip_rates = df.groupby("precip_bin", observed=False)["failed"].mean().reset_index()
sns.barplot(data=precip_rates, x="precip_bin", y="failed", ax=axes[1, 1], color="#457b9d")
axes[1, 1].set_title("Failure Rate by Precipitation Bin")
axes[1, 1].set_xlabel("Daily precipitation, mm")
axes[1, 1].set_ylabel("Failure rate")
axes[1, 1].yaxis.set_major_formatter(lambda x, pos: f"{x:.0%}")
clean_axis(axes[1, 1])

monthly_rate = df.groupby("month_name", observed=False)["failed"].mean().reset_index()
sns.lineplot(data=monthly_rate, x="month_name", y="failed", marker="o", ax=axes[1, 2], color="#e76f51")
axes[1, 2].set_title("Failure Rate by Month")
axes[1, 2].set_xlabel("Month")
axes[1, 2].set_ylabel("Failure rate")
axes[1, 2].tick_params(axis="x", rotation=45)
axes[1, 2].yaxis.set_major_formatter(lambda x, pos: f"{x:.0%}")
clean_axis(axes[1, 2])

savefig("04_bivariate_yelp_weather_time.png")
plt.show()

print(f"Correlation between Yelp rating and score: {yelp_plot[['yelp_rating', 'score']].corr().iloc[0, 1]:.3f}")

borough_summary = (
    df.groupby("boro")
    .agg(
        inspections=("failed", "size"),
        failure_rate=("failed", "mean"),
        avg_score=("score", "mean"),
        avg_violations=("violation_count", "mean"),
        avg_critical=("critical_count", "mean"),
        avg_food_complaints=("food_complaints_total", "mean"),
        median_income=("median_household_income", "median"),
    )
    .reindex(BOROUGH_ORDER)
    .reset_index()
)
display(borough_summary)

cuisine_summary = (
    df.groupby("cuisine_grouped")
    .agg(inspections=("failed", "size"), failure_rate=("failed", "mean"), avg_score=("score", "mean"), avg_critical=("critical_count", "mean"))
    .query("inspections >= 100")
    .sort_values("failure_rate", ascending=False)
)
display(cuisine_summary.head(20))

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

sns.barplot(data=borough_summary, x="boro", y="failure_rate", ax=axes[0], color="#e76f51")
axes[0].set_title("Failure Rate by Borough")
axes[0].set_xlabel("Borough")
axes[0].set_ylabel("Failure rate")
axes[0].tick_params(axis="x", rotation=35)
axes[0].yaxis.set_major_formatter(lambda x, pos: f"{x:.0%}")
clean_axis(axes[0])

sns.scatterplot(data=borough_summary, x="median_income", y="failure_rate", size="inspections", hue="boro", sizes=(120, 800), ax=axes[1])
for _, row in borough_summary.iterrows():
    axes[1].text(row["median_income"], row["failure_rate"] + 0.002, row["boro"].title(), ha="center", fontsize=9)
axes[1].set_title("Income vs Failure Rate by Borough")
axes[1].set_xlabel("Median household income")
axes[1].set_ylabel("Failure rate")
axes[1].yaxis.set_major_formatter(lambda x, pos: f"{x:.0%}")
axes[1].legend([], [], frameon=False)
clean_axis(axes[1])

top_risk = cuisine_summary.head(15).sort_values("failure_rate")
sns.barplot(data=top_risk.reset_index(), x="failure_rate", y="cuisine_grouped", ax=axes[2], color="#e76f51")
axes[2].set_title("Highest Failure-Rate Cuisine Groups")
axes[2].set_xlabel("Failure rate")
axes[2].set_ylabel("Cuisine group")
axes[2].xaxis.set_major_formatter(lambda x, pos: f"{x:.0%}")
clean_axis(axes[2])

savefig("05_bivariate_borough_cuisine.png")
plt.show()

# ## 3.3 Interaction Effects
# This section tests whether relationships change across context. Interaction analysis is especially useful here because regulatory outcomes may vary by borough, cuisine type, complaint environment, and socioeconomic context.
# Report sentence to reuse: *Interaction effects suggest that the impact of neighborhood complaints on failure rates varies significantly across boroughs, indicating that socioeconomic context moderates the relationship between civic complaints and regulatory outcomes.*

interaction_df = df.dropna(subset=["median_household_income", "food_complaints_total", "failed"]).copy()
interaction_df["income_quartile"] = qbin(interaction_df["median_household_income"], ["Q1 low income", "Q2", "Q3", "Q4 high income"])
interaction_df["complaint_quartile"] = qbin(interaction_df["food_complaints_total"], ["Q1 low complaints", "Q2", "Q3", "Q4 high complaints"])

income_complaint_heat = interaction_df.pivot_table(
    index="income_quartile",
    columns="complaint_quartile",
    values="failed",
    aggfunc="mean",
    observed=False,
)

top_cuisines = df["cuisine_grouped"].value_counts().head(12).index
cuisine_boro_heat = (
    df[df["cuisine_grouped"].isin(top_cuisines)]
    .pivot_table(index="boro", columns="cuisine_grouped", values="failed", aggfunc="mean", observed=False)
    .reindex(BOROUGH_ORDER)
)

month_boro_heat = (
    df.pivot_table(index="boro", columns="month_name", values="failed", aggfunc="mean", observed=False)
    .reindex(BOROUGH_ORDER)
)

fig, axes = plt.subplots(3, 1, figsize=(18, 18))

sns.heatmap(income_complaint_heat, annot=True, fmt=".1%", cmap="YlOrRd", linewidths=0.5, ax=axes[0])
axes[0].set_title("Failure Rate: Income x 311 Food Complaints")
axes[0].set_xlabel("Complaint quartile")
axes[0].set_ylabel("Income quartile")

sns.heatmap(cuisine_boro_heat, annot=True, fmt=".1%", cmap="rocket_r", linewidths=0.4, ax=axes[1])
axes[1].set_title("Failure Rate: Cuisine Group x Borough")
axes[1].set_xlabel("Cuisine group")
axes[1].set_ylabel("Borough")
axes[1].tick_params(axis="x", rotation=45, labelsize=9)

sns.heatmap(month_boro_heat, annot=True, fmt=".1%", cmap="YlGnBu", linewidths=0.4, ax=axes[2])
axes[2].set_title("Failure Rate: Borough x Month")
axes[2].set_xlabel("Month")
axes[2].set_ylabel("Borough")

savefig("06_interaction_heatmaps.png")
plt.show()

critical_cuisine = df[df["cuisine_grouped"].isin(top_cuisines)].copy()
critical_cuisine["critical_bucket_small"] = pd.cut(
    critical_cuisine["critical_count"],
    bins=[-0.1, 0, 1, 2, np.inf],
    labels=["0", "1", "2", "3+"],
)
critical_cuisine_heat = critical_cuisine.pivot_table(
    index="critical_bucket_small",
    columns="cuisine_grouped",
    values="failed",
    aggfunc="mean",
    observed=False,
)

predictor_corr_cols = [
    "violation_count", "critical_count", "temp_mean", "precipitation_sum", "rain_sum",
    "snowfall_sum", "wind_speed_mean", "wind_gust_mean", "cloud_cover_mean",
    "food_complaints_total", "rodent_complaints", "food_safety_complaints",
    "median_household_income", "total_population", "white_population", "white_pct",
    "inspection_month", "inspection_dow", "is_weekend", "prev_score", "prev_failed",
    "inspection_count", "score_trend", "grade_available", "has_yelp",
    "is_first_inspection", "has_history", "has_location",
]
predictor_corr_cols = [col for col in predictor_corr_cols if col in df.columns]
corr = df[predictor_corr_cols].corr(numeric_only=True)
mask = np.triu(np.ones_like(corr, dtype=bool))

fig, axes = plt.subplots(2, 1, figsize=(18, 20))

sns.heatmap(critical_cuisine_heat, annot=True, fmt=".1%", cmap="flare", linewidths=0.4, ax=axes[0])
axes[0].set_title("Failure Rate: Critical Violations x Cuisine Group")
axes[0].set_xlabel("Cuisine group")
axes[0].set_ylabel("Critical violation bucket")
axes[0].tick_params(axis="x", rotation=45, labelsize=9)

sns.heatmap(corr, mask=mask, cmap="vlag", center=0, linewidths=0.35, ax=axes[1], cbar_kws={"shrink": 0.75})
axes[1].set_title("Correlation Heatmap of Numeric Predictor Candidates")

savefig("07_interaction_and_correlation_heatmaps.png")
plt.show()

target_corr = (
    df[predictor_corr_cols + ["failed"]]
    .corr(numeric_only=True)["failed"]
    .drop("failed")
    .sort_values(key=lambda s: s.abs(), ascending=False)
    .head(20)
    .reset_index()
    .rename(columns={"index": "feature", "failed": "corr_with_failed"})
)
display(target_corr)

fig, ax = plt.subplots(figsize=(10, 7))
sns.barplot(data=target_corr.sort_values("corr_with_failed"), x="corr_with_failed", y="feature", ax=ax, palette="coolwarm")
ax.axvline(0, color="black", linewidth=1)
ax.set_title("Top Numeric Correlations With Failure Target")
ax.set_xlabel("Pearson correlation with failed")
ax.set_ylabel("Feature")
clean_axis(ax)
savefig("08_target_correlation_bars.png")
plt.show()

print("Multicollinearity note: violation_count and critical_count capture related inspection-severity information.")

# ## 3.4 Unsupervised Learning: PCA
# PCA is used as an exploratory dimensionality-reduction method. The selected variables include inspection severity, weather, 311 complaint context, demographics, and inspection history.
# Preprocessing:
# - Median imputation for missing numeric values
# - StandardScaler before PCA
# - Components inspected until cumulative explained variance reaches at least 80 percent

pca_features = [
    "score", "violation_count", "critical_count", "temp_mean", "precipitation_sum",
    "food_complaints_total", "rodent_complaints", "median_household_income",
    "white_pct", "prev_score", "inspection_count", "score_trend",
]
pca_features = [col for col in pca_features if col in df.columns]

X_pca_raw = df[pca_features].copy()
imputer = SimpleImputer(strategy="median")
scaler = StandardScaler()
X_pca_imputed = imputer.fit_transform(X_pca_raw)
X_pca_scaled = scaler.fit_transform(X_pca_imputed)

n_components = min(len(pca_features), 12)
pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
pca_scores = pca.fit_transform(X_pca_scaled)

pca_cols = [f"PC{i}" for i in range(1, n_components + 1)]
pca_df = pd.DataFrame(pca_scores, columns=pca_cols, index=df.index)
pca_df["failed"] = df["failed"].values
pca_df["failed_label"] = df["failed_label"].values
pca_df["boro"] = df["boro"].values

explained = pd.DataFrame({
    "component": pca_cols,
    "explained_variance_ratio": pca.explained_variance_ratio_,
    "cumulative_variance": np.cumsum(pca.explained_variance_ratio_),
})
n80 = int(np.argmax(explained["cumulative_variance"].values >= 0.80) + 1) if (explained["cumulative_variance"] >= 0.80).any() else n_components

print(f"PCA feature count: {len(pca_features)}")
print(f"Components needed to explain at least 80% variance: {n80}")
display(explained)

fig, axes = plt.subplots(2, 2, figsize=(18, 13))

sns.barplot(data=explained, x="component", y="explained_variance_ratio", ax=axes[0, 0], color="#457b9d")
axes[0, 0].set_title("PCA Scree Plot")
axes[0, 0].set_xlabel("Principal component")
axes[0, 0].set_ylabel("Explained variance ratio")
axes[0, 0].yaxis.set_major_formatter(lambda x, pos: f"{x:.0%}")
clean_axis(axes[0, 0])

sns.lineplot(data=explained, x="component", y="cumulative_variance", marker="o", ax=axes[0, 1], color="#e76f51")
axes[0, 1].axhline(0.80, color="black", linestyle="--", linewidth=1.2, label="80% threshold")
axes[0, 1].set_title("Cumulative Explained Variance")
axes[0, 1].set_xlabel("Principal component")
axes[0, 1].set_ylabel("Cumulative variance")
axes[0, 1].yaxis.set_major_formatter(lambda x, pos: f"{x:.0%}")
axes[0, 1].legend()
clean_axis(axes[0, 1])

pca_plot = pca_df.sample(n=min(14000, len(pca_df)), random_state=RANDOM_STATE)
sns.scatterplot(data=pca_plot, x="PC1", y="PC2", hue="failed_label", alpha=0.35, s=18, palette={"Passed": "#2a9d8f", "Failed": "#e76f51"}, ax=axes[1, 0])
axes[1, 0].set_title("PCA Projection Colored by Outcome")
axes[1, 0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
axes[1, 0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
clean_axis(axes[1, 0])

sns.kdeplot(data=pca_plot, x="PC1", hue="failed_label", common_norm=False, fill=True, alpha=0.25, palette={"Passed": "#2a9d8f", "Failed": "#e76f51"}, ax=axes[1, 1])
axes[1, 1].set_title("PC1 Density by Outcome")
axes[1, 1].set_xlabel("PC1")
axes[1, 1].set_ylabel("Density")
clean_axis(axes[1, 1])

savefig("09_pca_scree_projection.png")
plt.show()

loadings = pd.DataFrame(pca.components_.T, index=pca_features, columns=pca_cols)

fig, axes = plt.subplots(1, 3, figsize=(22, 6))

for ax, pc in zip(axes[:2], ["PC1", "PC2"]):
    vals = loadings[pc].sort_values()
    colors = np.where(vals.values >= 0, "#e76f51", "#457b9d")
    ax.barh(vals.index, vals.values, color=colors)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_title(f"{pc} Feature Loadings")
    ax.set_xlabel("Loading")
    ax.set_ylabel("Feature")
    clean_axis(ax)

pca_boro = pca_df.copy()
sns.boxplot(data=pca_boro, x="boro", y="PC1", order=BOROUGH_ORDER, ax=axes[2], color="#a8dadc")
axes[2].set_title("PC1 Distribution by Borough")
axes[2].set_xlabel("Borough")
axes[2].set_ylabel("PC1")
axes[2].tick_params(axis="x", rotation=35)
clean_axis(axes[2])

savefig("10_pca_loadings_borough.png")
plt.show()

display(loadings[["PC1", "PC2"]].sort_values("PC1"))

# ## 3.5 Unsupervised Learning: KMeans
# KMeans uses a separate standardized handoff feature matrix. Unlike the descriptive PCA, the clustering feature set excludes `score`, `grade`, `score_bucket`, and `score_trend` so the saved cluster label is less likely to behave like a direct proxy for the target. The notebook compares K = 2 through K = 8 with inertia and sampled silhouette score, then fits a final multi-segment KMeans model and saves cluster labels for the next project member.
# KMeans handoff clustering uses a separate feature set from the descriptive PCA.
# Exclude score, grade, score_bucket, and score_trend so the saved cluster label is not just a proxy for the target definition.
cluster_features = [
    "violation_count", "critical_count", "temp_mean", "precipitation_sum",
    "food_complaints_total", "rodent_complaints", "median_household_income",
    "white_pct", "prev_score", "prev_failed", "inspection_count",
    "has_history", "is_first_inspection", "has_yelp", "has_location",
]
cluster_features = [col for col in cluster_features if col in df.columns]

cluster_imputer = SimpleImputer(strategy="median")
cluster_scaler = StandardScaler()
X_cluster_raw = df[cluster_features].copy()
X_cluster = cluster_scaler.fit_transform(cluster_imputer.fit_transform(X_cluster_raw))

print("KMeans clustering features:")
print(cluster_features)

k_values = range(2, 9)
cluster_eval = []

rng = np.random.default_rng(RANDOM_STATE)
silhouette_n = min(10000, X_cluster.shape[0])
silhouette_idx = rng.choice(X_cluster.shape[0], size=silhouette_n, replace=False)

for k in k_values:
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=25)
    labels = km.fit_predict(X_cluster)
    sil = silhouette_score(X_cluster[silhouette_idx], labels[silhouette_idx])
    cluster_eval.append({"k": k, "inertia": km.inertia_, "silhouette": sil})

cluster_eval_df = pd.DataFrame(cluster_eval)
display(cluster_eval_df)
silhouette_best_k = int(cluster_eval_df.loc[cluster_eval_df["silhouette"].idxmax(), "k"])
multi_segment_candidates = cluster_eval_df[cluster_eval_df["k"].isin([3, 4])]
recommended_k = int(multi_segment_candidates.loc[multi_segment_candidates["silhouette"].idxmax(), "k"])
print(f"Overall silhouette-best K: {silhouette_best_k}")
print(f"Recommended multi-segment K for interpretation and handoff: {recommended_k}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.lineplot(data=cluster_eval_df, x="k", y="inertia", marker="o", ax=axes[0], color="#457b9d")
axes[0].set_title("KMeans Elbow Curve")
axes[0].set_xlabel("Number of clusters K")
axes[0].set_ylabel("Inertia")
clean_axis(axes[0])

sns.lineplot(data=cluster_eval_df, x="k", y="silhouette", marker="o", ax=axes[1], color="#e76f51")
axes[1].set_title("KMeans Silhouette Score")
axes[1].set_xlabel("Number of clusters K")
axes[1].set_ylabel("Silhouette score, sampled")
clean_axis(axes[1])

savefig("11_kmeans_elbow_silhouette.png")
plt.show()

# Final K choice:
# K=2 can maximize silhouette by making a coarse binary split, but a 3- or 4-cluster solution
# gives Person C a more useful behavioral segment feature. Use the better silhouette among K=3 and K=4.
FINAL_K = recommended_k
print(f"Final K selected for handoff clustering: {FINAL_K}")

kmeans = KMeans(n_clusters=FINAL_K, random_state=RANDOM_STATE, n_init=50)
df["cluster"] = kmeans.fit_predict(X_cluster)
pca_df["cluster"] = df["cluster"].values

profile = (
    df.groupby("cluster")
    .agg(
        inspections=("failed", "size"),
        restaurants=("camis", "nunique"),
        failure_rate=("failed", "mean"),
        avg_score=("score", "mean"),
        median_score=("score", "median"),
        avg_violations=("violation_count", "mean"),
        avg_critical=("critical_count", "mean"),
        avg_prev_score=("prev_score", "mean"),
        history_rate=("has_history", "mean"),
        first_inspection_rate=("is_first_inspection", "mean"),
        avg_inspection_count=("inspection_count", "mean"),
        avg_score_trend=("score_trend", "mean"),
        avg_food_complaints=("food_complaints_total", "mean"),
        avg_rodent_complaints=("rodent_complaints", "mean"),
        median_income=("median_household_income", "median"),
        yelp_coverage=("has_yelp", "mean"),
        avg_yelp_rating=("yelp_rating", "mean"),
    )
    .sort_index()
)

risk_order = profile["failure_rate"].sort_values().index.tolist()
cluster_names = {}

for c, row in profile.iterrows():
    if row["failure_rate"] == profile["failure_rate"].min():
        if row["yelp_coverage"] >= 0.8:
            cluster_names[c] = "Yelp-Matched Stable Compliant"
        else:
            cluster_names[c] = "Stable Compliant"
    elif row["first_inspection_rate"] >= 0.9 or row["history_rate"] <= 0.1:
        cluster_names[c] = "No-History Higher Risk"
    elif row["history_rate"] >= 0.8 and row["avg_prev_score"] >= profile["avg_prev_score"].median():
        cluster_names[c] = "Repeat Operators With Prior Risk"
    elif row["failure_rate"] == profile["failure_rate"].max():
        cluster_names[c] = "High-Risk Mixed Profile"
    else:
        cluster_names[c] = "Complaint-Context Mixed Risk"

df["cluster_name"] = df["cluster"].map(cluster_names)
pca_df["cluster_name"] = df["cluster_name"].values

display(profile)
display(pd.Series(cluster_names, name="cluster_name").rename_axis("cluster").reset_index())

fig, axes = plt.subplots(2, 2, figsize=(18, 12))

cluster_counts = df["cluster_name"].value_counts()
sns.barplot(x=cluster_counts.values, y=cluster_counts.index, ax=axes[0, 0], color="#457b9d")
axes[0, 0].set_title("Cluster Sizes")
axes[0, 0].set_xlabel("Number of inspections")
axes[0, 0].set_ylabel("Cluster")
clean_axis(axes[0, 0])

cluster_fail = df.groupby("cluster_name")["failed"].mean().sort_values().reset_index()
sns.barplot(data=cluster_fail, x="failed", y="cluster_name", ax=axes[0, 1], color="#e76f51")
axes[0, 1].set_title("Failure Rate by Cluster")
axes[0, 1].set_xlabel("Failure rate")
axes[0, 1].set_ylabel("Cluster")
axes[0, 1].xaxis.set_major_formatter(lambda x, pos: f"{x:.0%}")
clean_axis(axes[0, 1])

profile_numeric = profile[[
    "failure_rate", "avg_score", "avg_violations", "avg_critical", "history_rate",
    "first_inspection_rate", "avg_food_complaints", "avg_rodent_complaints",
    "median_income", "yelp_coverage",
]].copy()
profile_numeric.index = [cluster_names[i] for i in profile_numeric.index]
profile_scaled = pd.DataFrame(
    StandardScaler().fit_transform(profile_numeric),
    index=profile_numeric.index,
    columns=profile_numeric.columns,
)
sns.heatmap(profile_scaled, annot=True, fmt=".2f", cmap="vlag", center=0, linewidths=0.4, ax=axes[1, 0])
axes[1, 0].set_title("Standardized Cluster Profile Heatmap")
axes[1, 0].set_xlabel("Profile feature")
axes[1, 0].set_ylabel("Cluster")
axes[1, 0].tick_params(axis="x", rotation=35)

pca_cluster_plot = pca_df.sample(n=min(14000, len(pca_df)), random_state=RANDOM_STATE)
sns.scatterplot(data=pca_cluster_plot, x="PC1", y="PC2", hue="cluster_name", alpha=0.35, s=18, ax=axes[1, 1])
axes[1, 1].set_title("PCA Projection Colored by KMeans Cluster")
axes[1, 1].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
axes[1, 1].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
axes[1, 1].legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
clean_axis(axes[1, 1])

savefig("12_cluster_profiles_pca.png")
plt.show()

cluster_boro = pd.crosstab(df["cluster_name"], df["boro"], normalize="index").reindex(columns=BOROUGH_ORDER)
cluster_fail_tab = pd.crosstab(df["cluster_name"], df["failed_label"], normalize="index").reindex(columns=["Passed", "Failed"])

fig, axes = plt.subplots(1, 2, figsize=(18, 6))

cluster_boro.plot(kind="barh", stacked=True, ax=axes[0], colormap="tab20")
axes[0].set_title("Borough Mix Within Each Cluster")
axes[0].set_xlabel("Share of cluster")
axes[0].set_ylabel("Cluster")
axes[0].xaxis.set_major_formatter(lambda x, pos: f"{x:.0%}")
axes[0].legend(title="Borough", bbox_to_anchor=(1.02, 1), loc="upper left")
clean_axis(axes[0])

cluster_fail_tab.plot(kind="barh", stacked=True, ax=axes[1], color=["#2a9d8f", "#e76f51"])
axes[1].set_title("Outcome Mix Within Each Cluster")
axes[1].set_xlabel("Share of cluster")
axes[1].set_ylabel("Cluster")
axes[1].xaxis.set_major_formatter(lambda x, pos: f"{x:.0%}")
axes[1].legend(title="Outcome", bbox_to_anchor=(1.02, 1), loc="upper left")
clean_axis(axes[1])

savefig("13_cluster_composition.png")
plt.show()

top_cluster_cuisines = df["cuisine_grouped"].value_counts().head(12).index
cluster_cuisine = pd.crosstab(
    df.loc[df["cuisine_grouped"].isin(top_cluster_cuisines), "cluster_name"],
    df.loc[df["cuisine_grouped"].isin(top_cluster_cuisines), "cuisine_grouped"],
    normalize="index",
)

fig, ax = plt.subplots(figsize=(15, 6))
sns.heatmap(cluster_cuisine, annot=True, fmt=".1%", cmap="Blues", linewidths=0.4, ax=ax)
ax.set_title("Cuisine Composition Within Each Cluster")
ax.set_xlabel("Cuisine group")
ax.set_ylabel("Cluster")
ax.tick_params(axis="x", rotation=45, labelsize=9)
savefig("14_cluster_cuisine_composition.png")
plt.show()

clustered_path = DATA_DIR / "restaurant_clean_clustered.csv"
handoff_df = df[base_columns + ["cluster", "cluster_name"]].copy()
handoff_df.to_csv(clustered_path, index=False)

print(f"Saved clustered dataset: {clustered_path.relative_to(ROOT)}")
print(f"Rows saved: {len(handoff_df):,}")
print(f"Columns saved: {handoff_df.shape[1]:,}")
print("New handoff columns: cluster, cluster_name")

handoff_cols = [
    "camis", "inspection_date", "dba", "boro", "cuisine_grouped", "failed",
    "score", "violation_count", "critical_count", "cluster", "cluster_name",
]
display(handoff_df[handoff_cols].head(10))

assert clustered_path.exists(), "Clustered CSV was not saved."
assert {"cluster", "cluster_name"}.issubset(df.columns), "Cluster columns are missing."

# ### Person B Handoff Summary
# This notebook completes EDA, PCA, and KMeans clustering for Person B.

# Handoff for Person C:
# - Use `data/processed/restaurant_clean_clustered.csv`.
# - `cluster` is the numeric KMeans segment.
# - `cluster_name` is an interpretation label generated from each cluster profile, so it should be treated as descriptive reporting text rather than a stable model input.

# Leakage reminder:
# - `score`, `grade`, `score_bucket`, and `score_trend` should not be used as supervised predictors for `failed`.
# - The saved `cluster` feature is built without those direct outcome/proxy score fields, so it is safer for Person C to evaluate as a categorical segment feature.
