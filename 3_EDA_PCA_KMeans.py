
# Section 3: EDA, PCA, KMeans  —  refactored figure layout

# Figure map (all saved to figures/ folder):
#   fig01a  outcome counts + score distribution
#   fig01b  violation dist + critical violation dist
#   fig01c  grade dist + monthly inspection volume
#   fig02a  borough counts + top cuisine counts
#   fig02b  yelp rating dist + yelp price tier
#   fig03a  score by outcome + violation failure rate
#   fig03b  critical failure rate + prev_score scatter
#   fig03c  score trend by outcome + first vs repeat failure rate
#   fig04a  yelp rating vs score + yelp rating by outcome
#   fig04b  monthly failure rate + temperature failure rate
#   fig05a  failure rate by borough + income scatter
#   fig05b  top cuisine failure rate  (single wide panel)
#   fig06a  income x complaint interaction heatmap
#   fig06b  cuisine x borough interaction heatmap
#   fig06c  borough x month interaction heatmap
#   fig07   numeric predictor correlation heatmap
#   fig08   target correlation bar chart
#   fig09a  PCA scree + cumulative variance
#   fig09b  PCA 2D projection + PC1 density by outcome
#   fig10a  PC1 loadings + PC2 loadings
#   fig10b  PC1 distribution by borough
#   fig11   KMeans elbow + silhouette score
#   fig12a  cluster size + cluster failure rate
#   fig12b  cluster profile heatmap + PCA colored by cluster
#   fig13   cluster composition (borough + outcome)

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
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

RANDOM_STATE = 42

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path.cwd()
DATA_DIR = None
for candidate in [ROOT, ROOT.parent]:
    if (candidate / "raw" / "restaurant_clean.csv").exists():
        DATA_DIR = candidate / "raw"; break
    if (candidate / "data" / "processed" / "restaurant_clean.csv").exists():
        DATA_DIR = candidate / "data" / "processed"; break
if DATA_DIR is None:
    DATA_DIR = ROOT / "raw"   # fallback

FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Palette (used everywhere) ─────────────────────────────────────────────────
C_PASS   = "#2a9d8f"
C_FAIL   = "#e63946"
C_BLUE   = "#457b9d"
C_GOLD   = "#f4a261"
C_PURPLE = "#7b5ea7"
C_GRAY   = "#6c757d"

BOROUGH_COLORS = ["#264653", "#2a9d8f", "#457b9d", "#e9c46a", "#f4a261"]
PASS_FAIL_PAL  = {"Passed": C_PASS, "Failed": C_FAIL}

BOROUGH_ORDER = ["BRONX", "BROOKLYN", "MANHATTAN", "QUEENS", "STATEN ISLAND"]
MONTH_LABELS  = ["Jan","Feb","Mar","Apr","May","Jun",
                  "Jul","Aug","Sep","Oct","Nov","Dec"]
DOW_LABELS    = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]

sns.set_theme(style="whitegrid", context="notebook", font_scale=1.05)
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

def savefig(name):
    plt.tight_layout()
    plt.savefig(FIG_DIR / name, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  saved  {name}")

def fmt_pct(ax, axis="y"):
    fmt = mticker.FuncFormatter(lambda x, _: f"{x:.0%}")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)

def gradient_colors(cmap_name, n, skip=2):
    cm = plt.cm.get_cmap(cmap_name, n + skip)
    return [cm(i + skip) for i in range(n)]

# ── Load data ─────────────────────────────────────────────────────────────────
df      = pd.read_csv(DATA_DIR / "restaurant_clean.csv",
                       parse_dates=["inspection_date"], low_memory=False)
yelp_df = pd.read_csv(DATA_DIR / "restaurant_yelp_subset.csv",
                       parse_dates=["inspection_date"], low_memory=False)
base_columns = df.columns.tolist()

numeric_cols = [
    "score","violation_count","critical_count","failed","latitude","longitude",
    "yelp_rating","yelp_reviews","yelp_price","high_rating","temp_mean","temp_max",
    "temp_min","precipitation_sum","rain_sum","snowfall_sum","wind_speed_mean",
    "wind_gust_mean","cloud_cover_mean","food_complaints_total","rodent_complaints",
    "food_safety_complaints","median_household_income","total_population",
    "white_population","inspection_year","inspection_month","inspection_dow",
    "inspection_quarter","is_weekend","prev_score","prev_failed","inspection_count",
    "score_trend","grade_available","has_yelp","is_first_inspection","has_history",
    "white_pct","has_location",
]
for frame in [df, yelp_df]:
    for col in numeric_cols:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")

df["failed_label"] = df["failed"].map({0: "Passed", 1: "Failed"})
df["grade_display"] = df["grade"].fillna("No grade")
df["month_name"] = pd.Categorical(
    df["inspection_month"].map(
        lambda x: MONTH_LABELS[int(x)-1] if pd.notna(x) and 1<=int(x)<=12 else np.nan
    ), categories=MONTH_LABELS, ordered=True)
df["dow_name"] = pd.Categorical(
    df["inspection_dow"].map(
        lambda x: DOW_LABELS[int(x)] if pd.notna(x) and 0<=int(x)<=6 else np.nan
    ), categories=DOW_LABELS, ordered=True)
df["season"] = pd.Categorical(
    np.select(
        [df["inspection_month"].isin([12,1,2]), df["inspection_month"].isin([3,4,5]),
         df["inspection_month"].isin([6,7,8]),  df["inspection_month"].isin([9,10,11])],
        ["Winter","Spring","Summer","Fall"], default="Unknown"),
    categories=["Winter","Spring","Summer","Fall","Unknown"], ordered=True)

print(f"Rows: {len(df):,}  Restaurants: {df['camis'].nunique():,}")
print(f"Failure rate: {df['failed'].mean():.1%}  Yelp coverage: {df['has_yelp'].mean():.1%}")

# =============================================================================
# 3.1  UNIVARIATE
# =============================================================================
print("\n── 3.1 Univariate ──")
score_df = df.dropna(subset=["score"])

# fig01a ── outcome counts + score distribution
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
fc = df["failed_label"].value_counts().reindex(["Passed","Failed"])
bars = ax1.bar(fc.index, fc.values, color=[C_PASS, C_FAIL],
               width=0.5, edgecolor="white", linewidth=1.2)
for bar, v in zip(bars, fc.values):
    ax1.text(bar.get_x()+bar.get_width()/2, v+len(df)*0.006,
             f"{v:,.0f}\n({v/len(df):.1%})", ha="center", fontsize=9.5)
ax1.set_title("Inspection Outcome Counts")
ax1.set_xlabel("Outcome"); ax1.set_ylabel("Number of inspections")
ax1.set_ylim(0, fc.max()*1.15)

ax2.hist(score_df["score"], bins=45, color=C_BLUE, alpha=0.85,
         edgecolor="white", linewidth=0.4)
ax2.axvline(28, color=C_FAIL, linestyle="--", lw=2, label="Failure threshold = 28")
ax2.axvline(score_df["score"].median(), color=C_PASS, linestyle=":", lw=2,
            label=f"Median = {score_df['score'].median():.1f}")
ax2.set_title("Inspection Score Distribution")
ax2.set_xlabel("Inspection score (lower is better)")
ax2.set_ylabel("Count"); ax2.legend(fontsize=9)
fig.suptitle("3.1A  Inspection Outcomes and Score Distribution",
             fontsize=13, fontweight="bold")
savefig("fig01a_outcome_score.png")

# fig01b ── violation + critical violation
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
ax1.hist(df["violation_count"], bins=range(0, int(df["violation_count"].max())+2),
         color=C_BLUE, edgecolor="white", linewidth=0.4)
ax1.set_xlim(-0.5, min(12.5, df["violation_count"].max()+0.5))
ax1.set_title("Violation Count Distribution")
ax1.set_xlabel("Total violations per inspection"); ax1.set_ylabel("Count")

ax2.hist(df["critical_count"], bins=range(0, int(df["critical_count"].max())+2),
         color=C_FAIL, alpha=0.85, edgecolor="white", linewidth=0.4)
ax2.set_xlim(-0.5, min(10.5, df["critical_count"].max()+0.5))
ax2.set_title("Critical Violation Count Distribution")
ax2.set_xlabel("Critical violations per inspection"); ax2.set_ylabel("Count")
fig.suptitle("3.1B  Violation and Critical Violation Distributions",
             fontsize=13, fontweight="bold")
savefig("fig01b_violations.png")

# fig01c ── grade + monthly volume
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
grade_order  = ["A","B","C","No grade"]
grade_colors = [C_PASS, C_GOLD, C_FAIL, C_GRAY]
grade_counts = df["grade_display"].value_counts().reindex(grade_order).fillna(0)
ax1.bar(grade_counts.index, grade_counts.values,
        color=grade_colors, edgecolor="white", linewidth=1.2)
for i, v in enumerate(grade_counts.values):
    if v > 0:
        ax1.text(i, v+len(df)*0.004, f"{v:,.0f}", ha="center", fontsize=9)
ax1.set_title("Grade Distribution")
ax1.set_xlabel("Grade"); ax1.set_ylabel("Number of inspections")

monthly_counts = df["month_name"].value_counts().reindex(MONTH_LABELS)
ax2.plot(range(len(MONTH_LABELS)), monthly_counts.values,
         marker="o", color=C_BLUE, linewidth=2, markersize=7)
ax2.fill_between(range(len(MONTH_LABELS)), monthly_counts.values,
                 alpha=0.12, color=C_BLUE)
ax2.set_xticks(range(len(MONTH_LABELS)))
ax2.set_xticklabels(MONTH_LABELS, rotation=45, ha="right")
ax2.set_title("Inspection Volume by Month")
ax2.set_xlabel("Month"); ax2.set_ylabel("Number of inspections")
fig.suptitle("3.1C  Grade Distribution and Monthly Inspection Volume",
             fontsize=13, fontweight="bold")
savefig("fig01c_grade_timing.png")

# fig02a ── borough + cuisine
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
boro_counts = df["boro"].value_counts().reindex(BOROUGH_ORDER)
ax1.barh(boro_counts.index, boro_counts.values,
         color=BOROUGH_COLORS, edgecolor="white", linewidth=0.8)
ax1.set_title("Inspection Count by Borough")
ax1.set_xlabel("Number of inspections")
for i, v in enumerate(boro_counts.values):
    ax1.text(v+80, i, f"{v:,}", va="center", fontsize=9)

top_cuisine = df["cuisine_grouped"].value_counts().head(15).sort_values()
ax2.barh(top_cuisine.index, top_cuisine.values,
         color=gradient_colors("Blues", len(top_cuisine)), edgecolor="white")
ax2.set_title("Top 15 Cuisine Groups")
ax2.set_xlabel("Number of inspections")
fig.suptitle("3.1D  Borough and Cuisine Group Distributions",
             fontsize=13, fontweight="bold")
savefig("fig02a_borough_cuisine.png")

# fig02b ── yelp rating + yelp price
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
ax1.hist(yelp_df["yelp_rating"].dropna(), bins=np.arange(1, 5.25, 0.25),
         color=C_GOLD, alpha=0.85, edgecolor="white", linewidth=0.4)
ax1.axvline(4.0, color=C_FAIL, linestyle="--", lw=2, label="High-rating threshold = 4.0")
ax1.set_title("Yelp Rating Distribution")
ax1.set_xlabel("Yelp rating"); ax1.set_ylabel("Count"); ax1.legend(fontsize=9)

price_counts = yelp_df["yelp_price"].dropna().astype(int).value_counts().sort_index()
price_colors = [C_BLUE, C_PASS, C_GOLD, C_FAIL][:len(price_counts)]
ax2.bar(price_counts.index.astype(str), price_counts.values,
        color=price_colors, edgecolor="white", linewidth=1)
for i, v in enumerate(price_counts.values):
    ax2.text(i, v+30, f"{v:,}", ha="center", fontsize=9)
ax2.set_title("Yelp Price Tier Distribution")
ax2.set_xlabel(r"Price tier (\$, \$\$, \$\$\$)"); ax2.set_ylabel("Count")
fig.suptitle("3.1E  Yelp Rating and Price Tier Distributions",
             fontsize=13, fontweight="bold")
savefig("fig02b_yelp_rating_price.png")

# =============================================================================
# 3.2  BIVARIATE
# =============================================================================
print("\n── 3.2 Bivariate ──")
analysis = df.copy()
analysis["violation_bucket"] = pd.cut(
    analysis["violation_count"], bins=[-0.1,0,1,2,3,4,np.inf],
    labels=["0","1","2","3","4","5+"])
analysis["critical_bucket"] = pd.cut(
    analysis["critical_count"], bins=[-0.1,0,1,2,3,np.inf],
    labels=["0","1","2","3","4+"])

# fig03a ── score by outcome + violation failure rate
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
sns.boxplot(data=df, x="failed_label", y="score",
            order=["Passed","Failed"], palette=PASS_FAIL_PAL,
            ax=ax1, linewidth=1.2)
ax1.axhline(28, color=C_FAIL, linestyle="--", lw=1.8, label="Threshold = 28")
ax1.set_title("Inspection Score by Outcome")
ax1.set_xlabel("Outcome"); ax1.set_ylabel("Inspection score")
ax1.legend(fontsize=9)

vrates = analysis.groupby("violation_bucket", observed=False)["failed"].mean().reset_index()
ax2.bar(vrates["violation_bucket"].astype(str), vrates["failed"],
        color=gradient_colors("OrRd", len(vrates)), edgecolor="white")
fmt_pct(ax2)
ax2.set_title("Failure Rate by Violation Count")
ax2.set_xlabel("Violation count bucket"); ax2.set_ylabel("Failure rate")
fig.suptitle("3.2A  Inspection Score and Violation Failure Rates",
             fontsize=13, fontweight="bold")
savefig("fig03a_score_violation.png")

# fig03b ── critical failure rate + prev_score scatter
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
crates = analysis.groupby("critical_bucket", observed=False)["failed"].mean().reset_index()
ax1.bar(crates["critical_bucket"].astype(str), crates["failed"],
        color=gradient_colors("Reds", len(crates)), edgecolor="white")
fmt_pct(ax1)
ax1.set_title("Failure Rate by Critical Violation Count")
ax1.set_xlabel("Critical violation bucket"); ax1.set_ylabel("Failure rate")

history_df = df[(df["has_history"]==1)&(df["prev_score"]>=0)&df["score"].notna()].copy()
hplot = history_df.sample(n=min(10000, len(history_df)), random_state=RANDOM_STATE)
ax2.scatter(hplot["prev_score"], hplot["score"],
            alpha=0.15, s=14, color=C_BLUE, rasterized=True)
m, b = np.polyfit(hplot["prev_score"].dropna(), hplot["score"].dropna(), 1)
xs = np.linspace(hplot["prev_score"].min(), hplot["prev_score"].max(), 100)
ax2.plot(xs, m*xs+b, color=C_FAIL, linewidth=1.8)
ax2.axhline(28, color=C_FAIL, linestyle="--", lw=1, alpha=0.7)
ax2.axvline(28, color=C_FAIL, linestyle="--", lw=1, alpha=0.7)
ax2.set_title("Previous Score vs Current Score")
ax2.set_xlabel("Previous inspection score"); ax2.set_ylabel("Current inspection score")
fig.suptitle("3.2B  Critical Violations and Historical Score Patterns",
             fontsize=13, fontweight="bold")
savefig("fig03b_critical_history.png")

# fig03c ── score trend by outcome + first vs repeat
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
sns.boxplot(data=df, x="failed_label", y="score_trend",
            order=["Passed","Failed"], palette=PASS_FAIL_PAL,
            ax=ax1, linewidth=1.2)
ax1.axhline(0, color="black", linestyle=":", lw=1.2)
ax1.set_title("Score Trend by Outcome")
ax1.set_xlabel("Outcome"); ax1.set_ylabel("Current score minus previous score")

first_summary = (df.groupby("is_first_inspection")
                   .agg(failure_rate=("failed","mean"))
                   .rename(index={0:"Repeat", 1:"First"}).reset_index())
ax2.bar(["Repeat","First"], first_summary["failure_rate"],
        color=[C_BLUE, C_GOLD], edgecolor="white", lw=1.2, width=0.5)
fmt_pct(ax2)
for i, v in enumerate(first_summary["failure_rate"]):
    ax2.text(i, v+0.003, f"{v:.1%}", ha="center", fontsize=10)
ax2.set_title("Failure Rate: First vs Repeat Inspections")
ax2.set_xlabel("Inspection type"); ax2.set_ylabel("Failure rate")
ax2.set_ylim(0, first_summary["failure_rate"].max()*1.2)
fig.suptitle("3.2C  Score Trend and First-Inspection Effect",
             fontsize=13, fontweight="bold")
savefig("fig03c_trend_firstrepeat.png")

# fig04a ── yelp vs score + yelp by outcome
yelp_plot = (df[df["has_yelp"]==1].dropna(subset=["yelp_rating","score"])
               .sample(n=min(8000, df["has_yelp"].sum()), random_state=RANDOM_STATE))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
ax1.scatter(yelp_plot["yelp_rating"], yelp_plot["score"],
            alpha=0.2, s=14, color=C_GOLD, rasterized=True)
m2, b2 = np.polyfit(yelp_plot["yelp_rating"], yelp_plot["score"], 1)
xs2 = np.linspace(yelp_plot["yelp_rating"].min(), yelp_plot["yelp_rating"].max(), 100)
ax1.plot(xs2, m2*xs2+b2, color=C_FAIL, lw=1.8)
ax1.axhline(28, color=C_FAIL, linestyle="--", lw=1.5, alpha=0.7)
ax1.axvline(4.0, color=C_GOLD, linestyle="--", lw=1.5, alpha=0.7)
ax1.set_title("Yelp Rating vs Inspection Score")
ax1.set_xlabel("Yelp rating"); ax1.set_ylabel("Inspection score")

sns.boxplot(data=yelp_plot, x="failed_label", y="yelp_rating",
            order=["Passed","Failed"], palette=PASS_FAIL_PAL,
            ax=ax2, linewidth=1.2)
ax2.axhline(4.0, color=C_GOLD, linestyle="--", lw=1.5, label="High-rating = 4.0")
ax2.set_title("Yelp Rating by Inspection Outcome")
ax2.set_xlabel("Outcome"); ax2.set_ylabel("Yelp rating"); ax2.legend(fontsize=9)
fig.suptitle("3.2D  Yelp Rating vs Inspection Performance",
             fontsize=13, fontweight="bold")
savefig("fig04a_yelp_vs_outcome.png")

# fig04b ── monthly failure rate + temperature failure rate
df["temp_bin"] = pd.cut(df["temp_mean"], bins=[-20,0,10,20,30,45],
                         labels=["<=0","0-10","10-20","20-30","30+"])
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
monthly_rate = df.groupby("month_name", observed=False)["failed"].mean().reset_index()
ax1.plot(range(len(monthly_rate)), monthly_rate["failed"],
         marker="o", color=C_FAIL, lw=2, markersize=7)
ax1.fill_between(range(len(monthly_rate)), monthly_rate["failed"],
                 alpha=0.12, color=C_FAIL)
ax1.set_xticks(range(len(MONTH_LABELS)))
ax1.set_xticklabels(MONTH_LABELS, rotation=45, ha="right")
fmt_pct(ax1)
ax1.set_title("Failure Rate by Month")
ax1.set_xlabel("Month"); ax1.set_ylabel("Failure rate")

temp_rates = df.groupby("temp_bin", observed=False)["failed"].mean().reset_index()
ax2.bar(temp_rates["temp_bin"].astype(str), temp_rates["failed"],
        color=gradient_colors("coolwarm", len(temp_rates)), edgecolor="white")
fmt_pct(ax2)
ax2.set_title("Failure Rate by Temperature Bin")
ax2.set_xlabel("Mean daily temperature (C)"); ax2.set_ylabel("Failure rate")
fig.suptitle("3.2E  Temporal and Weather Effects on Failure Rate",
             fontsize=13, fontweight="bold")
savefig("fig04b_monthly_temp_failure.png")

# fig05a ── failure by borough + income scatter
borough_summary = (df.groupby("boro")
                     .agg(inspections=("failed","size"),
                          failure_rate=("failed","mean"),
                          median_income=("median_household_income","median"))
                     .reindex(BOROUGH_ORDER).reset_index())
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
ax1.bar(borough_summary["boro"], borough_summary["failure_rate"],
        color=BOROUGH_COLORS, edgecolor="white", lw=1)
ax1.set_xticklabels(borough_summary["boro"], rotation=30, ha="right")
fmt_pct(ax1)
for i, v in enumerate(borough_summary["failure_rate"]):
    ax1.text(i, v+0.003, f"{v:.1%}", ha="center", fontsize=9)
ax1.set_title("Inspection Failure Rate by Borough")
ax1.set_xlabel("Borough"); ax1.set_ylabel("Failure rate")
ax1.set_ylim(0, borough_summary["failure_rate"].max()*1.2)

for i, row in borough_summary.iterrows():
    ax2.scatter(row["median_income"], row["failure_rate"],
                s=row["inspections"]/8, color=BOROUGH_COLORS[i],
                alpha=0.85, edgecolors="white", linewidth=0.8)
    ax2.text(row["median_income"], row["failure_rate"]+0.002,
             row["boro"].title(), ha="center", fontsize=8.5)
fmt_pct(ax2, axis="y")
ax2.set_title("Income vs Failure Rate by Borough")
ax2.set_xlabel("Median household income ($)"); ax2.set_ylabel("Failure rate")
fig.suptitle("3.2F  Borough-Level Failure Rates and Socioeconomic Context",
             fontsize=13, fontweight="bold")
savefig("fig05a_borough_failure.png")

# fig05b ── top cuisine failure  (single wide)
cuisine_summary = (df.groupby("cuisine_grouped")
                     .agg(inspections=("failed","size"),
                          failure_rate=("failed","mean"))
                     .query("inspections >= 100")
                     .sort_values("failure_rate", ascending=False))
top_risk = cuisine_summary.head(15).sort_values("failure_rate")
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(top_risk.index, top_risk["failure_rate"],
        color=gradient_colors("OrRd", len(top_risk)), edgecolor="white")
fmt_pct(ax, axis="x")
for i, (idx, row) in enumerate(top_risk.iterrows()):
    ax.text(row["failure_rate"]+0.002, i, f"{row['failure_rate']:.1%}",
            va="center", fontsize=9)
ax.set_title("Top 15 Highest Failure-Rate Cuisine Groups",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Failure rate")
savefig("fig05b_cuisine_failure.png")

# =============================================================================
# 3.3  INTERACTION EFFECTS
# =============================================================================
print("\n── 3.3 Interaction Effects ──")
interaction_df = df.dropna(subset=["median_household_income",
                                    "food_complaints_total","failed"]).copy()
interaction_df["income_quartile"] = pd.qcut(
    interaction_df["median_household_income"].rank(method="first"), 4,
    labels=["Q1 low income","Q2","Q3","Q4 high income"])
interaction_df["complaint_quartile"] = pd.qcut(
    interaction_df["food_complaints_total"].rank(method="first"), 4,
    labels=["Q1 low complaints","Q2","Q3","Q4 high complaints"])

income_complaint_heat = interaction_df.pivot_table(
    index="income_quartile", columns="complaint_quartile",
    values="failed", aggfunc="mean", observed=False)

top_cuisines = df["cuisine_grouped"].value_counts().head(12).index
cuisine_boro_heat = (
    df[df["cuisine_grouped"].isin(top_cuisines)]
    .pivot_table(index="boro", columns="cuisine_grouped",
                 values="failed", aggfunc="mean", observed=False)
    .reindex(BOROUGH_ORDER))
month_boro_heat = (
    df.pivot_table(index="boro", columns="month_name",
                   values="failed", aggfunc="mean", observed=False)
    .reindex(BOROUGH_ORDER))

# fig06a
fig, ax = plt.subplots(figsize=(9, 5))
sns.heatmap(income_complaint_heat, annot=True, fmt=".1%",
            cmap="YlOrRd", linewidths=0.5, ax=ax,
            cbar_kws={"format": mticker.FuncFormatter(lambda x, _: f"{x:.0%}")})
ax.set_title("Failure Rate: Income × 311 Food Complaints",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Complaint quartile"); ax.set_ylabel("Income quartile")
savefig("fig06a_income_complaint_heatmap.png")

# fig06b
fig, ax = plt.subplots(figsize=(14, 5))
sns.heatmap(cuisine_boro_heat, annot=True, fmt=".1%",
            cmap="rocket_r", linewidths=0.4, ax=ax,
            cbar_kws={"format": mticker.FuncFormatter(lambda x, _: f"{x:.0%}")})
ax.set_title("Failure Rate: Cuisine Group × Borough",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Cuisine group"); ax.set_ylabel("Borough")
ax.tick_params(axis="x", rotation=40, labelsize=8.5)
savefig("fig06b_cuisine_borough_heatmap.png")

# fig06c
fig, ax = plt.subplots(figsize=(14, 4))
sns.heatmap(month_boro_heat, annot=True, fmt=".1%",
            cmap="YlGnBu", linewidths=0.4, ax=ax,
            cbar_kws={"format": mticker.FuncFormatter(lambda x, _: f"{x:.0%}")})
ax.set_title("Failure Rate: Borough × Month", fontsize=13, fontweight="bold")
ax.set_xlabel("Month"); ax.set_ylabel("Borough")
savefig("fig06c_borough_month_heatmap.png")

# fig07 ── correlation heatmap
predictor_corr_cols = [
    "violation_count","critical_count","temp_mean","precipitation_sum","rain_sum",
    "snowfall_sum","wind_speed_mean","wind_gust_mean","cloud_cover_mean",
    "food_complaints_total","rodent_complaints","food_safety_complaints",
    "median_household_income","total_population","white_population","white_pct",
    "inspection_month","inspection_dow","is_weekend","prev_score","prev_failed",
    "inspection_count","score_trend","grade_available","has_yelp",
    "is_first_inspection","has_history","has_location",
]
predictor_corr_cols = [c for c in predictor_corr_cols if c in df.columns]
corr = df[predictor_corr_cols].corr(numeric_only=True)
mask = np.triu(np.ones_like(corr, dtype=bool))
fig, ax = plt.subplots(figsize=(13, 11))
sns.heatmap(corr, mask=mask, cmap="vlag", center=0,
            linewidths=0.3, ax=ax, cbar_kws={"shrink": 0.7})
ax.set_title("Numeric Predictor Correlation Heatmap", fontsize=13, fontweight="bold")
ax.tick_params(labelsize=8)
savefig("fig07_correlation_heatmap.png")

# fig08 ── target correlation bars
target_corr = (
    df[predictor_corr_cols+["failed"]].corr(numeric_only=True)["failed"]
    .drop("failed")
    .sort_values(key=lambda s: s.abs(), ascending=False)
    .head(20).reset_index()
    .rename(columns={"index":"feature","failed":"corr_with_failed"}))
fig, ax = plt.subplots(figsize=(10, 7))
colors_corr = [C_FAIL if v>0 else C_BLUE for v in target_corr["corr_with_failed"]]
ax.barh(target_corr["feature"], target_corr["corr_with_failed"],
        color=colors_corr, edgecolor="white")
ax.axvline(0, color="black", lw=1)
ax.set_title("Top Numeric Correlations with Failure Target",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Pearson correlation with failed")
savefig("fig08_target_correlations.png")

# =============================================================================
# 3.4  PCA
# =============================================================================
print("\n── 3.4 PCA ──")
pca_features = [
    "score","violation_count","critical_count","temp_mean","precipitation_sum",
    "food_complaints_total","rodent_complaints","median_household_income",
    "white_pct","prev_score","inspection_count","score_trend",
]
pca_features = [c for c in pca_features if c in df.columns]

X_pca_scaled = StandardScaler().fit_transform(
    SimpleImputer(strategy="median").fit_transform(df[pca_features]))
n_components = min(len(pca_features), 12)
pca          = PCA(n_components=n_components, random_state=RANDOM_STATE)
pca_scores   = pca.fit_transform(X_pca_scaled)

pca_cols = [f"PC{i}" for i in range(1, n_components+1)]
pca_df   = pd.DataFrame(pca_scores, columns=pca_cols, index=df.index)
pca_df["failed"]       = df["failed"].values
pca_df["failed_label"] = df["failed_label"].values
pca_df["boro"]         = df["boro"].values

explained = pd.DataFrame({
    "component": pca_cols,
    "evr": pca.explained_variance_ratio_,
    "cumvar": np.cumsum(pca.explained_variance_ratio_)})
n80 = int(np.argmax(explained["cumvar"].values >= 0.80)+1)
print(f"Components for 80% variance: {n80}")
display(explained)

# fig09a ── scree + cumulative
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
ax1.bar(range(len(pca_cols)), pca.explained_variance_ratio_,
        color=[C_BLUE if i<n80 else C_GRAY for i in range(len(pca_cols))],
        edgecolor="white")
ax1.set_xticks(range(len(pca_cols)))
ax1.set_xticklabels(pca_cols, rotation=45)
fmt_pct(ax1)
ax1.set_title("PCA Scree Plot")
ax1.set_xlabel("Principal component"); ax1.set_ylabel("Explained variance ratio")

ax2.plot(range(len(pca_cols)), explained["cumvar"],
         marker="o", color=C_FAIL, lw=2, markersize=7)
ax2.axhline(0.80, color="black", linestyle="--", lw=1.2, label="80% threshold")
ax2.fill_between(range(len(pca_cols)), explained["cumvar"],
                 alpha=0.12, color=C_FAIL)
ax2.set_xticks(range(len(pca_cols)))
ax2.set_xticklabels(pca_cols, rotation=45)
fmt_pct(ax2)
ax2.set_title("Cumulative Explained Variance")
ax2.set_xlabel("Principal component"); ax2.set_ylabel("Cumulative variance")
ax2.legend(fontsize=9)
fig.suptitle("3.4A  PCA Explained Variance", fontsize=13, fontweight="bold")
savefig("fig09a_pca_variance.png")

# fig09b ── 2D projection + PC1 density
pca_plot = pca_df.sample(n=min(12000, len(pca_df)), random_state=RANDOM_STATE)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
for label, color in PASS_FAIL_PAL.items():
    sub = pca_plot[pca_plot["failed_label"]==label]
    ax1.scatter(sub["PC1"], sub["PC2"], alpha=0.25, s=14,
                color=color, label=label, rasterized=True)
ax1.set_title("PCA 2D Projection Colored by Outcome")
ax1.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
ax1.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
ax1.legend(fontsize=9)

for label, color in PASS_FAIL_PAL.items():
    sub = pca_plot[pca_plot["failed_label"]==label]["PC1"]
    ax2.hist(sub, bins=50, alpha=0.45, color=color, label=label, density=True)
ax2.set_title("PC1 Density by Outcome")
ax2.set_xlabel("PC1"); ax2.set_ylabel("Density"); ax2.legend(fontsize=9)
fig.suptitle("3.4B  PCA Projection and PC1 Outcome Separation",
             fontsize=13, fontweight="bold")
savefig("fig09b_pca_projection.png")

# fig10a ── PC1 + PC2 loadings
loadings = pd.DataFrame(pca.components_.T, index=pca_features, columns=pca_cols)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
for ax, pc in [(ax1,"PC1"), (ax2,"PC2")]:
    vals = loadings[pc].sort_values()
    ax.barh(vals.index, vals.values,
            color=[C_FAIL if v>=0 else C_BLUE for v in vals.values],
            edgecolor="white")
    ax.axvline(0, color="black", lw=1)
    ax.set_title(f"{pc} Feature Loadings")
    ax.set_xlabel("Loading")
fig.suptitle("3.4C  PCA Feature Loadings", fontsize=13, fontweight="bold")
savefig("fig10a_pca_loadings.png")

# fig10b ── PC1 by borough
fig, ax = plt.subplots(figsize=(10, 5))
sns.boxplot(data=pca_df, x="boro", y="PC1", order=BOROUGH_ORDER,
            palette=dict(zip(BOROUGH_ORDER, BOROUGH_COLORS)),
            ax=ax, linewidth=1.2)
ax.set_title("PC1 Distribution by Borough", fontsize=13, fontweight="bold")
ax.set_xlabel("Borough"); ax.set_ylabel("PC1")
ax.tick_params(axis="x", rotation=25)
savefig("fig10b_pc1_borough.png")

# =============================================================================
# 3.5  KMEANS
# =============================================================================
print("\n── 3.5 KMeans ──")
cluster_features = [
    "violation_count","critical_count","temp_mean","precipitation_sum",
    "food_complaints_total","rodent_complaints","median_household_income",
    "white_pct","prev_score","prev_failed","inspection_count",
    "has_history","is_first_inspection","has_yelp","has_location",
]
cluster_features = [c for c in cluster_features if c in df.columns]
X_cluster = StandardScaler().fit_transform(
    SimpleImputer(strategy="median").fit_transform(df[cluster_features]))

rng     = np.random.default_rng(RANDOM_STATE)
sil_idx = rng.choice(X_cluster.shape[0],
                      size=min(10000, X_cluster.shape[0]), replace=False)
cluster_eval = []
for k in range(2, 9):
    km  = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=25)
    lbl = km.fit_predict(X_cluster)
    cluster_eval.append({
        "k": k, "inertia": km.inertia_,
        "silhouette": silhouette_score(X_cluster[sil_idx], lbl[sil_idx])})
cluster_eval_df = pd.DataFrame(cluster_eval)
display(cluster_eval_df)

best_candidates = cluster_eval_df[cluster_eval_df["k"].isin([3,4])]
FINAL_K = int(best_candidates.loc[best_candidates["silhouette"].idxmax(), "k"])
print(f"Selected K = {FINAL_K}")

# fig11 ── elbow + silhouette
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
ax1.plot(cluster_eval_df["k"], cluster_eval_df["inertia"],
         marker="o", color=C_BLUE, lw=2, markersize=8)
ax1.fill_between(cluster_eval_df["k"], cluster_eval_df["inertia"],
                 alpha=0.1, color=C_BLUE)
ax1.set_title("KMeans Elbow Curve")
ax1.set_xlabel("Number of clusters K"); ax1.set_ylabel("Inertia")

ax2.plot(cluster_eval_df["k"], cluster_eval_df["silhouette"],
         marker="o", color=C_FAIL, lw=2, markersize=8)
ax2.axvline(FINAL_K, color=C_GOLD, linestyle="--", lw=1.8,
            label=f"Selected K = {FINAL_K}")
ax2.set_title("KMeans Silhouette Score")
ax2.set_xlabel("Number of clusters K"); ax2.set_ylabel("Silhouette score")
ax2.legend(fontsize=9)
fig.suptitle("3.5A  KMeans Model Selection", fontsize=13, fontweight="bold")
savefig("fig11_kmeans_selection.png")

# Fit final model
kmeans = KMeans(n_clusters=FINAL_K, random_state=RANDOM_STATE, n_init=50)
df["cluster"]  = kmeans.fit_predict(X_cluster)
pca_df["cluster"] = df["cluster"].values

profile = (df.groupby("cluster")
             .agg(inspections=("failed","size"),
                  failure_rate=("failed","mean"),
                  avg_prev_score=("prev_score","mean"),
                  history_rate=("has_history","mean"),
                  first_inspection_rate=("is_first_inspection","mean"),
                  avg_food_complaints=("food_complaints_total","mean"),
                  median_income=("median_household_income","median"),
                  yelp_coverage=("has_yelp","mean"))
             .sort_index())

cluster_names = {}
for c, row in profile.iterrows():
    if row["failure_rate"] == profile["failure_rate"].min():
        cluster_names[c] = ("Yelp-Matched Stable Compliant"
                             if row["yelp_coverage"] >= 0.8 else "Stable Compliant")
    elif row["first_inspection_rate"] >= 0.9 or row["history_rate"] <= 0.1:
        cluster_names[c] = "No-History Higher Risk"
    elif (row["history_rate"] >= 0.8 and
          row["avg_prev_score"] >= profile["avg_prev_score"].median()):
        cluster_names[c] = "Repeat Operators With Prior Risk"
    else:
        cluster_names[c] = "High-Risk Mixed Profile"

df["cluster_name"]     = df["cluster"].map(cluster_names)
pca_df["cluster_name"] = df["cluster_name"].values

CLUSTER_COLORS_LIST = [C_PASS, C_BLUE, C_GOLD, C_FAIL, C_PURPLE][:FINAL_K]
cluster_color_map   = {name: CLUSTER_COLORS_LIST[i]
                        for i, name in enumerate(sorted(set(cluster_names.values())))}

# fig12a ── cluster size + failure rate
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
counts = df["cluster_name"].value_counts()
ax1.barh(counts.index, counts.values,
         color=[cluster_color_map.get(n, C_GRAY) for n in counts.index],
         edgecolor="white")
ax1.set_title("Cluster Sizes"); ax1.set_xlabel("Number of inspections")

fr_by_cluster = df.groupby("cluster_name")["failed"].mean().sort_values()
ax2.barh(fr_by_cluster.index, fr_by_cluster.values,
         color=[cluster_color_map.get(n, C_GRAY) for n in fr_by_cluster.index],
         edgecolor="white")
fmt_pct(ax2, axis="x")
for i, v in enumerate(fr_by_cluster.values):
    ax2.text(v+0.003, i, f"{v:.1%}", va="center", fontsize=9)
ax2.set_title("Failure Rate by Cluster"); ax2.set_xlabel("Failure rate")
fig.suptitle("3.5B  KMeans Cluster Overview", fontsize=13, fontweight="bold")
savefig("fig12a_cluster_overview.png")

# fig12b ── profile heatmap + PCA colored by cluster
profile_numeric = profile[["failure_rate","history_rate","first_inspection_rate",
                             "avg_food_complaints","median_income","yelp_coverage"]].copy()
profile_numeric.index = [cluster_names[i] for i in profile_numeric.index]
profile_scaled = pd.DataFrame(
    StandardScaler().fit_transform(profile_numeric),
    index=profile_numeric.index, columns=profile_numeric.columns)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
sns.heatmap(profile_scaled, annot=True, fmt=".2f", cmap="vlag",
            center=0, linewidths=0.4, ax=ax1)
ax1.set_title("Standardized Cluster Profile Heatmap")
ax1.tick_params(axis="x", rotation=35, labelsize=8.5)

pca_cplot = pca_df.sample(n=min(12000, len(pca_df)), random_state=RANDOM_STATE)
for name, color in cluster_color_map.items():
    sub = pca_cplot[pca_cplot["cluster_name"]==name]
    ax2.scatter(sub["PC1"], sub["PC2"], alpha=0.3, s=14,
                color=color, label=name, rasterized=True)
ax2.set_title("PCA Projection by KMeans Cluster")
ax2.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
ax2.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
ax2.legend(fontsize=7.5, bbox_to_anchor=(1.01,1), loc="upper left")
fig.suptitle("3.5C  Cluster Profiles and PCA Projection",
             fontsize=13, fontweight="bold")
savefig("fig12b_cluster_profiles_pca.png")

# fig13 ── composition (borough + outcome)
cluster_boro     = pd.crosstab(df["cluster_name"], df["boro"], normalize="index").reindex(columns=BOROUGH_ORDER)
cluster_fail_tab = pd.crosstab(df["cluster_name"], df["failed_label"], normalize="index").reindex(columns=["Passed","Failed"])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
cluster_boro.plot(kind="barh", stacked=True, ax=ax1, color=BOROUGH_COLORS)
ax1.set_title("Borough Mix Within Each Cluster")
ax1.set_xlabel("Share of cluster")
fmt_pct(ax1, axis="x")
ax1.legend(title="Borough", bbox_to_anchor=(1.01,1), loc="upper left", fontsize=8)

cluster_fail_tab.plot(kind="barh", stacked=True, ax=ax2, color=[C_PASS, C_FAIL])
ax2.set_title("Outcome Mix Within Each Cluster")
ax2.set_xlabel("Share of cluster")
fmt_pct(ax2, axis="x")
ax2.legend(title="Outcome", bbox_to_anchor=(1.01,1), loc="upper left", fontsize=8)
fig.suptitle("3.5D  Cluster Composition by Borough and Outcome",
             fontsize=13, fontweight="bold")
savefig("fig13_cluster_composition.png")

# ── Save clustered dataset ────────────────────────────────────────────────────
clustered_path = DATA_DIR / "restaurant_clean_clustered.csv"
handoff_df = df[base_columns + ["cluster","cluster_name"]].copy()
handoff_df.to_csv(clustered_path, index=False)
print(f"\nSaved clustered dataset: {clustered_path}")
print(f"Rows: {len(handoff_df):,}  Cols: {handoff_df.shape[1]}")
print("\nAll figures saved to:", FIG_DIR)
