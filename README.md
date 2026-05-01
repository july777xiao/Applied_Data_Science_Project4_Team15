# Project 4: Predicting Restaurant Inspection Failures in New York City

## A Multi-Source Machine Learning Approach

> Columbia University · GU4243/GR5243 Applied Data Science · Spring 2026 · Team 15

**Team Members**
| Name | UNI | Role |
|------|-----|------|
| Xiao Xiao | xx2492 | Data acquisition, data cleaning, web application development |
| Shuzhi Yang | sy3321 | Exploratory data analysis, PCA, KMeans clustering |
| Pingyu Zhou | pz2341 | Feature engineering, leakage prevention, logistic regression |
| Junyang Li | jl7230 | Random Forest, Gradient Boosting, model evaluation and selection |

---

## 🔗 Links

- **Web Application**: [NYC Restaurant Intelligence on Streamlit](https://july777xiaoapplieddatascienceproject4team15.streamlit.app)
- **GitHub Repository**: [Applied_Data_Science_Project4_Team15](https://github.com/july777xiao/Applied_Data_Science_Project4_Team15)

---

## Project Overview

Food safety is a critical public health concern in New York City, where the Department of Health and Mental Hygiene (DOHMH) conducts tens of thousands of restaurant inspections each year. This project addresses the question: **can we predict whether a restaurant will fail its next health inspection before the inspector arrives?**

This project builds an end-to-end machine learning pipeline to predict two outcomes:

- **Primary task**: Will a restaurant **fail** its next health inspection? (binary classification, `failed = 1` if score ≥ 28)
- **Secondary task**: Does a restaurant have a **high Yelp rating** (≥ 4.0 stars)? (binary classification on Yelp-matched subset)

The dataset integrates six sources — regulatory inspection records, Yelp consumer data, daily weather, NYC 311 complaints, Census demographics, and violation code references — covering **41,556 inspection records** across **22,198 unique restaurants** in all five NYC boroughs from January 2023 to December 2024. The overall inspection failure rate is **18.0%**, indicating a moderately imbalanced classification setting.

---

## Repository Structure

```
Applied_Data_Science_Project4_Team15/
│
├── data/
│   ├── processed/                       # Cleaned and processed datasets (tracked by git)
│   │   ├── restaurant_clean.csv         # Final cleaned dataset (41,556 rows × 52 cols)
│   │   ├── restaurant_yelp_subset.csv   # Yelp-matched subset (8,279 rows)
│   │   └── restaurant_clean_clustered.csv  # With KMeans cluster labels (54 cols)
│   └── raw/                             # Raw data (not tracked by git)
│
├── figures/                             # All generated figures
│   ├── fig01a_outcome_score.png         # EDA figures (fig01a – fig13)
│   ├── ...
│   ├── logistic_roc_curve.png           # Logistic regression evaluation figures
│   ├── model_comparison_table.png       # Model comparison figures
│   ├── final_model_confusion_matrix.png
│   └── website/                         # Web app screenshots for report
│
├── models/
│   └── best_model.pkl                   # Final trained Logistic Regression model
│
├── outputs/                             # CSV outputs from modeling scripts
│   ├── model_comparison_metrics.csv
│   ├── final_model_selected_metrics.csv
│   ├── logistic_regression_coefficients.csv
│   ├── leakage_audit.csv
│   └── ...
│
├── 1_data_acquisition.py        # API calls, web scraping, panel construction
├── 2_data_cleaning.py           # Data cleaning, preprocessing, feature engineering
├── 3_EDA_PCA_KMeans.py          # EDA visualizations, PCA, KMeans clustering
├── 4_Feature_Engineering.py     # Leakage-safe feature set + logistic regression baseline
├── 5_supervised_modeling.py     # RF, GBM, model comparison, final model selection
├── app.py                       # Streamlit web application
├── requirements.txt             # Python dependencies
└── runtime.txt                  # Python version (3.11) for Streamlit Cloud
```

---

## Data Sources

| Source | Method | Records | Key Variables |
|--------|--------|---------|---------------|
| NYC DOHMH Restaurant Inspections | Socrata API | 142,591 raw → 41,556 cleaned | score, grade, violation codes, borough |
| Restaurant Coordinates | Socrata API | 4,487 geocoded | latitude, longitude |
| Yelp Business Data | Yelp Fusion API | ~3,000 matched restaurants | yelp_rating, yelp_reviews, yelp_price |
| Weather (Open-Meteo) | REST API (free, no key required) | 731 daily records | temp, precipitation, wind speed |
| NYC 311 Food Complaints | Socrata API | Borough × Date aggregated | food_complaints_total, rodent_complaints |
| U.S. Census ACS 2022 | Census Bureau API | 5 borough records | median_income, total_population |
| Violation Code Reference | Web scraping (fallback: static table) | 8 common codes | severity, category |

---

## Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/july777xiao/Applied_Data_Science_Project4_Team15.git
cd Applied_Data_Science_Project4_Team15
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. API Keys Required

Only the **Yelp Fusion API** requires a key. All other APIs are free and keyless.

| API | How to obtain |
|-----|--------------|
| Yelp Fusion | [Register at yelp.com/developers](https://docs.developer.yelp.com/docs/fusion-intro) — free tier |
| NYC Open Data (Socrata) | Optional app token at [data.cityofnewyork.us](https://data.cityofnewyork.us) — increases rate limit |
| Open-Meteo | No key required |
| U.S. Census | [api.census.gov/data/key_signup.html](https://api.census.gov/data/key_signup.html) — free |

Set the Yelp key in `1_data_acquisition.py`:
```python
YELP_API_KEY = "YOUR_YELP_API_KEY"
```

---

## How to Run

### Step 1 — Data Acquisition
```bash
python 1_data_acquisition.py
```
- Fetches all 6 data sources via API
- Saves raw files locally
- **Note**: The Yelp step takes ~1 hour (rate-limited at 0.25s/request). Supports interrupted runs via cache file.
- Output: raw panel CSV in `data/raw/`

### Step 2 — Data Cleaning
```bash
python 2_data_cleaning.py
```
- Runs full cleaning pipeline: type conversion, invalid record removal, structural missingness handling, sentinel encoding for first inspections, cuisine simplification
- Output: `data/processed/restaurant_clean.csv` and `data/processed/restaurant_yelp_subset.csv`

### Step 3 — EDA & Clustering
```bash
python 3_EDA_PCA_KMeans.py
```
- Generates 25 publication-quality EDA figures saved to `figures/`
- Runs PCA (first 5 components explain 81.9% of variance)
- Runs KMeans with K=3 (silhouette score = 0.293)
- Output: `data/processed/restaurant_clean_clustered.csv` + all figure files

### Step 4 — Feature Engineering + Logistic Regression
```bash
python 4_Feature_Engineering.py
```
- Applies strict leakage control (excludes `score`, `grade`, `violation_count`, etc.)
- Engineers new features: `poor_history_flag`, `complaint_intensity`, cyclic temporal encodings, `log_yelp_reviews`
- Performs time-based train/test split (pre-2024 = train, 2024+ = test)
- Trains and evaluates Logistic Regression baseline with threshold tuning
- Output: `models/logistic_model.pkl`, evaluation figures, metric CSVs

### Step 5 — Model Comparison & Final Selection
```bash
python 5_supervised_modeling.py
```
- Trains Logistic Regression, Random Forest, and Gradient Boosting
- Hyperparameters tuned via 3-fold stratified cross-validation (ROC-AUC)
- Evaluates all models; selects final model
- Output: `models/best_model.pkl`, comparison figures, metric CSVs

### Step 6 — Web Application
```bash
streamlit run app.py
```
- Opens interactive dashboard at `http://localhost:8501`
- Expects `data/processed/restaurant_clean.csv` and `models/best_model.pkl`
- Falls back to demo mode if model file is not found

---

## Dataset Description

### `restaurant_clean.csv` — Full cleaned dataset (41,556 rows × 52 columns)

| Column | Type | Description |
|--------|------|-------------|
| `camis` | str | Unique restaurant identifier |
| `inspection_date` | datetime | Date of inspection |
| `dba` | str | Restaurant name |
| `boro` | str | Borough (BRONX / BROOKLYN / MANHATTAN / QUEENS / STATEN ISLAND) |
| `zipcode` | str | ZIP code (zero-padded) |
| `cuisine` | str | Raw cuisine description |
| `score` | float | Inspection score (lower = better; ≥28 = failing) |
| `grade` | str | Letter grade (A / B / C) |
| `violation_count` | int | Number of violations at this inspection |
| `critical_count` | int | Number of critical violations |
| `failed` | int | **Target 1**: 1 if score ≥ 28 or grade = C, else 0 |
| `latitude` | float | Restaurant latitude (missing for 79.8% of records) |
| `longitude` | float | Restaurant longitude |
| `yelp_rating` | float | Yelp star rating (1.0–5.0) |
| `yelp_reviews` | float | Number of Yelp reviews |
| `yelp_price` | float | Price tier (1–3: $ / $$ / $$$) |
| `yelp_category` | str | Yelp cuisine tags (comma-separated) |
| `high_rating` | float | **Target 2**: 1 if yelp_rating ≥ 4.0, else 0 |
| `temp_mean` | float | Mean temperature on inspection day (°C) |
| `temp_max` | float | Max temperature (°C) |
| `temp_min` | float | Min temperature (°C) |
| `precipitation_sum` | float | Total precipitation (mm) |
| `rain_sum` | float | Total rainfall (mm) |
| `snowfall_sum` | float | Total snowfall (mm) |
| `wind_speed_mean` | float | Mean wind speed at 10m (km/h) |
| `wind_gust_mean` | float | Mean wind gusts (km/h) |
| `cloud_cover_mean` | float | Mean cloud cover (%) |
| `food_complaints_total` | float | Daily food-related 311 complaints in borough |
| `rodent_complaints` | float | Daily rodent 311 complaints in borough |
| `food_safety_complaints` | float | Daily food safety 311 complaints in borough |
| `median_household_income` | float | Borough median household income (USD) |
| `total_population` | float | Borough total population |
| `white_population` | float | Borough white-alone population |
| `inspection_year` | int | Year of inspection |
| `inspection_month` | int | Month of inspection (1–12) |
| `inspection_dow` | int | Day of week (0=Mon, 6=Sun) |
| `inspection_quarter` | int | Quarter (1–4) |
| `is_weekend` | int | 1 if inspection on Saturday or Sunday |
| `prev_score` | float | Score from previous inspection (−1 if first inspection) |
| `prev_failed` | float | Failed status of previous inspection (−1 if first) |
| `inspection_count` | int | Cumulative number of inspections for this restaurant |
| `score_trend` | float | Score change since last inspection (0 if first) |
| `grade_available` | int | 1 if grade was assigned at this inspection |
| `has_yelp` | int | 1 if Yelp data was successfully matched |
| `is_first_inspection` | int | 1 if this is the restaurant's first recorded inspection |
| `has_history` | int | 1 if a prior inspection record exists |
| `cuisine_grouped` | str | Top-20 cuisine types; others = "Other" |
| `yelp_category_primary` | str | First Yelp category tag; "unknown" if no Yelp data |
| `white_pct` | float | White population as fraction of borough total |
| `score_bucket` | str | Score category: A (≤13) / B (14–27) / C (≥28) / NA |
| `has_location` | int | 1 if latitude/longitude are available |

### `restaurant_yelp_subset.csv` — Yelp-matched subset (8,279 rows)

Same schema as above, filtered to `has_yelp == 1`. Used exclusively for modeling `high_rating`.

### `restaurant_clean_clustered.csv` — With cluster labels (41,556 rows × 54 columns)

Same as `restaurant_clean.csv` with two additional columns:

| Column | Description |
|--------|-------------|
| `cluster` | KMeans cluster label (0, 1, or 2) |
| `cluster_name` | Interpretable cluster name (see KMeans results below) |

---

## Key Results

### Model Performance (primary task: predicting `failed`)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|----|---------|
| **Logistic Regression** ✅ | 0.622 | 0.253 | **0.538** | **0.344** | **0.630** |
| Random Forest | 0.719 | 0.271 | 0.310 | 0.289 | 0.614 |
| Gradient Boosting | 0.815 | 0.200 | 0.000 | 0.000 | 0.616 |

> Evaluation at default threshold 0.50. Logistic Regression threshold tuned to **0.47** for deployment → Recall = **63.7%**, F1 = **0.354**.

**Final model**: Logistic Regression — best balance of recall, F1, ROC-AUC, and interpretability.

### KMeans Clustering Results (K=3, silhouette=0.293)

| Cluster | Name | Records | Failure Rate | Key Characteristics |
|---------|------|---------|-------------|---------------------|
| 0 | Repeat Operators With Prior Risk | 15,447 | 17.9% | High history rate; elevated prior scores |
| 1 | No-History Higher Risk | 17,943 | 20.1% | Nearly all first inspections; highest failure rate |
| 2 | Yelp-Matched Stable Compliant | 8,166 | 13.6% | Full Yelp coverage; lowest failure rate |

### Key Findings

- **Historical performance** is the strongest leakage-safe predictor of future inspection failure
- **Chinese, Korean, and Asian/Asian Fusion** cuisines show higher predicted failure risk
- **`has_history`** (prior inspection records) is associated with lower failure probability
- **Yelp rating** shows only weak correlation with inspection outcomes (consumer satisfaction ≠ regulatory compliance)
- **Summer months** and higher temperatures are modestly associated with elevated failure rates
- **PCA**: first 5 components explain 81.9% of variance; PC1 = compliance severity, PC2 = neighborhood context

---

## Web Application

The **NYC Restaurant Intelligence** Streamlit app provides five interactive pages:

| Page | Description |
|------|-------------|
| 📊 Overview | Summary statistics, borough failure rates, monthly inspection trends, cuisine failure rates |
| 🗺️ Map Explorer | Interactive PyDeck map with 4 color modes: inspection result, score bucket, borough, cuisine |
| 📈 EDA Insights | 5-tab interface: Univariate · Bivariate · Interactions · PCA/KMeans · Yelp & Neighborhood |
| 🤖 Prediction | Real-time failure probability estimation using the trained Logistic Regression model |
| ℹ️ About | Project summary, data sources, pipeline overview, model performance metrics |

**Global sidebar filters** (borough, year, cuisine type) propagate across all pages simultaneously.

The Prediction page accepts: borough, cuisine, number of past inspections, previous score, previous result, violation counts, temperature, precipitation, month, and weekend indicator.

---

## Leakage Control

The following variables are excluded from all supervised models to prevent target leakage:

| Excluded Variable | Reason |
|------------------|--------|
| `score` | Directly defines `failed`; including it would perfectly predict the target |
| `grade` | Assigned at inspection time; directly reflects the outcome |
| `score_bucket` | Derived from score |
| `violation_count` | Recorded during the current inspection |
| `critical_count` | Recorded during the current inspection |
| `score_trend` | Computed using the current score |
| `action` | Post-inspection administrative field |
| `cluster` | KMeans segment partially encodes inspection severity |

---

## Requirements

```
streamlit
pandas
numpy
plotly
pydeck
scikit-learn>=1.3.0
joblib
statsmodels
```

Python version: **3.11** (specified in `runtime.txt`)

---

## Data Citations

- NYC DOHMH Restaurant Inspection Results: https://data.cityofnewyork.us/resource/43nn-pn8j.json
- NYC 311 Service Requests: https://data.cityofnewyork.us/resource/erm2-nwe9.json
- Yelp Fusion API: https://docs.developer.yelp.com/docs/fusion-intro
- Open-Meteo Historical Weather API: https://open-meteo.com/en/docs/historical-weather-api
- U.S. Census Bureau ACS 5-Year Estimates 2022: https://api.census.gov/data/2022/acs/acs5
