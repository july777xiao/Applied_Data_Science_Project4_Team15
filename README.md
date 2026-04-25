# Project 4: Predicting Restaurant Inspection Failures in New York City
## A Multi-Source Machine Learning Approach

> Columbia University · GU4243/GR5243 Applied Data Science · Spring 2026 · Team 15
>
> Shuzhi Yang (sy3321) · Pingyu Zhou (pz2341) · Junyang Li (jl7230) · Xiao Xiao (xx2492)

---

## Project Overview

This project builds an end-to-end machine learning pipeline to predict two outcomes for NYC restaurants:

- **Primary task**: Will a restaurant **fail** its next health inspection? (binary classification, `failed = 1` if score ≥ 28)
- **Secondary task**: Does a restaurant have a **high Yelp rating** (≥ 4.0 stars)? (binary classification on Yelp-matched subset)

The dataset integrates six sources — regulatory inspection records, Yelp consumer data, weather, 311 complaints, Census demographics, and violation code references — collected via Socrata API, Yelp Fusion API, Open-Meteo API, Census Bureau API, and web scraping.

---

## Repository Structure

```
project4/
├── README.md
├── requirements.txt
│
├── 1_data_acquisition.py        # API calls, web scraping, panel construction
├── 2_data_cleaning.py           # Full cleaning pipeline
├── 3_eda_clustering.py          # EDA, PCA, KMeans (Person B)
├── 4_feature_engineering.py     # Feature encoding and construction (Person C)
├── 5_modeling.py                # Model training and evaluation (Person C + D)
│
├── app.py                       # Streamlit web application
│
├── raw/
│   ├── nyc_restaurant_panel.csv         # Raw merged panel (before cleaning)
│   ├── restaurant_clean.csv             # Final cleaned dataset (41,556 rows × 52 cols)
│   └── restaurant_yelp_subset.csv       # Yelp-matched subset (8,279 rows)
│
└── models/
    └── best_model.pkl                   # Final trained model (for web app)
```

---

## Data Sources

| Source | Method | Records | Key Variables |
|--------|--------|---------|---------------|
| NYC DOHMH Inspections | Socrata API | 142,591 raw → 41,771 aggregated | Score, grade, violation codes |
| Restaurant Coordinates | Socrata API | 4,487 with coordinates | Latitude, longitude |
| Yelp Business Data | Yelp Fusion API | ~3,000 restaurants matched | Rating, reviews, price tier |
| Weather (Open-Meteo) | REST API (free, no key) | 731 daily records | Temperature, precipitation |
| NYC 311 Complaints | Socrata API | Food/sanitation complaints | Daily borough-level counts |
| U.S. Census ACS 2022 | Census Bureau API | 5 NYC boroughs | Income, population |
| Violation Code Reference | Web scraping (fallback) | 8 common codes | Severity, category |

---

## Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/[your-org]/Applied_Data_Science_Project4_Team15.git
cd Applied_Data_Science_Project4_Team15
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install pandas numpy requests beautifulsoup4 scikit-learn xgboost \
            matplotlib seaborn plotly streamlit pydeck
```

### 3. API Keys Required

Only **Yelp Fusion API** requires a key. All other APIs are free and keyless.

1. Register for a free Yelp API key at: https://docs.developer.yelp.com/docs/fusion-intro
2. Open `1_data_acquisition.py` and replace line 16:
   ```python
   YELP_API_KEY = "YOUR_YELP_API_KEY"
   ```

---

## How to Run

### Step 1 — Data Acquisition
```bash
python 1_data_acquisition.py
```
- Fetches all 6 data sources
- Saves raw files to `raw/`
- **Note**: The Yelp step takes ~1 hour (rate-limited at 0.25s/request). Supports interrupted runs via cache file.
- Output: `raw/nyc_restaurant_panel.csv`

### Step 2 — Data Cleaning
```bash
python 2_data_cleaning.py
```
- Runs full cleaning pipeline (type conversion, missing value handling, feature engineering)
- Output: `raw/restaurant_clean.csv` and `raw/restaurant_yelp_subset.csv`

### Step 3 — EDA & Clustering
```bash
python 3_eda_clustering.py
```
- Generates EDA visualizations
- Runs PCA and KMeans clustering
- Adds cluster labels to the dataset
- Output: `raw/restaurant_clean_clustered.csv` + figure files

### Step 4 — Feature Engineering
```bash
python 4_feature_engineering.py
```
- Encodes categorical features, engineers interaction terms
- Performs time-based train/test split (2023 = train, 2024 = test)
- Output: processed feature matrices

### Step 5 — Modeling
```bash
python 5_modeling.py
```
- Trains Logistic Regression, Random Forest, XGBoost
- Evaluates with AUC, F1, Precision, Recall, calibration curve
- Saves final model to `models/best_model.pkl`

### Step 6 — Web Application
```bash
streamlit run app.py
```
- Opens interactive dashboard at `http://localhost:8501`
- Pages: Overview · Map Explorer · EDA Insights · Prediction · About

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
| `yelp_price` | float | Price tier (1–4: $–$$$$) |
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
| `prev_score` | float | Score from previous inspection (-1 if first inspection) |
| `prev_failed` | float | Failed status of previous inspection (-1 if first) |
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

### `restaurant_yelp_subset.csv` — Yelp-matched subset (8,279 rows × 52 columns)

Same schema as above, filtered to `has_yelp == 1`. Used for modeling `high_rating`.

---

## Key Results

| Model | AUC | F1 | Notes |
|-------|-----|----|-------|
| Logistic Regression | TBD | TBD | Baseline |
| Random Forest | TBD | TBD | — |
| XGBoost | TBD | TBD | Final model |

> Results will be updated after modeling is complete.

---

## Web Application

The Streamlit dashboard is deployed at: **[link to be added after deployment]**

To run locally:
```bash
streamlit run app.py
```

Pages:
- **Overview**: KPI metrics, borough comparison, cuisine failure rates
- **Map Explorer**: Interactive NYC map with inspection outcomes
- **EDA Insights**: Weather, temporal, Yelp, and neighborhood analysis
- **Prediction**: Input restaurant features → predicted failure probability
- **About**: Project description, team contributions, data sources

---

## Team Contributions

| Member | Role | Responsibilities |
|--------|------|-----------------|
| Xiao Xiao (xx2492) | Data Acquisition + Web App | Data pipeline (`1_data_acquisition.py`, `2_data_cleaning.py`), Streamlit app (`app.py`) |
| [Member B] | EDA + Unsupervised Learning | `3_eda_clustering.py`, EDA visualizations, PCA, KMeans clustering |
| [Member C] | Feature Engineering + Baseline Model | `4_feature_engineering.py`, Logistic Regression, feature selection |
| [Member D] | Modeling + Final Model Selection | `5_modeling.py`, Random Forest, XGBoost, model comparison, final model |

---

## Requirements

```
pandas>=2.0
numpy>=1.24
requests>=2.28
beautifulsoup4>=4.12
scikit-learn>=1.3
xgboost>=2.0
matplotlib>=3.7
seaborn>=0.12
plotly>=5.15
streamlit>=1.28
pydeck>=0.8
shap>=0.42
```

---

## Citation & Data Sources

- NYC DOHMH Restaurant Inspection Results: https://data.cityofnewyork.us/resource/43nn-pn8j.json
- NYC 311 Service Requests: https://data.cityofnewyork.us/resource/erm2-nwe9.json
- Yelp Fusion API: https://docs.developer.yelp.com/docs/fusion-intro
- Open-Meteo Historical Weather API: https://open-meteo.com
- U.S. Census Bureau ACS 2022: https://api.census.gov/data/2022/acs/acs5
