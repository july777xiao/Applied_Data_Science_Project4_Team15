"""
1_data_acquisition.py
Project 4 Data Acquisition — NYC Restaurant Health & Rating Prediction
Time range: 2023-01-01 ~ 2024-12-31

Data sources:
  1. NYC Restaurant Inspection Records  — Socrata API     (NYC Open Data)
  2. NYC Restaurant Master List         — Socrata API     (NYC Open Data)
  3. Yelp Business Data                 — Yelp Fusion API (requires free API key)
  4. Weather Data                       — Open-Meteo API  (free, no key required)
  5. NYC 311 Food Complaints            — Socrata API     (NYC Open Data)
  6. U.S. Census Demographics           — Census API      (county level, free)
  7. Violation Code Reference           — BeautifulSoup   (NYC DOH, with fallback)
  8. Panel Construction                 — merge all sources

Before running, set your Yelp API key below.
Register at: https://docs.developer.yelp.com/docs/fusion-intro
"""

import requests
import pandas as pd
import numpy as np
import os
import time
from bs4 import BeautifulSoup

# ── Configuration ─────────────────────────────────────────────────────────────
RAW_DIR      = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'raw')
os.makedirs(RAW_DIR, exist_ok=True)

YELP_API_KEY = "YOUR_YELP_API_KEY"   # ← replace with your key
START_DATE   = '2023-01-01'
END_DATE     = '2024-12-31'
NYC_BOROUGHS = ['BRONX', 'BROOKLYN', 'MANHATTAN', 'QUEENS', 'STATEN ISLAND']


# =============================================================================
# 1. NYC Restaurant Inspection Records  (Socrata API)
#    endpoint: https://data.cityofnewyork.us/resource/43nn-pn8j.json
# =============================================================================
print("=" * 60)
print("1. Fetching NYC Restaurant Inspection Records (Socrata API)...")

INSPECTION_URL = "https://data.cityofnewyork.us/resource/43nn-pn8j.json"
LIMIT  = 50000
offset = 0
records = []

while True:
    params = {
        "$where":  f"inspection_date >= '{START_DATE}T00:00:00' AND inspection_date <= '{END_DATE}T23:59:59'",
        "$select": "camis,dba,boro,zipcode,cuisine_description,inspection_date,"
                   "action,violation_code,violation_description,critical_flag,score,grade",
        "$limit":  LIMIT,
        "$offset": offset,
        "$order":  "inspection_date ASC",
    }
    r = requests.get(INSPECTION_URL, params=params, timeout=60)
    r.raise_for_status()
    batch = r.json()
    if not batch:
        break
    records.extend(batch)
    offset += LIMIT
    print(f"   Fetched {len(records):,} inspection records...")
    time.sleep(0.3)

raw_inspections = pd.DataFrame(records)
raw_inspections['inspection_date'] = pd.to_datetime(raw_inspections['inspection_date'])
raw_inspections['score'] = pd.to_numeric(raw_inspections['score'], errors='coerce')
raw_inspections['boro']  = raw_inspections['boro'].str.upper()
raw_inspections.to_csv(os.path.join(RAW_DIR, 'nyc_inspections_raw.csv'), index=False)
print(f"   Saved: nyc_inspections_raw.csv  ({len(raw_inspections):,} rows)")

# Aggregate to one row per inspection (one inspection can have multiple violation rows)
inspection_agg = raw_inspections.groupby(['camis', 'inspection_date']).agg(
    dba=('dba', 'first'),
    boro=('boro', 'first'),
    zipcode=('zipcode', 'first'),
    cuisine=('cuisine_description', 'first'),
    score=('score', 'first'),
    grade=('grade', 'first'),
    action=('action', 'first'),
    violation_count=('violation_code', 'count'),
    critical_count=('critical_flag', lambda x: (x == 'Critical').sum()),
).reset_index()

# Target variable 1: failed = 1 if score >= 28 or grade == 'C'
inspection_agg['failed'] = (
    (inspection_agg['grade'] == 'C') |
    (inspection_agg['score'] >= 28)
).astype(int)

inspection_agg.to_csv(os.path.join(RAW_DIR, 'inspections_agg.csv'), index=False)
print(f"   Saved: inspections_agg.csv  ({len(inspection_agg):,} unique inspections)")


# =============================================================================
# 2. NYC Restaurant Master List  (Socrata API, same endpoint)
# =============================================================================
print("\n" + "=" * 60)
print("2. Fetching NYC Restaurant Master List (Socrata API)...")

params = {
    "$select": "camis,dba,boro,building,street,zipcode,phone,cuisine_description,latitude,longitude",
    "$limit":  50000,
    "$offset": 0,
    "$order":  "camis ASC",
    "$where":  "latitude IS NOT NULL",
}
r = requests.get(INSPECTION_URL, params=params, timeout=60)
r.raise_for_status()
raw_restaurants = pd.DataFrame(r.json()).drop_duplicates(subset='camis')
raw_restaurants['latitude']  = pd.to_numeric(raw_restaurants['latitude'],  errors='coerce')
raw_restaurants['longitude'] = pd.to_numeric(raw_restaurants['longitude'], errors='coerce')
raw_restaurants['boro'] = raw_restaurants['boro'].str.upper()
raw_restaurants.to_csv(os.path.join(RAW_DIR, 'restaurants_master.csv'), index=False)
print(f"   Saved: restaurants_master.csv  ({len(raw_restaurants):,} restaurants)")


# =============================================================================
# 3. Yelp Business Data  (Yelp Fusion API)
#    Supports resume: already-fetched results cached to yelp_cache.csv
# =============================================================================
print("\n" + "=" * 60)
print("3. Fetching Yelp Business Data (Yelp Fusion API)...")

YELP_SEARCH_URL = "https://api.yelp.com/v3/businesses/search"
yelp_headers    = {"Authorization": f"Bearer {YELP_API_KEY}"}
yelp_cache_path = os.path.join(RAW_DIR, 'yelp_cache.csv')

if os.path.exists(yelp_cache_path):
    yelp_cache = pd.read_csv(yelp_cache_path)
    done_camis = set(yelp_cache['camis'].astype(str))
    print(f"   Cache found: {len(done_camis)} already fetched")
else:
    yelp_cache = pd.DataFrame()
    done_camis = set()

yelp_rows = []
to_fetch  = raw_restaurants[~raw_restaurants['camis'].astype(str).isin(done_camis)]

total = len(to_fetch)
for idx, (_, row) in enumerate(to_fetch.iterrows(), 1):
    if idx % 50 == 0 or idx == 1:
        print(f"   [{idx}/{total}] {idx/total*100:.1f}% — {row['dba'][:30]}")
    try:
        building = str(row.get('building', '') or '')
        street   = str(row.get('street', '') or '')
        zipcode  = str(row.get('zipcode', '') or '')
        building = '' if building == 'nan' else building
        street   = '' if street   == 'nan' else street
        zipcode  = '' if zipcode  == 'nan' else zipcode

        params = {
            "term":     row['dba'][:64],
            "location": f"{building} {street} {zipcode} New York".strip(),
            "limit":    1,
        }
        r = requests.get(YELP_SEARCH_URL, headers=yelp_headers, params=params, timeout=15)
        if r.status_code == 429:
            print("   Rate limit hit, sleeping 60s...")
            time.sleep(60)
            continue
        r.raise_for_status()
        biz = r.json().get('businesses', [])
        if biz:
            b = biz[0]
            yelp_rows.append({
                'camis':         row['camis'],
                'yelp_id':       b.get('id'),
                'yelp_rating':   b.get('rating'),
                'yelp_reviews':  b.get('review_count'),
                'yelp_price':    len(b.get('price', '') or ''),
                'yelp_category': ','.join([c['alias'] for c in b.get('categories', [])]),
            })
        else:
            yelp_rows.append({'camis': row['camis'], 'yelp_id': None,
                              'yelp_rating': np.nan, 'yelp_reviews': np.nan,
                              'yelp_price': np.nan, 'yelp_category': None})
        time.sleep(0.25)
    except Exception as e:
        print(f"   Warning: {row['dba']} — {e}")

raw_yelp = pd.DataFrame(yelp_rows)
if not yelp_cache.empty:
    raw_yelp = pd.concat([yelp_cache, raw_yelp], ignore_index=True)

raw_yelp['high_rating'] = (raw_yelp['yelp_rating'] >= 4.0).astype(int)
raw_yelp.to_csv(yelp_cache_path, index=False)
raw_yelp.to_csv(os.path.join(RAW_DIR, 'yelp_raw.csv'), index=False)
print(f"   Saved: yelp_raw.csv  ({len(raw_yelp):,} restaurants)")


# =============================================================================
# 4. Weather Data  (Open-Meteo, free, no API key required)
# =============================================================================
print("\n" + "=" * 60)
print("4. Fetching Weather Data (Open-Meteo)...")

WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude":   40.7128,
    "longitude": -74.0060,
    "start_date": START_DATE,
    "end_date":   END_DATE,
    "hourly": ",".join([
        "temperature_2m", "precipitation", "rain", "snowfall",
        "wind_speed_10m", "wind_gusts_10m", "cloud_cover"
    ]),
    "timezone": "America/New_York",
}
r = requests.get(WEATHER_URL, params=params, timeout=60)
r.raise_for_status()
raw_weather = pd.DataFrame(r.json()["hourly"])
raw_weather.rename(columns={"time": "timestamp_local"}, inplace=True)
raw_weather['timestamp_local'] = pd.to_datetime(raw_weather['timestamp_local'])
raw_weather['date'] = raw_weather['timestamp_local'].dt.normalize()

weather_daily = raw_weather.groupby('date').agg(
    temp_mean=('temperature_2m', 'mean'),
    temp_max=('temperature_2m', 'max'),
    temp_min=('temperature_2m', 'min'),
    precipitation_sum=('precipitation', 'sum'),
    rain_sum=('rain', 'sum'),
    snowfall_sum=('snowfall', 'sum'),
    wind_speed_mean=('wind_speed_10m', 'mean'),
    wind_gust_mean=('wind_gusts_10m', 'mean'),
    cloud_cover_mean=('cloud_cover', 'mean'),
).reset_index()

raw_weather.to_csv(os.path.join(RAW_DIR, 'weather_raw.csv'), index=False)
weather_daily.to_csv(os.path.join(RAW_DIR, 'weather_daily.csv'), index=False)
print(f"   Saved: weather_daily.csv  ({len(weather_daily):,} days)")


# =============================================================================
# 5. NYC 311 Food-related Complaints  (Socrata API)
# =============================================================================
print("\n" + "=" * 60)
print("5. Fetching 311 Food-related Complaints (Socrata API)...")

FOOD_TYPES = [
    "Food Establishment", "Food Poisoning", "Unsanitary Condition",
    "Food Safety", "Rodents", "Cockroaches",
]
where_clause = (
    "(" + " OR ".join([f"complaint_type='{t}'" for t in FOOD_TYPES]) + ")"
    + f" AND created_date >= '{START_DATE}T00:00:00'"
    + f" AND created_date <= '{END_DATE}T23:59:59'"
)
params = {
    "$where":  where_clause,
    "$select": "unique_key,created_date,complaint_type,borough,incident_zip",
    "$limit":  50000,
    "$order":  "created_date ASC",
}
r = requests.get("https://data.cityofnewyork.us/resource/erm2-nwe9.json",
                 params=params, timeout=60)
r.raise_for_status()
raw_311 = pd.DataFrame(r.json())
raw_311['created_date'] = pd.to_datetime(raw_311['created_date'])
raw_311['date']    = raw_311['created_date'].dt.normalize()
raw_311['borough'] = raw_311['borough'].str.upper()
raw_311 = raw_311[raw_311['borough'].isin(NYC_BOROUGHS)].copy()
raw_311.to_csv(os.path.join(RAW_DIR, 'nyc_311_food_raw.csv'), index=False)

complaints_daily = raw_311.groupby(['date', 'borough']).agg(
    food_complaints_total=('unique_key', 'count'),
    rodent_complaints=('complaint_type', lambda x: (x == 'Rodents').sum()),
    food_safety_complaints=('complaint_type', lambda x: (x == 'Food Safety').sum()),
).reset_index()
complaints_daily.to_csv(os.path.join(RAW_DIR, 'food_complaints_daily.csv'), index=False)
print(f"   Saved: food_complaints_daily.csv  ({len(complaints_daily):,} borough-day rows)")


# =============================================================================
# 6. U.S. Census Demographics  (Census Bureau API, county level)
#    NYC county FIPS: 061=Manhattan 047=Brooklyn 081=Queens 005=Bronx 085=Staten Island
#    Note: querying at county level (not ZCTA) to avoid HTTP 400 errors
# =============================================================================
print("\n" + "=" * 60)
print("6. Fetching Census Demographics (county level)...")

COUNTY_FIPS = '061,047,081,005,085'
r = requests.get(
    'https://api.census.gov/data/2022/acs/acs5',
    params={
        "get": "B19013_001E,B01003_001E,B02001_002E,NAME",
        "for": f"county:{COUNTY_FIPS}",
        "in":  "state:36",
    },
    timeout=60
)
r.raise_for_status()
census_data = r.json()
raw_census  = pd.DataFrame(census_data[1:], columns=census_data[0])
raw_census.rename(columns={
    "B19013_001E": "median_household_income",
    "B01003_001E": "total_population",
    "B02001_002E": "white_population",
}, inplace=True)

county_to_boro = {
    '061': 'MANHATTAN',
    '047': 'BROOKLYN',
    '081': 'QUEENS',
    '005': 'BRONX',
    '085': 'STATEN ISLAND',
}
raw_census['boro'] = raw_census['county'].map(county_to_boro)
for col in ["median_household_income", "total_population", "white_population"]:
    raw_census[col] = pd.to_numeric(raw_census[col], errors='coerce')
raw_census.replace(-666666666, np.nan, inplace=True)
raw_census.to_csv(os.path.join(RAW_DIR, 'census_demographics_raw.csv'), index=False)
print(f"   Saved: census_demographics_raw.csv  ({len(raw_census)} boroughs)")


# =============================================================================
# 7. Violation Code Reference  (BeautifulSoup, with static fallback)
# =============================================================================
print("\n" + "=" * 60)
print("7. Scraping Violation Code Details (BeautifulSoup)...")

VIOLATION_URL  = "https://www1.nyc.gov/site/doh/business/food-operators/violations.page"
scrape_headers = {'User-Agent': 'Mozilla/5.0 (Educational Research Project)'}

try:
    r = requests.get(VIOLATION_URL, headers=scrape_headers, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, 'html.parser')
    violation_rows = []
    for table in soup.find_all('table'):
        for row in table.find_all('tr')[1:]:
            cells = row.find_all(['td', 'th'])
            if len(cells) >= 2:
                code = cells[0].get_text(strip=True)
                desc = cells[1].get_text(strip=True)
                if code and desc:
                    violation_rows.append({'violation_code': code, 'violation_desc_official': desc})
    if not violation_rows:
        raise ValueError("No table rows parsed")
    violation_lookup = pd.DataFrame(violation_rows).drop_duplicates(subset='violation_code')
    print(f"   Scraped {len(violation_lookup)} violation codes from NYC DOH")
except Exception as e:
    print(f"   Scrape failed ({e}), using static fallback...")
    violation_lookup = pd.DataFrame([
        {'violation_code': '04L', 'severity': 'critical',     'category': 'food_temperature'},
        {'violation_code': '04N', 'severity': 'critical',     'category': 'food_contamination'},
        {'violation_code': '06D', 'severity': 'critical',     'category': 'personal_hygiene'},
        {'violation_code': '08A', 'severity': 'not_critical', 'category': 'vermin'},
        {'violation_code': '10F', 'severity': 'not_critical', 'category': 'facility'},
        {'violation_code': '10B', 'severity': 'not_critical', 'category': 'facility'},
        {'violation_code': '02B', 'severity': 'critical',     'category': 'food_source'},
        {'violation_code': '09C', 'severity': 'not_critical', 'category': 'other'},
    ])

violation_lookup.to_csv(os.path.join(RAW_DIR, 'violation_lookup.csv'), index=False)
print(f"   Saved: violation_lookup.csv  ({len(violation_lookup):,} codes)")


# =============================================================================
# 8. Panel Construction  (merge all sources)
#    Primary key: camis × inspection_date
# =============================================================================
print("\n" + "=" * 60)
print("8. Building Final Dataset...")

df = inspection_agg.copy()

# Restaurant coordinates
df = df.merge(
    raw_restaurants[['camis', 'latitude', 'longitude']],
    on='camis', how='left'
)

# Yelp features (restaurant-level static)
df = df.merge(
    raw_yelp[['camis', 'yelp_rating', 'yelp_reviews', 'yelp_price', 'yelp_category', 'high_rating']],
    on='camis', how='left'
)

# Weather (inspection day)
df = df.merge(weather_daily, left_on='inspection_date', right_on='date', how='left')
df.drop(columns=['date'], inplace=True)

# 311 complaints (inspection day × borough)
complaints_daily['date'] = pd.to_datetime(complaints_daily['date'])
df['boro_key'] = df['boro'].str.upper()
complaints_daily['borough'] = complaints_daily['borough'].str.upper()
df = df.merge(
    complaints_daily.rename(columns={'borough': 'boro_key', 'date': 'inspection_date'}),
    on=['inspection_date', 'boro_key'], how='left'
)
df.drop(columns=['boro_key'], inplace=True)

# Census (borough level)
raw_census['boro'] = raw_census['boro'].str.upper()
df['boro'] = df['boro'].str.upper()
df = df.merge(
    raw_census[['boro', 'median_household_income', 'total_population', 'white_population']],
    on='boro', how='left'
)

# Temporal features
df['inspection_year']    = df['inspection_date'].dt.year
df['inspection_month']   = df['inspection_date'].dt.month
df['inspection_dow']     = df['inspection_date'].dt.dayofweek
df['inspection_quarter'] = df['inspection_date'].dt.quarter
df['is_weekend']         = (df['inspection_dow'] >= 5).astype(int)

# Historical inspection features (per restaurant)
df = df.sort_values(['camis', 'inspection_date'])
df['prev_score']       = df.groupby('camis')['score'].shift(1)
df['prev_failed']      = df.groupby('camis')['failed'].shift(1).fillna(-1)
df['inspection_count'] = df.groupby('camis').cumcount() + 1
df['score_trend']      = df['score'] - df['prev_score']

# Fill complaint NaN with 0 (no complaints on that day = 0)
for col in ['food_complaints_total', 'rodent_complaints', 'food_safety_complaints']:
    if col in df.columns:
        df[col] = df[col].fillna(0)

# Save
out_path = os.path.join(RAW_DIR, 'nyc_restaurant_panel.csv')
df.to_csv(out_path, index=False)

print(f"\n   Final dataset: {df.shape[0]:,} rows x {df.shape[1]} columns")
print(f"   Date range:    {df['inspection_date'].min().date()} to {df['inspection_date'].max().date()}")
print(f"   Unique restaurants: {df['camis'].nunique():,}")
print(f"   Target 1 (failed) rate:      {df['failed'].mean():.1%}")
print(f"   Target 2 (high_rating) rate: {df['high_rating'].mean():.1%}")
print(f"\n   Saved: {out_path}")

print("\n" + "=" * 60)
print("DATA ACQUISITION COMPLETE")
print("=" * 60)
files = [
    'nyc_inspections_raw.csv',      # Raw inspection records (one row per violation)
    'inspections_agg.csv',          # Aggregated (one row per inspection)
    'restaurants_master.csv',       # Restaurant coordinates
    'yelp_cache.csv',               # Yelp cache (resume support)
    'yelp_raw.csv',                 # Yelp ratings
    'weather_raw.csv',              # Hourly weather
    'weather_daily.csv',            # Daily weather
    'nyc_311_food_raw.csv',         # 311 complaints raw
    'food_complaints_daily.csv',    # 311 complaints aggregated
    'census_demographics_raw.csv',  # Census demographics
    'violation_lookup.csv',         # Violation code reference
    'nyc_restaurant_panel.csv',     # Final merged panel
]
for fname in files:
    fpath  = os.path.join(RAW_DIR, fname)
    exists = os.path.exists(fpath)
    size   = os.path.getsize(fpath) / 1024 if exists else 0
    print(f"  {'✓' if exists else '✗'} {fname:<42} {size:>8.0f} KB")
