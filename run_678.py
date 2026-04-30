"""
只跑步骤 6 Census + 7 爬虫 + 8 面板合并
前提: ~/Downloads/raw/ 里已有:
  - inspections_agg.csv
  - restaurants_master.csv
  - yelp_raw.csv
  - weather_daily.csv
  - food_complaints_daily.csv
"""

import requests
import pandas as pd
import numpy as np
import os
from bs4 import BeautifulSoup

RAW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'raw')

# =============================================================================
# 6. Census (county级别，更稳定)
# =============================================================================
print("=" * 60)
print("6. Fetching Census Demographics (county level)...")

# NYC 5个borough对应的county FIPS代码
# 061=Manhattan 047=Brooklyn 081=Queens 005=Bronx 085=Staten Island
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
raw_census = pd.DataFrame(census_data[1:], columns=census_data[0])
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
# 7. 爬虫 violation codes
# =============================================================================
print("\n" + "=" * 60)
print("7. Scraping Violation Code Details...")

VIOLATION_URL = "https://www1.nyc.gov/site/doh/business/food-operators/violations.page"
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
        raise ValueError("No rows parsed")
    violation_lookup = pd.DataFrame(violation_rows).drop_duplicates(subset='violation_code')
    print(f"   Scraped {len(violation_lookup)} violation codes")
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
# 8. 面板合并
# =============================================================================
print("\n" + "=" * 60)
print("8. Building Final Dataset...")

inspection_agg   = pd.read_csv(os.path.join(RAW_DIR, 'inspections_agg.csv'),        parse_dates=['inspection_date'])
raw_restaurants  = pd.read_csv(os.path.join(RAW_DIR, 'restaurants_master.csv'))
# raw_yelp = pd.read_csv(os.path.join(RAW_DIR, 'yelp_cache.csv'))raw_yelp         = pd.read_csv(os.path.join(RAW_DIR, 'yelp_raw.csv'))
raw_yelp = pd.read_csv(os.path.join(RAW_DIR, 'yelp_cache.csv'))
weather_daily    = pd.read_csv(os.path.join(RAW_DIR, 'weather_daily.csv'),           parse_dates=['date'])
complaints_daily = pd.read_csv(os.path.join(RAW_DIR, 'food_complaints_daily.csv'),   parse_dates=['date'])

if 'high_rating' not in raw_yelp.columns:
    raw_yelp['high_rating'] = (raw_yelp['yelp_rating'] >= 4.0).astype(int)

df = inspection_agg.copy()

# 餐厅坐标
df = df.merge(raw_restaurants[['camis', 'latitude', 'longitude']], on='camis', how='left')

# Yelp
df = df.merge(
    raw_yelp[['camis', 'yelp_rating', 'yelp_reviews', 'yelp_price', 'yelp_category', 'high_rating']],
    on='camis', how='left'
)

# 天气
df = df.merge(weather_daily, left_on='inspection_date', right_on='date', how='left')
df.drop(columns=['date'], inplace=True)

# 311投诉
df['boro_key'] = df['boro'].str.upper()
complaints_daily['borough'] = complaints_daily['borough'].str.upper()
df = df.merge(
    complaints_daily.rename(columns={'borough': 'boro_key', 'date': 'inspection_date'}),
    on=['inspection_date', 'boro_key'], how='left'
)
df.drop(columns=['boro_key'], inplace=True)

# Census（按borough合并）
raw_census['boro'] = raw_census['boro'].str.upper()
df['boro'] = df['boro'].str.upper()
df = df.merge(
    raw_census[['boro', 'median_household_income', 'total_population', 'white_population']],
    on='boro', how='left'
)

# 时间特征
df['inspection_year']    = df['inspection_date'].dt.year
df['inspection_month']   = df['inspection_date'].dt.month
df['inspection_dow']     = df['inspection_date'].dt.dayofweek
df['inspection_quarter'] = df['inspection_date'].dt.quarter
df['is_weekend']         = (df['inspection_dow'] >= 5).astype(int)

# 历史检查特征
df = df.sort_values(['camis', 'inspection_date'])
df['prev_score']       = df.groupby('camis')['score'].shift(1)
df['prev_failed']      = df.groupby('camis')['failed'].shift(1).fillna(-1)
df['inspection_count'] = df.groupby('camis').cumcount() + 1
df['score_trend']      = df['score'] - df['prev_score']

# 缺失值填充
for col in ['food_complaints_total', 'rodent_complaints', 'food_safety_complaints']:
    if col in df.columns:
        df[col] = df[col].fillna(0)

# 保存
out_path = os.path.join(RAW_DIR, 'nyc_restaurant_panel.csv')
df.to_csv(out_path, index=False)

print(f"\n   Final dataset: {df.shape[0]:,} rows x {df.shape[1]} columns")
print(f"   Date range:    {df['inspection_date'].min().date()} to {df['inspection_date'].max().date()}")
print(f"   Unique restaurants: {df['camis'].nunique():,}")
print(f"   Target 1 (failed) rate:      {df['failed'].mean():.1%}")
print(f"   Target 2 (high_rating) rate: {df['high_rating'].mean():.1%}")
print(f"\n   Saved: {out_path}")
print("\nDONE")