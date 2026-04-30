"""
Project 4 Data Acquisition — NYC Restaurant Health & Rating Prediction
数据时间范围: 2023-01-01 ~ 2024-12-31

数据源:
  1. NYC餐厅卫生检查记录   — Socrata API      (NYC Open Data)
  2. NYC餐厅基本信息        — Socrata API      (NYC Open Data)
  3. Yelp评分/评论          — Yelp Fusion API
  4. 天气数据               — Open-Meteo API   (免费无需Key，同Project 1)
  5. 311食品相关投诉        — Socrata API      (同Project 1)
  6. 人口/收入              — Census API       (同Project 1)
  7. 餐厅违规类型详情        — BeautifulSoup爬虫 (NYC卫生局网页)

运行前请填入:
  YELP_API_KEY  — https://docs.developer.yelp.com/docs/fusion-intro 免费注册
"""


YELP_API_KEY = "sl86UwPKMLudf-5uICsT7gZTkMhSgwa-oGjLspEUzUnasr1E6r1cnW0eSzkJ7qTtXy27SnkM67WyT9DtZ5mxIZUIHY4Rgs-mQX1XXyWCBgsr1CNBfRD-SOR_gfPraXYx"
import requests
import pandas as pd
import numpy as np
import os
import time
from bs4 import BeautifulSoup

# ── 配置 ──────────────────────────────────────────────────────────────────────

RAW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'raw')
os.makedirs(RAW_DIR, exist_ok=True)
YELP_API_KEY = "sl86UwPKMLudf-5uICsT7gZTkMhSgwa-oGjLspEUzUnasr1E6r1cnW0eSzkJ7qTtXy27SnkM67WyT9DtZ5mxIZUIHY4Rgs-mQX1XXyWCBgsr1CNBfRD-SOR_gfPraXYx"
START_DATE   = '2023-01-01'
END_DATE     = '2024-12-31'
NYC_BOROUGHS = ['BRONX', 'BROOKLYN', 'MANHATTAN', 'QUEENS', 'STATEN ISLAND']


# =============================================================================
# 1. NYC餐厅卫生检查记录  (Socrata API)
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

# 每次检查聚合为一行（一次检查可能有多条violation记录）
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

# 目标变量1: 是否不合格（score >= 28 或 grade == C）
inspection_agg['failed'] = (
    (inspection_agg['grade'] == 'C') |
    (inspection_agg['score'] >= 28)
).astype(int)

inspection_agg.to_csv(os.path.join(RAW_DIR, 'inspections_agg.csv'), index=False)
print(f"   Saved: inspections_agg.csv  ({len(inspection_agg):,} unique inspections)")


# =============================================================================
# 2. NYC餐厅基本信息  (Socrata API，同一endpoint取静态字段)
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
# 3. Yelp评分/评论数据  (Yelp Fusion API)
#    支持断点续传：已拉取的结果缓存到yelp_cache.csv
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
to_fetch   = raw_restaurants[~raw_restaurants['camis'].astype(str).isin(done_camis)].head(3000)


total = len(to_fetch)
for idx, (_, row) in enumerate(to_fetch.iterrows(), 1):
    if idx % 50 == 0 or idx == 1:
        print(f"   [{idx}/{total}] {idx/total*100:.1f}% — {row['dba'][:30]}")
    try:
# for _, row in to_fetch.iterrows():
#     try:
        # params = {
        #     "term":     row['dba'],
        #     "location": f"{row.get('building','')} {row.get('street','')} {row.get('zipcode','')} New York",
        #     "limit":    1,
        # }
        building = str(row.get('building', '') or '')
        street   = str(row.get('street', '') or '')
        zipcode  = str(row.get('zipcode', '') or '')
        # 过滤掉nan字符串
        building = '' if building == 'nan' else building
        street   = '' if street == 'nan' else street
        zipcode  = '' if zipcode == 'nan' else zipcode

        params = {
            "term":     row['dba'][:64],   # Yelp term最长64字符
            "location": f"{building} {street} {zipcode} New York".strip(),
            "limit":    1,
        }
        
        
        r = requests.get(YELP_SEARCH_URL, headers=yelp_headers, params=params, timeout=15)
        if r.status_code == 429:
            print("   Rate limit, sleeping 60s...")
            time.sleep(60)
            continue
        r.raise_for_status()
        biz = r.json().get('businesses', [])
        if biz:
            b = biz[0]
            yelp_rows.append({
                'camis':        row['camis'],
                'yelp_id':      b.get('id'),
                'yelp_rating':  b.get('rating'),
                'yelp_reviews': b.get('review_count'),
                'yelp_price':   len(b.get('price', '') or ''),   # $→1 $$→2 $$$→3
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

# 目标变量2: Yelp高评分（>= 4.0）
raw_yelp['high_rating'] = (raw_yelp['yelp_rating'] >= 4.0).astype(int)
raw_yelp.to_csv(yelp_cache_path, index=False)
raw_yelp.to_csv(os.path.join(RAW_DIR, 'yelp_raw.csv'), index=False)
print(f"   Saved: yelp_raw.csv  ({len(raw_yelp):,} restaurants)")


# =============================================================================
# 4. 天气数据  (Open-Meteo，同Project 1，代码完全复用)
# =============================================================================
print("\n" + "=" * 60)
print("4. Fetching Weather Data (Open-Meteo, same as Project 1)...")

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
# 5. 311食品相关投诉  (Socrata API，同Project 1逻辑)
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
# 6. 人口/收入  (Census API，同Project 1)
# =============================================================================
print("\n" + "=" * 60)
print("6. Fetching Census Demographics (same as Project 1)...")

CENSUS_URL = "https://api.census.gov/data/2022/acs/acs5"
params = {
   "get": "B19013_001E,B01003_001E,B02001_002E,NAME",
    "for": "zip code tabulation area:*",
    "in":  "state:36",
}
r = requests.get(CENSUS_URL, params=params, timeout=60)
r.raise_for_status()
census_data = r.json()
raw_census  = pd.DataFrame(census_data[1:], columns=census_data[0])
raw_census.rename(columns={
    "B19013_001E": "median_household_income",
    "B01003_001E": "total_population",
    "B02001_002E": "white_population",
    "zip code tabulation area": "zipcode",
}, inplace=True)
for col in ["median_household_income", "total_population", "white_population"]:
    raw_census[col] = pd.to_numeric(raw_census[col], errors='coerce')
raw_census.replace(-666666666, np.nan, inplace=True)
raw_census.to_csv(os.path.join(RAW_DIR, 'census_demographics_raw.csv'), index=False)
print(f"   Saved: census_demographics_raw.csv  ({len(raw_census):,} ZCTAs)")


# =============================================================================
# 7. 违规类型详情  (BeautifulSoup 爬虫)
#    爬取NYC卫生局violation说明页，构建violation严重程度映射表
# =============================================================================
print("\n" + "=" * 60)
print("7. Scraping Violation Code Details (BeautifulSoup)...")

VIOLATION_URL = "https://www1.nyc.gov/site/doh/business/food-operators/violations.page"
headers = {'User-Agent': 'Mozilla/5.0 (Educational Research Project)'}

try:
    r = requests.get(VIOLATION_URL, headers=headers, timeout=30)
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
    print(f"   Scrape failed ({e}), using static severity mapping...")
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
# 8. 最终数据集构建
#    主键: camis × inspection_date
# =============================================================================
print("\n" + "=" * 60)
print("8. Building Final Dataset...")

df = inspection_agg.copy()

# 餐厅基本信息
df = df.merge(
    raw_restaurants[['camis', 'latitude', 'longitude']],
    on='camis', how='left'
)

# Yelp静态特征
df = df.merge(
    raw_yelp[['camis', 'yelp_rating', 'yelp_reviews', 'yelp_price', 'yelp_category', 'high_rating']],
    on='camis', how='left'
)

# 天气（检查当天）
df = df.merge(weather_daily, left_on='inspection_date', right_on='date', how='left')
df.drop(columns=['date'], inplace=True)

# 311投诉（检查当天 × 行政区）
complaints_daily['date'] = pd.to_datetime(complaints_daily['date'])
df['boro_key'] = df['boro'].str.upper()
df = df.merge(
    complaints_daily.rename(columns={'borough': 'boro_key', 'date': 'inspection_date'}),
    on=['inspection_date', 'boro_key'], how='left'
)
df.drop(columns=['boro_key'], inplace=True)

# Census（按zipcode）
df['zipcode'] = df['zipcode'].astype(str).str.zfill(5)
raw_census['zipcode'] = raw_census['zipcode'].astype(str).str.zfill(5)
df = df.merge(
    raw_census[['zipcode', 'median_household_income', 'total_population',
                'white_population', 'commute_time_mean']],
    on='zipcode', how='left'
)

# 时间特征
df['inspection_year']    = df['inspection_date'].dt.year
df['inspection_month']   = df['inspection_date'].dt.month
df['inspection_dow']     = df['inspection_date'].dt.dayofweek
df['inspection_quarter'] = df['inspection_date'].dt.quarter
df['is_weekend']         = (df['inspection_dow'] >= 5).astype(int)

# 餐厅历史特征（同一餐厅的时序信息）
df = df.sort_values(['camis', 'inspection_date'])
df['prev_score']       = df.groupby('camis')['score'].shift(1)
df['prev_failed']      = df.groupby('camis')['failed'].shift(1).fillna(-1)
df['inspection_count'] = df.groupby('camis').cumcount() + 1
df['score_trend']      = df['score'] - df['prev_score']

# 缺失值填充
for col in ['food_complaints_total', 'rodent_complaints', 'food_safety_complaints']:
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

print("\n" + "=" * 60)
print("DATA ACQUISITION COMPLETE")
print("=" * 60)
files = [
    'nyc_inspections_raw.csv',      # 原始检查记录（每条violation一行）
    'inspections_agg.csv',          # 聚合后（每次检查一行）
    'restaurants_master.csv',       # 餐厅基本信息
    'yelp_raw.csv',                 # Yelp评分
    'weather_raw.csv',              # 原始小时天气
    'weather_daily.csv',            # 日均天气
    'nyc_311_food_raw.csv',         # 311食品投诉原始
    'food_complaints_daily.csv',    # 311投诉日×行政区聚合
    'census_demographics_raw.csv',  # Census人口收入
    'violation_lookup.csv',         # 违规类型说明（爬虫）
    'nyc_restaurant_panel.csv',     # 最终面板（直接用这个）
]
for fname in files:
    fpath = os.path.join(RAW_DIR, fname)
    exists = os.path.exists(fpath)
    size   = os.path.getsize(fpath) / 1024 if exists else 0
    print(f"  {'✓' if exists else '✗'} {fname:<40} {size:>8.0f} KB")
