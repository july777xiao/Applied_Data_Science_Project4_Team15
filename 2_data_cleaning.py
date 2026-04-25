"""
2_data_cleaning.py
Data Cleaning Script — NYC Restaurant Health & Rating Panel
输入: project4/raw/nyc_restaurant_panel.csv
输出:
  project4/raw/restaurant_clean.csv        全量清洗数据（用于failed预测）
  project4/raw/restaurant_yelp_subset.csv  Yelp子集（用于high_rating预测）
"""

import pandas as pd
import numpy as np
import os

RAW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'raw')
df = pd.read_csv(os.path.join(RAW_DIR, 'nyc_restaurant_panel.csv'), low_memory=False)

print("=" * 60)
print("Original shape:", df.shape)
print("=" * 60)

# ─────────────────────────────────────────────────────────────
# 1. 类型转换
# ─────────────────────────────────────────────────────────────
print("\n1. Type conversion...")

df['inspection_date'] = pd.to_datetime(df['inspection_date'])
df['camis']   = df['camis'].astype(str)
df['zipcode'] = df['zipcode'].astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(5)
df['zipcode'] = df['zipcode'].replace('nan', np.nan)

print("   inspection_date → datetime")
print("   camis → string")
print("   zipcode → zero-padded string")

# ─────────────────────────────────────────────────────────────
# 2. 删除 borough 异常值
# ─────────────────────────────────────────────────────────────
print("\n2. Removing invalid borough values...")

valid_boroughs = ['BRONX', 'BROOKLYN', 'MANHATTAN', 'QUEENS', 'STATEN ISLAND']
before = len(df)
df = df[df['boro'].isin(valid_boroughs)].copy()
print(f"   Removed {before - len(df)} rows with boro='0'")

# ─────────────────────────────────────────────────────────────
# 3. score 缺失处理
#    score为空 且 grade也为空 且 violation_count==0 → 无实质检查结果，删除
#    其余score缺失（有violation记录）→ 保留，建模时排除
# ─────────────────────────────────────────────────────────────
print("\n3. Handling missing score...")

no_result_mask = (
    df['score'].isna() &
    df['grade'].isna() &
    (df['violation_count'] == 0)
)
before = len(df)
df = df[~no_result_mask].copy()
print(f"   Removed {before - len(df)} rows with no inspection result")
print(f"   Remaining score NaN: {df['score'].isna().sum()} (kept, has violation info)")

# ─────────────────────────────────────────────────────────────
# 4. grade 缺失 → 新增标记列
# ─────────────────────────────────────────────────────────────
print("\n4. Encoding grade availability...")

df['grade_available'] = df['grade'].notna().astype(int)
print(f"   grade_available=1: {df['grade_available'].sum():,} rows")
print(f"   grade_available=0: {(df['grade_available']==0).sum():,} rows")

# ─────────────────────────────────────────────────────────────
# 5. latitude/longitude
#    不删，不填，建模不用，可视化时按需过滤
#    新增 has_location 标记列
# ─────────────────────────────────────────────────────────────
print("\n5. Handling lat/lon...")

df['has_location'] = df['latitude'].notna().astype(int)
print(f"   has_location=1: {df['has_location'].sum():,} rows ({df['has_location'].mean():.1%})")
print(f"   has_location=0: {(df['has_location']==0).sum():,} rows (kept, not used in modeling)")

# ─────────────────────────────────────────────────────────────
# 6. Yelp 缺失处理
#    - 新增 has_yelp 标记列
#    - yelp_price == 0 → NaN（无价格信息）
#    - 其余Yelp字段保留NaN，不填充
# ─────────────────────────────────────────────────────────────
print("\n6. Handling Yelp missing values...")

df['has_yelp'] = df['yelp_rating'].notna().astype(int)
df.loc[df['yelp_price'] == 0, 'yelp_price'] = np.nan
print(f"   has_yelp=1: {df['has_yelp'].sum():,} rows ({df['has_yelp'].mean():.1%})")
print(f"   has_yelp=0: {(df['has_yelp']==0).sum():,} rows")

# ─────────────────────────────────────────────────────────────
# 7. Census 缺失
#    borough='0'已删，剩余Census缺失用borough中位数填充
# ─────────────────────────────────────────────────────────────
print("\n7. Filling remaining Census NaN...")

for col in ['median_household_income', 'total_population', 'white_population']:
    missing = df[col].isna().sum()
    if missing > 0:
        df[col] = df.groupby('boro')[col].transform(lambda x: x.fillna(x.median()))
        print(f"   {col}: filled {missing} NaN with borough median")

# ─────────────────────────────────────────────────────────────
# 8. prev_score / score_trend 缺失
#    首次检查的餐厅没有历史记录，属于正常缺失
#    新增 is_first_inspection / has_history 标记列
# ─────────────────────────────────────────────────────────────
print("\n8. Handling historical inspection features...")

df['is_first_inspection'] = (df['inspection_count'] == 1).astype(int)
df['has_history']         = df['prev_score'].notna().astype(int)
df['prev_score']          = df['prev_score'].fillna(-1)
df['score_trend']         = df['score_trend'].fillna(0)
df['prev_failed']         = df['prev_failed'].fillna(-1)

print(f"   First-time inspections: {df['is_first_inspection'].sum():,}")
print(f"   Restaurants with history: {df['has_history'].sum():,}")

# ─────────────────────────────────────────────────────────────
# 9. cuisine 标准化
# ─────────────────────────────────────────────────────────────
print("\n9. Cleaning cuisine field...")

df['cuisine'] = df['cuisine'].str.strip().str.title()
top_cuisines  = df['cuisine'].value_counts().head(20).index.tolist()
df['cuisine_grouped'] = df['cuisine'].where(df['cuisine'].isin(top_cuisines), other='Other')
print(f"   Top 20 cuisines kept, rest grouped as 'Other'")

# ─────────────────────────────────────────────────────────────
# 10. yelp_category 简化
#     取第一个tag（逗号分隔），缺失填 'unknown'
# ─────────────────────────────────────────────────────────────
print("\n10. Simplifying yelp_category...")

df['yelp_category_primary'] = (
    df['yelp_category']
    .fillna('unknown')
    .str.split(',')
    .str[0]
    .str.strip()
)
print(f"   Unique primary categories: {df['yelp_category_primary'].nunique()}")

# ─────────────────────────────────────────────────────────────
# 11. 衍生特征
#     - white_pct: 白人人口占比
#     - score_bucket: 分数分箱（A/B/C/NA）
# ─────────────────────────────────────────────────────────────
print("\n11. Adding derived features...")

df['white_pct'] = df['white_population'] / df['total_population']

def score_to_bucket(s):
    if pd.isna(s):
        return 'NA'
    elif s <= 13:
        return 'A'
    elif s <= 27:
        return 'B'
    else:
        return 'C'

df['score_bucket'] = df['score'].apply(score_to_bucket)
print("   Added: white_pct, score_bucket")

# ─────────────────────────────────────────────────────────────
# 12. score_bucket NaN修复（score和grade同时缺失时）
# ─────────────────────────────────────────────────────────────
print("\n12. Fixing score_bucket NaN...")

before_nan = df['score_bucket'].isna().sum()
df['score_bucket'] = df['score_bucket'].fillna('NA')
print(f"   Fixed {before_nan} NaN → 'NA'")
print(f"   score_bucket distribution: {df['score_bucket'].value_counts().to_dict()}")

# ─────────────────────────────────────────────────────────────
# 13. 最终检查
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Final shape:", df.shape)
print("\nRemaining missing values:")
missing = df.isnull().sum()
remaining = missing[missing > 0]
if len(remaining) > 0:
    print(remaining.to_string())
print(f"\nTarget 1 - failed rate:          {df['failed'].mean():.1%}")
print(f"Target 2 - high_rating coverage: {df['has_yelp'].mean():.1%}")
print(f"has_location=1:                  {df['has_location'].mean():.1%}")

# ─────────────────────────────────────────────────────────────
# 14. 保存
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Saving files...")

clean_path = os.path.join(RAW_DIR, 'restaurant_clean.csv')
df.to_csv(clean_path, index=False)
print(f"   Saved: restaurant_clean.csv  ({len(df):,} rows x {df.shape[1]} cols)")

yelp_subset = df[df['has_yelp'] == 1].copy()
yelp_path   = os.path.join(RAW_DIR, 'restaurant_yelp_subset.csv')
yelp_subset.to_csv(yelp_path, index=False)
print(f"   Saved: restaurant_yelp_subset.csv  ({len(yelp_subset):,} rows x {yelp_subset.shape[1]} cols)")

print("\nCLEANING COMPLETE")
print("=" * 60)
print("New indicator columns added:")
print("  grade_available  — whether grade was assigned")
print("  has_location     — whether lat/lon exists")
print("  has_yelp         — whether Yelp data was matched")
print("  has_history      — whether prior inspection exists")
print("  is_first_inspection — first-time inspection flag")
print("\nFiles for downstream use:")
print("  restaurant_clean.csv        → EDA + failed prediction (full dataset)")
print("  restaurant_yelp_subset.csv  → high_rating prediction (Yelp matched)")
