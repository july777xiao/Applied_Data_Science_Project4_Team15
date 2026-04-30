"""
修复脚本 — 只修复 score_bucket 的 NaN 问题
输入/输出: project4/raw/ 目录
"""

import pandas as pd
import numpy as np
import os

RAW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'raw')

# ── 读取 ──────────────────────────────────────────────────────
df   = pd.read_csv(os.path.join(RAW_DIR, 'restaurant_clean.csv'),       low_memory=False)
yelp = pd.read_csv(os.path.join(RAW_DIR, 'restaurant_yelp_subset.csv'), low_memory=False)

print("Before fix:")
print("  score_bucket NaN:", df['score_bucket'].isna().sum())

# ── 修复 score_bucket ─────────────────────────────────────────
# score和grade都缺失的行 → 'NA'字符串
df['score_bucket'] = df['score_bucket'].fillna('NA')
yelp['score_bucket'] = yelp['score_bucket'].fillna('NA')

print("After fix:")
print("  score_bucket NaN:", df['score_bucket'].isna().sum())
print("  score_bucket distribution:")
print(" ", df['score_bucket'].value_counts().to_dict())

# ── 顺便确认 yelp_price 4档是正常的 ──────────────────────────
print()
print("yelp_price distribution (4 tiers is correct):")
print(" ", df['yelp_price'].value_counts(dropna=False).to_dict())

# ── 保存 ─────────────────────────────────────────────────────
df.to_csv(os.path.join(RAW_DIR, 'restaurant_clean.csv'), index=False)
yelp.to_csv(os.path.join(RAW_DIR, 'restaurant_yelp_subset.csv'), index=False)

print()
print("DONE — files saved:")
print(f"  restaurant_clean.csv        ({len(df):,} rows x {df.shape[1]} cols)")
print(f"  restaurant_yelp_subset.csv  ({len(yelp):,} rows x {yelp.shape[1]} cols)")
print()
print("Remaining intentional NaN (not problems):")
for col in ['score', 'grade', 'latitude', 'longitude', 'yelp_rating',
            'yelp_price', 'yelp_category', 'high_rating']:
    n = df[col].isna().sum()
    if n > 0:
        print(f"  {col}: {n} (expected)")
