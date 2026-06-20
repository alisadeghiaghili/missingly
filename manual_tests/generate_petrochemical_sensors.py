# generate_petrochemical_sensors.py
"""Generate a dirty dataset of ~1M sensor readings from petrochemical tanks (Persian columns)."""
import pandas as pd
import numpy as np

np.random.seed(123)

n = 1_000_000  # number of readings

# Generate clean data
timestamps = pd.date_range('2024-01-01', periods=n, freq='30s')[:n]
tank_ids = [f'مخزن{i:04d}' for i in range(1, 201)]  # 200 tanks

data = {
    'زمان ثبت': timestamps,
    'کد مخزن': np.random.choice(tank_ids, size=n),
    'دما (سانتی\u200cگراد)': np.random.normal(45, 10, size=n).round(2),
    'فشار (bar)': np.random.normal(5, 1.5, size=n).round(3),
    'سرعت جریان (m³/h)': np.random.exponential(50, size=n).round(1),
    'سطح مایع (%)': np.random.uniform(20, 100, size=n).round(1),
    'PH': np.random.uniform(6.5, 8.5, size=n).round(2),
    'وضعیت شیر': np.random.choice(['باز', 'بسته', 'نیمه\u200cباز'], size=n, p=[0.6, 0.3, 0.1]),
    'نوع محصول': np.random.choice(['اتان', 'پروپان', 'بوتان', 'پلی\u200cاتیلن', 'بنزین'],
                                  size=n, p=[0.3, 0.25, 0.2, 0.15, 0.1]),
}

df = pd.DataFrame(data)

# ----- Inject dirt -----

# 1. Missing values (8% random in most numeric columns, 3% in categoricals)
for col in ['دما (سانتی\u200cگراد)', 'فشار (bar)', 'سرعت جریان (m³/h)', 'سطح مایع (%)', 'PH']:
    mask = np.random.random(n) < 0.08
    df.loc[mask, col] = np.nan

for col in ['وضعیت شیر', 'نوع محصول']:
    mask = np.random.random(n) < 0.03
    df.loc[mask, col] = np.nan

# 2. Outliers in temperature (below -50 or above 150)
outlier_temp = np.random.random(n) < 0.003
df.loc[outlier_temp, 'دما (سانتی\u200cگراد)'] = np.random.choice([-100, 200, 999], size=outlier_temp.sum())

# 3. Outliers in pressure (negative or > 100 bar)
outlier_pres = np.random.random(n) < 0.002
df.loc[outlier_pres, 'فشار (bar)'] = np.random.choice([-5, 150, 0], size=outlier_pres.sum())

# 4. Outliers in flow rate (enormous values)
outlier_flow = np.random.random(n) < 0.001
df.loc[outlier_flow, 'سرعت جریان (m³/h)'] = np.random.uniform(5000, 10000, size=outlier_flow.sum())

# 5. Duplicate rows (exact)
dup_count = 1500
dup_indices = np.random.choice(n, dup_count, replace=False)
df = pd.concat([df, df.iloc[dup_indices]], ignore_index=True)

# 6. Inconsistent categoricals in 'نوع محصول'
typo_map = {'اتان': 'اتانن', 'پروپان': 'پروپانن', 'بوتان': 'بوتانن'}
for orig, typo in typo_map.items():
    mask = df['نوع محصول'] == orig
    subset = mask[mask].sample(frac=0.02, random_state=42).index
    df.loc[subset, 'نوع محصول'] = typo

# 7. Future timestamps (impossible)
future_mask = np.random.random(len(df)) < 0.001
df.loc[future_mask, 'زمان ثبت'] = pd.Timestamp('2050-06-15 12:00:00')

# 8. Duplicate tank IDs with extra spaces
space_tank_mask = np.random.random(len(df)) < 0.005
df.loc[space_tank_mask, 'کد مخزن'] = df.loc[space_tank_mask, 'کد مخزن'].apply(lambda x: x + ' ')

# Shuffle
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save
df.to_csv('petrochemical_tank_sensors.csv', index=False, encoding='utf-8-sig')
print("Generated petrochemical_tank_sensors.csv with", len(df), "rows.")
print("Missing values per column:\n", df.isna().sum())
