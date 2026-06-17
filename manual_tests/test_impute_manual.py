# test_impute_on_petrochemical.py
# Test imputation functions from missingly.impute on the petrochemical dataset.
# Uses a small sample (first 1000 rows) for speed.

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")  # suppress user warnings for speed

DATASET_DIR = 'manual_tests\petrochemical_tank_sensors.csv'

# ------------------------------------------------------------
# 1. Load or create a small dirty dataset
# ------------------------------------------------------------
DATASET_DIR = 'manual_tests\petrochemical_tank_sensors.csv'
if os.path.exists(DATASET_DIR):
    print("Loading full CSV, then taking first 1000 rows for testing...")
    full_df = pd.read_csv(DATASET_DIR, encoding='utf-8-sig')
    df_sample = full_df.iloc[:1000].copy().reset_index(drop=True)
else:
    print("CSV not found. Creating a small synthetic dataset with dirty data...")
    np.random.seed(42)
    n = 1000
    data = {
        'زمان ثبت': pd.date_range('2024-01-01', periods=n, freq='H'),
        'کد مخزن': np.random.choice([f'مخزن{i:04d}' for i in range(1, 6)], n),
        'دما (سانتی‌گراد)': np.random.normal(45, 10, n).round(2),
        'فشار (bar)': np.random.normal(5, 1.5, n).round(3),
        'سرعت جریان (m³/h)': np.random.exponential(50, n).round(1),
        'سطح مایع (%)': np.random.uniform(20, 100, n).round(1),
        'PH': np.random.uniform(6.5, 8.5, n).round(2),
        'وضعیت شیر': np.random.choice(['باز', 'بسته', 'نیمه‌باز'], n, p=[0.6,0.3,0.1]),
        'نوع محصول': np.random.choice(['اتان', 'پروپان', 'بوتان', 'پلی‌اتیلن', 'بنزین'], n),
    }
    df_sample = pd.DataFrame(data)
    # Inject missing values (8% numeric, 3% categorical)
    for col in ['دما (سانتی‌گراد)', 'فشار (bar)', 'سرعت جریان (m³/h)', 'سطح مایع (%)', 'PH']:
        mask = np.random.random(n) < 0.08
        df_sample.loc[mask, col] = np.nan
    for col in ['وضعیت شیر', 'نوع محصول']:
        mask = np.random.random(n) < 0.03
        df_sample.loc[mask, col] = np.nan

print(f"Sample shape: {df_sample.shape}")
print(f"Columns: {list(df_sample.columns)}")
print(f"Missing counts before imputation:\n{df_sample.isna().sum()}\n")

# ------------------------------------------------------------
# 2. Import impute functions
# ------------------------------------------------------------
from missingly.impute import (
    impute_mean,
    impute_median,
    impute_mode,
    impute_knn,
    impute_mice,
    # impute_rf, impute_gb  # optional, slower
)

# ------------------------------------------------------------
# 3. Test impute_mean
# ------------------------------------------------------------
print("="*50)
print("Testing impute_mean ...")
df_mean = impute_mean(df_sample, numeric_only=False)
print(f"Missing after impute_mean: {df_mean.isna().sum().sum()}")
# Check that numeric columns have no NaN
assert df_mean.isna().sum().sum() == 0, "impute_mean left NaN!"
print("✅ impute_mean PASSED (all NaN filled)\n")

# ------------------------------------------------------------
# 4. Test impute_median
# ------------------------------------------------------------
print("Testing impute_median ...")
df_median = impute_median(df_sample, numeric_only=False)
print(f"Missing after impute_median: {df_median.isna().sum().sum()}")
assert df_median.isna().sum().sum() == 0
print("✅ impute_median PASSED\n")

# ------------------------------------------------------------
# 5. Test impute_mode
# ------------------------------------------------------------
print("Testing impute_mode ...")
df_mode = impute_mode(df_sample)
print(f"Missing after impute_mode: {df_mode.isna().sum().sum()}")
assert df_mode.isna().sum().sum() == 0
print("✅ impute_mode PASSED\n")

# ------------------------------------------------------------
# 6. Test impute_knn (Euclidean) – on a subset (200 rows) for speed
# ------------------------------------------------------------
print("Testing impute_knn (metric='euclidean') on first 200 rows ...")
df_knn = impute_knn(df_sample.iloc[:200], n_neighbors=5, metric="euclidean")
print(f"Missing after impute_knn: {df_knn.isna().sum().sum()}")
assert df_knn.isna().sum().sum() == 0
print("✅ impute_knn (euclidean) PASSED\n")

# ------------------------------------------------------------
# 7. Test impute_knn with Gower (mixed) – smaller subset (100 rows)
# ------------------------------------------------------------
print("Testing impute_knn (metric='mixed') on first 100 rows ...")
df_knn_mixed = impute_knn(df_sample.iloc[:100], n_neighbors=3, metric="mixed")
print(f"Missing after impute_knn mixed: {df_knn_mixed.isna().sum().sum()}")
# Note: mixed KNN may still leave NaN if a column is entirely missing in the subset.
# In our sample it should work.
assert df_knn_mixed.isna().sum().sum() == 0
print("✅ impute_knn (mixed) PASSED\n")

# ------------------------------------------------------------
# 8. Test impute_mice – small subset (100 rows, 1 imputation)
# ------------------------------------------------------------
print("Testing impute_mice on first 100 rows (max_iter=3) ...")
df_mice = impute_mice(df_sample.iloc[:100], max_iter=3, random_state=42, n_imputations=1)
print(f"Missing after impute_mice: {df_mice.isna().sum().sum()}")
assert df_mice.isna().sum().sum() == 0
print("✅ impute_mice PASSED\n")

# ------------------------------------------------------------
# 9. Test FittedImputer (fit/transform on train/test split)
# ------------------------------------------------------------
print("Testing FittedImputer (make_imputer) ...")
from missingly.impute import make_imputer

# Split into train/test (80/20)
train = df_sample.iloc[:800].copy()
test  = df_sample.iloc[800:1000].copy()

# Create and fit imputer on train only
imp = make_imputer(strategy='mean')
imp.fit(train)
print(f"Is fitted? {imp.is_fitted}")

# Transform test data (use train statistics)
test_imputed = imp.transform(test)
print(f"Missing in test after transform: {test_imputed.isna().sum().sum()}")
assert test_imputed.isna().sum().sum() == 0
print("✅ FittedImputer PASSED\n")

# ------------------------------------------------------------
# 10. (Optional) Compare imputation values – show a few rows
# ------------------------------------------------------------
print("Sample comparison (first 5 rows of original vs impute_mean):")
print("Original (first 5 rows):")
print(df_sample.head())
print("\nImputed with mean:")
print(df_mean.head())

print("\n✅ All imputation tests completed successfully.")