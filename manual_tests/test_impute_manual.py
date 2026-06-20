# test_impute_on_petrochemical.py
# Test imputation functions from missingly.impute on the petrochemical dataset.
# Uses a small sample (first 1000 rows) for speed.

import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DATASET_PATH = "manual_tests/petrochemical_tank_sensors.csv"

# ------------------------------------------------------------
# 1. Load or create a small dirty dataset
# ------------------------------------------------------------
if os.path.exists(DATASET_PATH):
    print("Loading full CSV, then taking first 1000 rows for testing...")
    full_df = pd.read_csv(DATASET_PATH, encoding="utf-8-sig")
    df_sample = full_df.iloc[:1000].copy().reset_index(drop=True)
else:
    print("CSV not found. Creating a small synthetic dataset with dirty data...")
    np.random.seed(42)
    n = 1000
    data = {
        "\u0632\u0645\u0627\u0646 \u062b\u0628\u062a": pd.date_range("2024-01-01", periods=n, freq="H"),
        "\u06a9\u062f \u0645\u062e\u0632\u0646": np.random.choice([f"\u0645\u062e\u0632\u0646{i:04d}" for i in range(1, 6)], n),
        "\u062f\u0645\u0627 (\u0633\u0627\u0646\u062a\u06cc\u200c\u06af\u0631\u0627\u062f)": np.random.normal(45, 10, n).round(2),
        "\u0641\u0634\u0627\u0631 (bar)": np.random.normal(5, 1.5, n).round(3),
        "\u0633\u0631\u0639\u062a \u062c\u0631\u06cc\u0627\u0646 (m\u00b3/h)": np.random.exponential(50, n).round(1),
        "\u0633\u0637\u062d \u0645\u0627\u06cc\u0639 (%)": np.random.uniform(20, 100, n).round(1),
        "PH": np.random.uniform(6.5, 8.5, n).round(2),
        "\u0648\u0636\u0639\u06cc\u062a \u0634\u06cc\u0631": np.random.choice(["\u0628\u0627\u0632", "\u0628\u0633\u062a\u0647", "\u0646\u06cc\u0645\u0647\u200c\u0628\u0627\u0632"], n, p=[0.6, 0.3, 0.1]),
        "\u0646\u0648\u0639 \u0645\u062d\u0635\u0648\u0644": np.random.choice(["\u0627\u062a\u0627\u0646", "\u067e\u0631\u0648\u067e\u0627\u0646", "\u0628\u0648\u062a\u0627\u0646", "\u067e\u0644\u06cc\u200c\u0627\u062a\u06cc\u0644\u0646", "\u0628\u0646\u0632\u06cc\u0646"], n),
    }
    df_sample = pd.DataFrame(data)
    for col in ["\u062f\u0645\u0627 (\u0633\u0627\u0646\u062a\u06cc\u200c\u06af\u0631\u0627\u062f)", "\u0641\u0634\u0627\u0631 (bar)", "\u0633\u0631\u0639\u062a \u062c\u0631\u06cc\u0627\u0646 (m\u00b3/h)", "\u0633\u0637\u062d \u0645\u0627\u06cc\u0639 (%)", "PH"]:
        mask = np.random.random(n) < 0.08
        df_sample.loc[mask, col] = np.nan
    for col in ["\u0648\u0636\u0639\u06cc\u062a \u0634\u06cc\u0631", "\u0646\u0648\u0639 \u0645\u062d\u0635\u0648\u0644"]:
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
    make_imputer,
)

# ------------------------------------------------------------
# 3. Test impute_mean
# ------------------------------------------------------------
print("=" * 50)
print("Testing impute_mean ...")
df_mean = impute_mean(df_sample, numeric_only=False)
print(f"Missing after impute_mean: {df_mean.isna().sum().sum()}")
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
train = df_sample.iloc[:800].copy()
test = df_sample.iloc[800:1000].copy()

imp = make_imputer(strategy="mean")
imp.fit(train)
print(f"Is fitted? {imp.is_fitted}")

test_imputed = imp.transform(test)
print(f"Missing in test after transform: {test_imputed.isna().sum().sum()}")
assert test_imputed.isna().sum().sum() == 0
print("✅ FittedImputer PASSED\n")

# ------------------------------------------------------------
# 10. Sample comparison
# ------------------------------------------------------------
print("Sample comparison (first 5 rows of original vs impute_mean):")
print("Original (first 5 rows):")
print(df_sample.head())
print("\nImputed with mean:")
print(df_mean.head())

print("\n✅ All imputation tests completed successfully.")
