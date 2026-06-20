# test_summary_on_petrochemical.py
# Load the dirty petrochemical sensor dataset and test all summary functions.

import pandas as pd
import numpy as np

from missingly.summary import (
    n_miss, n_complete, pct_miss, pct_complete,
    miss_var_summary, miss_case_summary,
    summarise, miss_summary, bind_shadow, miss_scan_count,
)

DATASET_PATH = "manual_tests/petrochemical_tank_sensors.csv"

print("Loading petrochemical_tank_sensors.csv ...")
df = pd.read_csv(DATASET_PATH, encoding="utf-8-sig")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}\n")

# Basic counts
print("=== Basic counts (NaN only) ===")
print(f"Total missing values: {n_miss(df)}")
print(f"Total non-missing values: {n_complete(df)}")
print(f"Percentage missing: {pct_miss(df):.2f}%")
print(f"Percentage complete: {pct_complete(df):.2f}%\n")

# Per-variable summary
print("=== Missingness per variable (top 10) ===")
var_summary = miss_var_summary(df)
print(var_summary.sort_values("pct_miss", ascending=False).head(10))
print()

# Per-case summary
print("=== Missingness per case (top 10 rows) ===")
case_summary = miss_case_summary(df)
print(case_summary.sort_values("n_miss", ascending=False).head(10))
print()

# Dataset-level summary
print("=== Dataset-level summary ===")
s = summarise(df)
print(s.to_string(index=False))
print()

# Comprehensive summary
print("=== Comprehensive summary (miss_summary) ===")
comprehensive = miss_summary(df, digits=2)
print("Summary (dataset):")
print(comprehensive["summary"].to_string(index=False))
print("\nBy variable (first 5 rows):")
print(comprehensive["by_variable"].head())
print("\nBy case (first 5 rows):")
print(comprehensive["by_case"].head())
print()

# Shadow matrix
print("=== Shadow matrix (first 5 rows) ===")
shadowed = bind_shadow(df)
print(shadowed.head())
print(f"Shadow columns added: {[c for c in shadowed.columns if c.endswith('_NA')][:3]} ...\n")

# Sentinel scan
print("=== Scanning for sentinel values (999, -100, 0 in pressure?) ===")
sentinel_search = [999, -100, -5, 150, -99]
scan_result = miss_scan_count(df, search=sentinel_search)
if len(scan_result) > 0:
    print(scan_result)
else:
    print("No sentinel values found among those listed.")
print()

# Custom missing_values
print("=== Treating some outliers as missing (custom missing_values) ===")
custom_miss = [-100, 999, -5, 150]
print(f"n_miss with custom sentinels: {n_miss(df, missing_values=custom_miss)}")
print(f"pct_miss with custom sentinels: {pct_miss(df, missing_values=custom_miss):.2f}%")
print()

print("All summary tests completed successfully.")
