# Manual Tests

This directory contains manual test scripts for validating `missingly` functionality end-to-end with real datasets.

## Setup

First generate the test datasets:

```bash
# Generate English sensor dataset (small, ~500 rows)
python manual_tests/generate_sensor_data.py

# Generate Persian petrochemical dataset (~1M rows)
python manual_tests/generate_petrochemical_sensors.py
```

## Running Tests

```bash
# Summary functions
python manual_tests/test_summary_manual.py

# Imputation
python manual_tests/test_impute_manual.py

# Statistics (MCAR/MAR/MNAR)
python manual_tests/test_stats_manual.py
python manual_tests/test_stats_extra_manual.py

# Multiple Imputation
python manual_tests/test_mi_manual.py

# Compare imputation strategies
python manual_tests/test_compare_manual.py
python manual_tests/test_compare_manual_2.py

# Time series
python manual_tests/test_timeseries_manual.py

# HTML Report
python manual_tests/test_report_manual.py
```

## Notes

- Generated output files (CSV, PNG, HTML, NPY) are excluded from git via `.gitignore`
- Run scripts from the **project root**, not from inside `manual_tests/`
- The petrochemical dataset is ~1M rows and takes ~30s to generate
