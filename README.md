# missingly

> **Missing data analysis for pandas — batteries included.**

[![PyPI](https://img.shields.io/pypi/v/missingly)](https://pypi.org/project/missingly/)
[![Python](https://img.shields.io/pypi/pyversions/missingly)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

missingly is a Python package for **diagnosing, visualising, and imputing
missing data** in pandas DataFrames. It provides:

- A fluent `df.miss.*` accessor that mirrors the ergonomics of the R `naniar` package.
- sklearn-compatible transformers (`MissinglyImputer`) for use inside `Pipeline`.
- One-shot HTML reports (`create_report`).
- Statistical tests for MCAR / MAR / MNAR mechanisms.
- Time-series-aware gap analysis and imputation.

---

## Public API (v1)

The symbols below are the **stable, supported API surface** for v1.
Breaking changes to these will be announced via a major-version bump.

### `df.miss.*` accessor

```python
import missingly  # registers df.miss automatically
import pandas as pd
import numpy as np

df = pd.DataFrame({"a": [1, np.nan, 3], "b": [np.nan, np.nan, 6]})

df.miss.n_miss()            # 3  — total missing count
df.miss.pct_miss()          # 50.0  — % missing across whole DataFrame
df.miss.miss_var_summary()  # per-column summary table
df.miss.vis_miss()          # missingness matrix visualisation
df.miss.impute(strategy="mean")  # returns imputed DataFrame
```

### Summary & Diagnosis

```python
import missingly as mi

mi.n_miss(df)             # int — total missing count
mi.pct_miss(df)           # float — overall % missing
mi.miss_var_summary(df)   # pd.DataFrame — per-column breakdown
mi.miss_case_summary(df)  # pd.DataFrame — per-row breakdown
mi.mcar_test(df)          # Little's MCAR test result
mi.mar_mnar_test(df)      # MAR vs MNAR indicator
mi.diagnose_missing(df)   # mechanism + recommendation dict
```

### Visualisation

```python
mi.vis_miss(df)          # missingness overview matrix
mi.matrix(df)            # nullity matrix (missingno-style)
mi.bar(df)               # bar chart of missing counts
mi.upset(df)             # UpSet plot of co-missing patterns
mi.miss_patterns(df)     # pattern frequency bar chart
mi.miss_cooccurrence(df) # co-occurrence heatmap
mi.heatmap(df)           # nullity correlation heatmap
mi.miss_cluster(df)      # hierarchical cluster heatmap
```

### `MissinglyImputer` inside a sklearn Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from missingly import MissinglyImputer

pipe = Pipeline([
    ("imputer", MissinglyImputer(strategy="knn")),
    ("clf",     RandomForestClassifier()),
])
pipe.fit(X_train, y_train)
predictions = pipe.predict(X_test)
```

### Report

```python
import missingly as mi

mi.create_report(df, output_path="missing_report.html")
# Opens a standalone HTML report in the current directory
```

### Time Series

```python
mi.miss_ts_summary(df)          # gap count, mean/max gap per column
mi.vis_ts_miss(df)              # tile heatmap of missingness over time
mi.vis_miss_over_time(df)       # rolling missingness rate chart
mi.impute_ts(df, method="time") # time-aware interpolation
```

### Advanced / Experimental Utilities

The following tools are available but **experimental** — they emit a
`FutureWarning` on every call and **may be moved to separate packages,
renamed, or removed** in a future release without a major-version bump:

- **Simulation:** `simulate_mcar`, `simulate_mar`, `simulate_mnar`,
  `simulate_mixed` — introduce controlled missingness for benchmarking.
  These may migrate to a dedicated `missingly-sim` package in 0.3.0+.
- **Time-series extras:** `gap_table`, `vis_gap_lengths` — detailed
  per-gap diagnostics. Core time-series functions (`miss_ts_summary`,
  `vis_ts_miss`, `impute_ts`) are stable v1 API.
- **Performance helpers:** `optimize_dtypes`, `memory_usage_mb`,
  `chunk_apply` — utility shims kept for one release cycle; may move
  to `data_quality_toolkit` in 0.3.0+.
- **Stats extras:** `hotelling_test`, `pattern_monotone_test`,
  `missing_correlation_matrix` (and `test_*` aliases).
- **Data manipulation helpers:** `clean_names`, `remove_empty`,
  `coalesce_columns`, `miss_as_feature`.

---

## Why missingly?

Python's missing-data ecosystem is fragmented:
[missingno](https://github.com/ResidentMario/missingno) visualises,
[fancyimpute](https://github.com/iskandr/fancyimpute) imputes, and
scikit-learn transformers handle the pipeline — but nothing ties them
together in one coherent, opinionated API.

missingly draws inspiration from R's
[naniar](https://github.com/njtierney/naniar) and
[VIM](https://github.com/statistikat/VIM) to give you a **single,
pandas-native entry point** for the full missing-data workflow.

---

## Installation

```bash
pip install missingly
```

---

## Quick Start

```python
import pandas as pd
import numpy as np
import missingly as mi

# Sample data
df = pd.DataFrame({
    "age":    [25, np.nan, 30, np.nan, 45],
    "income": [50000, 60000, np.nan, 80000, np.nan],
    "score":  [7.2, 8.1, np.nan, 6.5, 9.0],
})

# Summarise
print(mi.miss_var_summary(df))

# Visualise
mi.vis_miss(df)

# Test mechanism
print(mi.mcar_test(df))

# Impute
from missingly import MissinglyImputer
imp = MissinglyImputer(strategy="knn")
df_clean = pd.DataFrame(imp.fit_transform(df), columns=df.columns)

# Report
mi.create_report(df, output_path="report.html")
```

---

## License

MIT © Ali Sadeghi Aghili
