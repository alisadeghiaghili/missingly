# missingly

> **This README describes the v1.0.0+ public API. For historical experiments see the [`legacy-experiments`](https://github.com/alisadeghiaghili/missingly/tree/legacy-experiments) branch.**

> **Missing data analysis for pandas — batteries included.**

[![PyPI](https://img.shields.io/pypi/v/missingly)](https://pypi.org/project/missingly/)
[![Python](https://img.shields.io/pypi/pyversions/missingly)]()
[![CI](https://github.com/alisadeghiaghili/missingly/actions/workflows/ci.yml/badge.svg)](https://github.com/alisadeghiaghili/missingly/actions/workflows/ci.yml)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-0969da)](https://alisadeghiaghili.github.io/missingly/)
[![codecov](https://codecov.io/gh/alisadeghiaghili/missingly/branch/main/graph/badge.svg)](https://codecov.io/gh/alisadeghiaghili/missingly)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![API stability](https://img.shields.io/badge/API-v1.0.0%20stable-brightgreen)](https://github.com/alisadeghiaghili/missingly/releases/tag/v1.0.0)

🌐 English | [Deutsch](README_DE.md) | [فارسی](README_FA.md)

missingly is a Python package for **diagnosing, visualising, and imputing
missing data** in pandas DataFrames. It provides:

- A fluent `df.miss.*` accessor that mirrors the ergonomics of the R `naniar` package.
- sklearn-compatible transformers (`MissinglyImputer`) for use inside `Pipeline`.
- One-shot HTML reports (`create_report`).
- Little's MCAR test and explicitly limited observed-data missingness diagnostics.
- Time-series-aware gap analysis and imputation.

### Multiple Imputation (advanced)

`impute_mice(..., n_imputations=m)` can generate multiple completed datasets
for an analysis workflow. Its default numeric target contract is posterior
Bayesian ridge; categorical targets use an ordinal-encoded Bayesian-ridge
approximation. The latter is not a validated multinomial or ordinal FCS kernel,
and the package does not claim general parity with R `mice` or valid inference
for every missing-data mechanism. Inspect `return_result=True` provenance and
warnings, assess convergence, and validate study-specific assumptions before
pooling model results with Rubin's Rules via `missingly.mi`:

Supplying `estimator=` replaces the numeric target estimator, but categorical
targets are still ordinal-encoded before the caller-supplied estimator. That
custom-estimator categorical path remains an approximation rather than a
validated multinomial or ordinal FCS kernel.

```python
import numpy as np
import pandas as pd
from missingly import impute_mice
from missingly.mi import pool_scalar_estimates
from sklearn.linear_model import LinearRegression

# 1. Generate m imputed datasets
dfs = impute_mice(df, n_imputations=5)

# 2. Fit model on each imputed dataset
beta1_ests, beta1_vars = [], []
for d in dfs:
    reg = LinearRegression().fit(d[["x"]], d["y"])
    beta1_ests.append(float(reg.coef_[0]))
    resid = d["y"] - reg.predict(d[["x"]])
    ss_x = float(((d["x"] - d["x"].mean()) ** 2).sum())
    beta1_vars.append(float(np.var(resid, ddof=2)) / ss_x)

# 3. Pool with Rubin's Rules
result = pool_scalar_estimates(beta1_ests, beta1_vars)
print(f"Pooled beta1 = {result['q_bar']:.3f}  (total var = {result['t']:.4f})")
```

For multivariate models use `pool_linear_regression_results(coefs, covs)`
which accepts arrays of shape `(m, p)` and `(m, p, p)` and returns a
pooled coefficient vector plus a pooled covariance matrix.

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
# `outcome` must be fully observed and have one value per row in `df`.
mi.missingness_association_test(df, outcome)  # observed-data association screen
mi.diagnose_missing(df)   # MCAR evidence + assumption-aware guidance
```

### Visualisation

The visualisation layer lives in `missingly.visualisation` and is re-exported
through `missingly.visualise` for backwards compatibility.

> **Module layout**
> | Module | Contents |
> |---|---|
> | `missingly.visualisation.static` | All matplotlib-based functions |
> | `missingly.visualisation.interactive` | All Plotly backends (called when `interactive=True`) |
> | `missingly.visualisation._base` | Shared helpers: `_rtl_safe`, `_safe_labels`, `_nullity`, `_pct_labels` |
> | `missingly.visualise` | Thin re-export facade — use this in application code |

#### Basic

```python
import missingly as mi
import pandas as pd, numpy as np

df = pd.DataFrame({
    "age":    [25, np.nan, 47, 33, np.nan],
    "income": [50000, 62000, np.nan, np.nan, 71000],
    "city":   ["A", "B", np.nan, "A", "C"],
})

mi.vis_miss(df)          # annotated tile matrix with per-column % labels
mi.matrix(df)            # raw presence/absence heatmap
mi.bar(df)               # bar chart: count of missing per column
mi.miss_case(df)         # bar chart: count of missing per row
mi.miss_var_pct(df)      # horizontal bars: % missing per variable, sorted
```

#### Patterns

```python
mi.miss_patterns(df)     # horizontal bars: top-N most frequent missingness patterns
mi.miss_cooccurrence(df) # symmetric heatmap: how often two columns miss together
mi.upset(df)             # UpSet plot of intersecting missingness sets
                         # returns dict of Axes: {"intersections", "matrix", "totals"}
```

#### Correlation / Clustering

```python
# Nullity-correlation heatmap (Pearson on binary missingness indicators).
# The core renderer is local, offline, and independent of DQT.
mi.heatmap(df)
mi.heatmap(df, mask_insignificant=True)  # grey out non-significant cells

# Hierarchical clustering of rows by missingness pattern.
# Returns a single matplotlib.axes.Axes (not a dict).
ax = mi.miss_cluster(df)

# Dendrogram of variables clustered by nullity correlation.
mi.dendrogram(df)
```

#### Interactive

Pass `interactive=True` to any function below to get a Plotly figure
that can be panned, zoomed, and exported to HTML.
Install the explicit optional dependency first with
`pip install "missingly[interactive]"`. Without Plotly, interactive calls
raise an actionable `ImportError`; they do not silently change backend.

```python
mi.vis_miss(df, interactive=True)
mi.heatmap(df, interactive=True)
mi.matrix(df, interactive=True)
mi.bar(df, interactive=True)
mi.miss_var_pct(df, interactive=True)
mi.miss_cooccurrence(df, interactive=True)
mi.miss_case(df, interactive=True)
mi.upset(df, interactive=True)
mi.miss_patterns(df, interactive=True)
```

### Imputation

```python
mi.impute_mean(df)           # mean imputation
mi.impute_median(df)         # median imputation
mi.impute_mode(df)           # mode imputation
mi.impute_knn(df)            # k-NN imputation (Euclidean distance, numeric-safe)
mi.impute_mice(df)           # MICE (IterativeImputer + BayesianRidge)
mi.impute_rf(df)             # Random Forest imputation
mi.impute_gb(df)             # Gradient Boosting imputation

# Multiple Imputation — generate m datasets for Rubin pooling
dfs = mi.impute_mice(df, n_imputations=5)
```

#### KNN with Gower distance (mixed numeric + categorical)

By default `impute_knn` ordinal-encodes categorical columns and uses
Euclidean distance — fast and suitable for mostly-numeric datasets.

For datasets dominated by categorical columns, pass `metric="mixed"` to
use **Gower distance** instead.  Gower treats numeric and categorical
columns correctly: numeric columns are normalised by their range, nominal
columns are compared by exact match.

```python
df_mixed = pd.DataFrame({
    "age":    [25, np.nan, 35, 40],
    "city":   ["London", "Paris", None, "Berlin"],
    "grade":  ["A", "B", "A", None],
})

# Euclidean KNN (default) — fast, ordinal-encodes categoricals
result = mi.impute_knn(df_mixed, n_neighbors=3)

# Gower KNN — statistically sound for heavy-categorical data
result = mi.impute_knn(df_mixed, n_neighbors=3, metric="mixed")
```

> **Performance note:** Gower distance is **O(n²)** in both memory and
> runtime.  Avoid `metric="mixed"` for datasets with more than ~10 000 rows.

`metric="mixed"` is rejected by `MissinglyImputer` because the current Gower
implementation computes distances across the supplied frame and
therefore has no safe inductive train/test donor contract. Use the standalone
function only for transductive exploration; use the default Euclidean
transformer mode inside a sklearn pipeline:

```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ("impute", mi.MissinglyImputer(strategy="knn", n_neighbors=5)),
    ("model",  LogisticRegression()),
])
pipe.fit(X_train, y_train)
```

### Time-series missingness

For time-indexed data, missingly provides gap-aware summary statistics,
visualisation helpers, and interpolation-based imputation.

```python
import missingly as mi
import pandas as pd
import numpy as np

# Build a temperature series with some gaps
index = pd.date_range("2024-01-01", periods=14, freq="D")
temp  = [5.1, 4.8, np.nan, np.nan, 6.2, 6.5, np.nan, 7.0,
         7.3, np.nan, np.nan, np.nan, 8.1, 8.4]
ts = pd.DataFrame({"temp": temp}, index=index)

# 1. Summarise gaps
summary = mi.miss_ts_summary(ts)
print(summary)
#           n_miss  pct_miss  n_gaps  mean_gap  max_gap
# variable
# temp           6     42.86       3       2.0        3

# 2. Visualise missingness over the time axis
ax = mi.vis_ts_miss(ts)

# 3. Impute with linear interpolation
ts_filled = mi.impute_ts(ts, strategy="linear")
print(ts_filled.isnull().sum())  # temp    0
```

**Available strategies for `impute_ts`:** `ffill`, `bfill`, `linear`, `time`,
`spline`. Use `limit=n` to cap how many consecutive NaNs are filled.

```python
# Fill at most 2 consecutive NaNs, leave longer gaps as-is
ts_partial = mi.impute_ts(ts, strategy="linear", limit=2)
```

**Gap inspection with `gap_table`:**

```python
from missingly.timeseries import gap_table

gt = gap_table(ts)
print(gt)
#    column  gap_start   gap_end  gap_length
# 0    temp 2024-01-03 2024-01-04           2
# 1    temp 2024-01-07 2024-01-07           1
# 2    temp 2024-01-10 2024-01-12           3
```

### sklearn Pipeline integration

```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ("impute", mi.MissinglyImputer(strategy="knn")),
    ("model",  LogisticRegression()),
])
pipe.fit(X_train, y_train)
```

---

## Installation

```bash
# Core package
pip install missingly

# With interactive Plotly charts
pip install missingly[interactive]

# With Persian / Arabic (RTL) support for static matplotlib plots
# Required when column names or labels contain Persian/Arabic characters
pip install missingly[rtl]

# Everything (interactive + RTL)
pip install missingly[all]
```

> **Persian/Arabic users:** static matplotlib plots require `missingly[rtl]`
> (installs `arabic-reshaper` and `python-bidi`) **plus** a compatible font
> such as [Vazirmatn](https://fonts.google.com/specimen/Vazirmatn) installed
> on your system. Static rendering never downloads fonts or writes a font cache:
> it uses a packaged font only when a distribution explicitly includes one, then
> falls back to installed system fonts with a warning. Interactive Plotly charts
> (`interactive=True`) require the `missingly[interactive]` extra shown above.

---

## License

Apache 2.0 — Copyright 2025 Ali Sadeghi Aghili. See [LICENSE](LICENSE).
