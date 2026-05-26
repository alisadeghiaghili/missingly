# missingly

> **This README describes the v1.0.0+ public API. For historical experiments see the [`legacy-experiments`](https://github.com/alisadeghiaghili/missingly/tree/legacy-experiments) branch.**

> **Missing data analysis for pandas — batteries included.**

[![PyPI](https://img.shields.io/pypi/v/missingly)](https://pypi.org/project/missingly/)
[![Python](https://img.shields.io/pypi/pyversions/missingly)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![API stability](https://img.shields.io/badge/API-v1.0.0%20stable-brightgreen)](https://github.com/alisadeghiaghili/missingly/releases/tag/v1.0.0)

missingly is a Python package for **diagnosing, visualising, and imputing
missing data** in pandas DataFrames. It provides:

- A fluent `df.miss.*` accessor that mirrors the ergonomics of the R `naniar` package.
- sklearn-compatible transformers (`MissinglyImputer`) for use inside `Pipeline`.
- One-shot HTML reports (`create_report`).
- Statistical tests for MCAR / MAR / MNAR mechanisms.
- Time-series-aware gap analysis and imputation.

### Multiple Imputation (advanced)

For statistically valid inference after imputation, generate *m* datasets
with `impute_mice(..., n_imputations=m)` and pool the model results using
Rubin's Rules via the utilities in `missingly.mi`:

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
mi.mar_mnar_test(df)      # MAR vs MNAR indicator
mi.diagnose_missing(df)   # mechanism + recommendation dict
```

### Visualisation

```python
mi.vis_miss(df)            # missingness matrix (static or interactive)
mi.matrix(df)              # similar matrix plot
mi.bar(df)                 # bar chart of % missing per column
mi.upset(df)               # UpSet plot of missingness patterns
mi.miss_patterns(df)       # pattern heatmap
mi.miss_cooccurrence(df)   # co-occurrence matrix
mi.heatmap(df)             # correlation heatmap of missingness
mi.miss_cluster(df)        # clustered missingness heatmap
```

### Imputation

```python
mi.impute_mean(df)           # mean imputation
mi.impute_median(df)         # median imputation
mi.impute_mode(df)           # mode imputation
mi.impute_knn(df)            # k-NN imputation
mi.impute_mice(df)           # MICE (IterativeImputer + BayesianRidge)
mi.impute_rf(df)             # Random Forest imputation
mi.impute_gb(df)             # Gradient Boosting imputation

# Multiple Imputation — generate m datasets for Rubin pooling
dfs = mi.impute_mice(df, n_imputations=5)
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

### Interactive Plots

Pass `interactive=True` to get Plotly-powered plots that can be panned,
zoomed, and exported:

```python
mi.vis_miss(df, interactive=True)
mi.heatmap(df, interactive=True)
mi.bar(df, interactive=True)
```

The following functions support `interactive=True`:
`vis_miss`, `heatmap`, `matrix`, `miss_var_pct`, `miss_cooccurrence`,
`bar`, `miss_case`, `upset`, `miss_patterns`.

---

## Installation

```bash
pip install missingly
```

---

## License

MIT — see [LICENSE](LICENSE).


