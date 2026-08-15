---
name: missingly
description: AI co-pilot context for the missingly Python package — missing data analysis, imputation, diagnostics, simulation, and reporting.
version: 1.0.0
author: Ali Sadeghi Aghili
repo: github.com/alisadeghiaghili/missingly
updated: 2026-08-15
---

# SKILL: missingly — AI Co-Pilot Context File

> **Caveman Skill**: This file is the ground truth for any AI assistant working on the `missingly` package.
> Read this before writing a single line of code. No exceptions.

---

## 1. What Is missingly?

`missingly` is a Python package under active development for **missing data analysis, imputation, diagnostics, simulation, and reporting**. Its core dependencies include `pandas`, `numpy`, `scipy`, `scikit-learn`, `statsmodels`, `matplotlib`, `seaborn`, and `jinja2`; `upsetplot` is an optional extra. Treat release readiness and competitive superiority as evidence-based goals, not current capability claims.

- **Target Python**: 3.9+
- **Current large-data support**: pandas operations warn above the configured
  threshold, and the optional Polars adapter provides native missingness
  summaries for eager and lazy frames. Full Arrow/Polars execution and a
  1M-row benchmark are planned work, not a current capability claim.
- **Design philosophy**: pandas-native API, sklearn-compatible transformers, sensible defaults, loud failures
- **Future scalability**: full Arrow/Polars execution is planned; do not infer it
  from the existing native Polars summary adapter.
- **Runtime safety**: static rendering is offline-only. It never downloads
  fonts or writes a user font cache; it uses an explicitly packaged font when
  available, otherwise a system font and an actionable warning. Core plotting
  never imports DQT.
- **Legacy DQT exception**: deprecated forwarding shims remain lazy for the
  current SemVer window and are tracked for removal or redesign in issue #81.
  Do not add a new DQT import to core code.

---

## 2. Package Structure

```
missingly/
├── __init__.py           # Public API surface — imports and __all__
├── accessor.py           # pandas .miss accessor (df.miss.n_miss(), etc.)
├── compare.py            # Compare missingness across datasets/groups
├── config.py             # Global configuration (MissinglyConfig)
├── diagnostics.py        # MCAR tests and observed-data association diagnostics
├── exceptions.py         # Custom exceptions (all loud, actionable messages)
├── impute.py             # Imputation methods (single imputation)
├── manipulation.py       # Insert/remove missing values programmatically
├── mi.py                 # Multiple imputation core
├── mi_workflow.py        # Multiple imputation workflow orchestration
├── performance.py        # Profiling and memory utilities
├── report.py             # HTML report generation (Jinja2)
├── simulate.py           # MCAR/MAR/MNAR simulation
├── stats_extra.py        # Extra statistical utilities
├── timeseries.py         # Time-series-specific missing data handling
├── transformer.py        # MissinglyImputer (sklearn-compatible)
├── visualise.py          # Top-level visualisation dispatcher
├── visualisation/        # Visualisation subpackage (heatmap, matrix, bar, etc.)
├── templates/            # Jinja2 HTML templates for reports
├── _distance.py          # Distance metrics for imputation
├── _validation.py        # Input validation utilities
└── _deprecation.py       # Deprecation warning utilities
```

Root-level files:
```
conftest.py               # pytest fixtures shared across all tests
tests/                    # pytest test suite
manual_tests/             # non-automated exploratory tests
docs/                     # documentation source
CONVENTIONS.md            # MUST READ before writing any code
CHANGELOG.md              # version history
pyproject.toml            # build config (PEP 517)
requirements.txt          # direct-install dependency floors
```

---

## 3. Core API

### 3.1 pandas Accessor — `df.miss`

Registered via `@pd.api.extensions.register_dataframe_accessor("miss")` in `accessor.py`.

```python
import pandas as pd
import missingly  # registers .miss accessor

df.miss.n_miss()                 # total missing count
df.miss.miss_var_summary()       # per-column count, %, dtype
df.miss.matrix()                 # missingno-style matrix plot
df.miss.bar()                    # bar chart of missingness
df.miss.upset()                  # UpSet plot of co-occurrence patterns
df.miss.impute(strategy="mean")  # returns an imputed DataFrame
```

### 3.2 Imputation — `missingly.impute` + `MissinglyImputer`

```python
from missingly.impute import impute_mean, impute_knn, impute_mice
from missingly.transformer import MissinglyImputer  # sklearn API

# Functional
df_imputed = impute_mean(df)
df_imputed = impute_knn(df, n_neighbors=5)
df_imputed = impute_mice(df)

# Transformer (sklearn Pipeline compatible)
imp = MissinglyImputer(strategy="knn", n_neighbors=5)
imp.fit(X_train)
X_imputed = imp.transform(X_test)
```

### 3.3 Diagnostics — `missingly.diagnostics`

```python
from missingly.diagnostics import (
    mcar_test,                      # Little's MCAR test
    missingness_association_test,   # observed-data association screen
    diagnose_missing,               # MCAR evidence and guidance
)

result = mcar_test(df)
result["chi_square"]  # chi-square statistic
result["p_value"]     # small values are evidence against MCAR

# This screen does not identify MAR versus MNAR from observed data.
associations = missingness_association_test(df, outcome)
guidance = diagnose_missing(df)
```

### 3.4 Simulation — `missingly.simulate`

```python
from missingly.simulate import simulate_mcar, simulate_mar, simulate_mnar

df_with_missing = simulate_mcar(df, frac=0.2, random_state=42)
df_with_missing = simulate_mar(df, target_col="y", predictor_col="x", frac=0.3)
df_with_missing = simulate_mnar(df, target_col="y", threshold=0.5, frac=0.25)
```

### 3.5 Multiple Imputation — `missingly.mi`

```python
from missingly import impute_mice
from missingly.mi import pool_scalar_estimates
from missingly.mi_workflow import mi_with

imputed_datasets = impute_mice(df, n_imputations=5, random_state=42)
results = mi_with(imputed_datasets, my_model_fn)
pooled = pool_scalar_estimates(estimates, variances)  # scalar Rubin pooling
```

### 3.6 Compare — `missingly.compare`

```python
from missingly.compare import compare_imputations, cv_compare_imputations

# `compare_imputations` evaluates reconstruction on artificially masked
# values in a complete frame; it is an exploration tool.
summary = compare_imputations(df_complete)
# Cross-validation uses train-fold fit/transform semantics for model selection.
scores = cv_compare_imputations(X, y, estimator)
```

### 3.7 Report — `missingly.report`

```python
from missingly.report import create_report

output_path = create_report(df, output_path="missing_report.html")
```

### 3.8 Time Series — `missingly.timeseries`

```python
from missingly.timeseries import gap_table, miss_ts_summary, vis_ts_miss, impute_ts

summary = miss_ts_summary(df)
gaps = gap_table(df)  # experimental helper; emits a deprecation warning
ax = vis_ts_miss(df)
filled = impute_ts(df, strategy="linear")
```

---

## 4. Exceptions

All package-specific exceptions are in `missingly/exceptions.py`. Prefer the
most specific domain exception at public boundaries when one applies; any
built-in exception must still carry an actionable message.

| Exception | When to use |
|---|---|
| `MissinglyError` | Base exception |
| `ImputationError` | Estimator fit or prediction failed during imputation |
| `InsufficientDataError` | Not enough rows/cols for operation |
| `InvalidStrategyError` | Unknown or misconfigured strategy value |
| `MissingColumnError` | Referenced column missing from DataFrame |
| `ConfigurationError` | Invalid config values |

---

## 5. Conventions (MUST FOLLOW)

Full details in `CONVENTIONS.md`. Summary:

- **Docstrings**: NumPy style. Every public function must have: one-line summary, Parameters, Returns, Examples, Notes (if stats-heavy).
- **PEP 8**: enforced. Line length 88 (black).
- **Type hints**: required on all public functions.
- **Validation**: use `_validation.py` utilities. Validate at the earliest possible point.
- **No silent failures**: if data is bad, raise a `MissinglyError` subclass with a clear message.
- **No new dependencies** without explicit discussion. Core dependencies are
  `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`, `scikit-learn`,
  `statsmodels`, and `jinja2`; `upsetplot` is optional.

---

## 6. Testing Rules

- Test file location: `tests/`
- Shared fixtures: `conftest.py` in root
- Framework: `pytest`
- Authoritative coverage baseline (Python 3.12 CI artifact): **91.98% (3761/4089)**.
  Retrieve the `coverage-py312` artifact from the CI run when measuring
  progress. The current enforcement threshold remains 80% until the real
  suite reaches the separately tracked 98% target; do not mask uncovered
  code with exclusions.
- **Always** test these edge cases:
  - All-missing column/row
  - No missing data at all
  - Mixed dtypes (int, float, str, category, datetime)
  - High-cardinality categorical columns
  - Single row / single column DataFrames
  - DataFrames with duplicate column names (should raise, not silently corrupt)

```python
# conftest.py fixture pattern
@pytest.fixture
def df_mixed():
    return pd.DataFrame({
        "int_col": [1, 2, None, 4],
        "float_col": [1.0, None, 3.0, None],
        "cat_col": pd.Categorical(["a", None, "b", "a"]),
        "str_col": ["x", "y", None, "x"],
        "dt_col": pd.to_datetime(["2021-01", None, "2021-03", "2021-04"]),
    })
```

---

## 7. Performance Rules

- Do not claim a large-data capability without a benchmark and a public
  execution contract.
- Profile memory before shipping operations intended for large frames.
- Use `performance.py` profilers for benchmarking.
- Keep full Arrow/Polars execution, chunked reporting, and 1M-row benchmarks
  explicitly scoped as planned until their tests and benchmarks land.

---

## 8. Historical Review Evidence and Current Limits

The following are historical review leads, not assertions about the current
release. Reproduce each with a focused test before acting on it:

1. Categorical dtype preservation after imputation.
2. `compare.py` behavior when one input has no missing values.
3. Train/test column-consistency validation in `MissinglyImputer`.
4. `mcar_test()` behavior for an all-complete DataFrame.

Current, verified limits include the absence of a full inductive Gower KNN
transformer and the absence of a complete FCS statistical-validation engine.
Those modes must fail loudly or remain outside the public claim surface.

---

## 9. Development Workflow

When implementing a new feature:
1. Read the relevant existing module(s) before writing anything.
2. Propose architecture in plain language — don't just start coding.
3. Implement with full docstrings and type hints.
4. Write tests covering happy path + all edge cases listed in §6.
5. Update `CHANGELOG.md` under `[Unreleased]`.
6. If public API changes: update `README.md` and relevant `docs/` pages.

When fixing a bug:
1. First write a **failing test** that reproduces the bug.
2. Explain root cause in a comment or PR description.
3. Fix.
4. Verify the previously failing test now passes.
5. Run full test suite — no regressions allowed.

---

## 10. Quick Reference: Module Responsibilities

| Module | Responsibility |
|---|---|
| `accessor.py` | pandas `.miss` extension API |
| `impute.py` | Single imputation (mean/median/mode/KNN/MICE/model-based) |
| `transformer.py` | sklearn-compatible `MissinglyImputer` |
| `diagnostics.py` | Statistical tests (MCAR/MAR/MNAR) |
| `simulate.py` | Inject missingness under different mechanisms |
| `mi.py` | Multiple imputation core |
| `mi_workflow.py` | MI pipeline: impute → analyze → pool (Rubin's rules) |
| `compare.py` | Cross-dataset / cross-group missingness comparison |
| `report.py` | HTML report generation |
| `timeseries.py` | Time-series gap analysis and interpolation |
| `visualise.py` + `visualisation/` | All plots (heatmap, matrix, bar, upset, timeline) |
| `manipulation.py` | Programmatic insertion/removal of missing values |
| `config.py` | Global settings (MissinglyConfig) |
| `exceptions.py` | Custom exception hierarchy |
| `_validation.py` | Reusable input validators |
| `_distance.py` | Distance metrics for KNN imputation |
| `_deprecation.py` | Deprecation warning decorators |
| `performance.py` | Memory/time profiling utilities |
| `stats_extra.py` | Extra stats helpers (not in scipy/statsmodels) |

---

*Last updated: 2026-08-15 | Repo: github.com/alisadeghiaghili/missingly*
