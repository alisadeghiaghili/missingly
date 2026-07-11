---
name: missingly
description: AI co-pilot context for the missingly Python package — missing data analysis, imputation, diagnostics, simulation, and reporting.
version: 1.0.0
author: Ali Sadeghi Aghili
repo: github.com/alisadeghiaghili/missingly
updated: 2026-07-11
---

# SKILL: missingly — AI Co-Pilot Context File

> **Caveman Skill**: This file is the ground truth for any AI assistant working on the `missingly` package.
> Read this before writing a single line of code. No exceptions.

---

## 1. What Is missingly?

`missingly` is a production-grade Python package for **missing data analysis, imputation, diagnostics, simulation, and reporting**. It is built on top of `pandas`, `numpy`, `scipy`, `scikit-learn`, `statsmodels`, `matplotlib`, `seaborn`, `jinja2`, and `upsetplot`.

- **Target Python**: 3.8+
- **Target scale**: datasets up to 10M rows on a single machine
- **Design philosophy**: pandas-native API, sklearn-compatible transformers, sensible defaults, loud failures
- **Future scalability**: designed for Dask/Polars integration (not yet implemented)

---

## 2. Package Structure

```
missingly/
├── __init__.py           # Public API surface — imports and __all__
├── accessor.py           # pandas .miss accessor (df.miss.summary(), etc.)
├── compare.py            # Compare missingness across datasets/groups
├── config.py             # Global configuration (MissinglyConfig)
├── diagnostics.py        # MCAR/MAR/MNAR tests, Little's test, etc.
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
requirements.txt          # pinned dependencies
```

---

## 3. Core API

### 3.1 pandas Accessor — `df.miss`

Registered via `@pd.api.extensions.register_dataframe_accessor("miss")` in `accessor.py`.

```python
import pandas as pd
import missingly  # registers .miss accessor

df.miss.summary()          # per-column missing count, %, dtype
df.miss.heatmap()          # correlation heatmap of missingness
df.miss.matrix()           # missingno-style matrix plot
df.miss.bar()              # bar chart of missing %
df.miss.upset()            # UpSet plot of co-occurrence patterns
df.miss.report()           # generate HTML report
df.miss.compare(df2)       # compare missingness between two DataFrames
```

### 3.2 Imputation — `missingly.impute` + `MissinglyImputer`

```python
from missingly.impute import impute_simple, impute_knn, impute_iterative
from missingly.transformer import MissinglyImputer  # sklearn API

# Functional
df_imputed = impute_simple(df, strategy="mean")
df_imputed = impute_knn(df, n_neighbors=5)

# Transformer (sklearn Pipeline compatible)
imp = MissinglyImputer(method="knn", n_neighbors=5)
imp.fit(X_train)
X_imputed = imp.transform(X_test)
```

### 3.3 Diagnostics — `missingly.diagnostics`

```python
from missingly.diagnostics import (
    test_mcar,          # Little's MCAR test
    test_mar,           # logistic regression-based MAR indicator
    missing_pattern,    # returns pattern matrix
    missing_mechanism,  # heuristic classification: MCAR/MAR/MNAR
)

result = test_mcar(df)
result.statistic   # chi-square statistic
result.p_value     # p-value (< 0.05 → reject MCAR)
```

### 3.4 Simulation — `missingly.simulate`

```python
from missingly.simulate import simulate_mcar, simulate_mar, simulate_mnar

df_with_missing = simulate_mcar(df, missing_rate=0.2, random_state=42)
df_with_missing = simulate_mar(df, target_col="y", predictor_col="x", missing_rate=0.3)
df_with_missing = simulate_mnar(df, target_col="y", threshold=0.5, missing_rate=0.25)
```

### 3.5 Multiple Imputation — `missingly.mi`

```python
from missingly.mi import MultipleImputer
from missingly.mi_workflow import MIWorkflow

mi = MultipleImputer(m=5, method="mice", random_state=42)
imputed_datasets = mi.fit_transform(df)  # returns list of m DataFrames

workflow = MIWorkflow(imputer=mi)
workflow.fit(df)
results = workflow.analyze(analysis_func=my_model_fn)
pooled = workflow.pool(results)  # Rubin's rules
```

### 3.6 Compare — `missingly.compare`

```python
from missingly.compare import compare_datasets, compare_groups

summary = compare_datasets(df_before, df_after)
group_summary = compare_groups(df, group_col="treatment")
```

### 3.7 Report — `missingly.report`

```python
from missingly.report import MissinglyReport

report = MissinglyReport(df)
report.generate(output_path="missing_report.html")
```

### 3.8 Time Series — `missingly.timeseries`

```python
from missingly.timeseries import TimeSeriesMissingnessAnalyzer

ts = TimeSeriesMissingnessAnalyzer(df, time_col="date")
ts.summary()
ts.plot_gaps()
ts.interpolate(method="linear")
```

---

## 4. Exceptions

All exceptions are in `missingly/exceptions.py`. Use them — never use bare `ValueError` without actionable message.

| Exception | When to use |
|---|---|
| `MissinglyError` | Base exception |
| `InsufficientDataError` | Not enough rows/cols for operation |
| `InvalidMethodError` | Unknown imputation/test method |
| `ColumnNotFoundError` | Referenced column missing from DataFrame |
| `MixedDtypeError` | Operation requires uniform dtype |
| `ConfigurationError` | Invalid config values |

---

## 5. Conventions (MUST FOLLOW)

Full details in `CONVENTIONS.md`. Summary:

- **Docstrings**: NumPy style. Every public function must have: one-line summary, Parameters, Returns, Examples, Notes (if stats-heavy).
- **PEP 8**: enforced. Line length 88 (black).
- **Type hints**: required on all public functions.
- **Validation**: use `_validation.py` utilities. Validate at the earliest possible point.
- **No silent failures**: if data is bad, raise a `MissinglyError` subclass with a clear message.
- **No new dependencies** without explicit discussion. Current deps: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`, `scikit-learn`, `statsmodels`, `jinja2`, `upsetplot`.

---

## 6. Testing Rules

- Test file location: `tests/`
- Shared fixtures: `conftest.py` in root
- Framework: `pytest`
- Coverage target: **90%+**
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

- Functions operating on > 1M rows: **profile memory** before shipping.
- Visualisations with > 1M rows: **implement intelligent sampling** (e.g., stratified 10k-row sample with a `sample_size` param).
- Use `performance.py` profilers for benchmarking.
- Chunking is preferred over loading full data for `report.py` and `diagnostics.py` on large datasets.

---

## 8. Known Bugs & Critical Weak Spots

1. **Categorical imputation**: `impute.py` — categorical columns sometimes cast to `object` after imputation. Always preserve `pd.CategoricalDtype`.
2. **`compare.py` metrics**: some edge cases return `NaN` metrics when one dataset has zero missing values. Add a guard.
3. **`MissinglyImputer.transform()`**: does not validate that columns seen in `fit()` exist in `transform()` input. Add column consistency check.
4. **`test_mcar()`**: fails silently on DataFrames with all-complete columns (no missing at all). Should return a clear result, not raise or hang.

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
| `impute.py` | Single imputation (mean/median/mode/knn/iterative) |
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

*Last updated: 2026-07-11 | Repo: github.com/alisadeghiaghili/missingly*
