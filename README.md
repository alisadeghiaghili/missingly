# missingly

<p align="center">
  <a href="https://pypi.org/project/missingly/"><img src="https://badge.fury.io/py/missingly.svg" alt="PyPI version"></a>
  <a href="https://github.com/alisadeghiaghili/missingly/actions/workflows/ci.yml"><img src="https://github.com/alisadeghiaghili/missingly/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/alisadeghiaghili/missingly/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <a href="https://github.com/alisadeghiaghili/missingly"><img src="https://img.shields.io/github/stars/alisadeghiaghili/missingly?style=social" alt="GitHub stars"></a>
  <a href="https://pepy.tech/project/missingly"><img src="https://pepy.tech/badge/missingly" alt="Downloads"></a>
</p>

**A comprehensive Python package for missing data analysis, visualization, and imputation.**

`missingly` provides an intuitive and powerful interface for exploring, analyzing, and handling missing data in pandas DataFrames. Inspired by the R `naniar` package, it brings together the best practices from data science and statistical communities into a cohesive Python toolkit.

---

## Table of Contents

- [Why missingly?](#why-missingly)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Pandas Chaining API](#pandas-chaining-api)
- [Comprehensive Feature Overview](#comprehensive-feature-overview)
- [Advanced Usage Examples](#advanced-usage-examples)
- [API Reference](#api-reference)
- [Performance Considerations](#performance-considerations)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)
- [About the Author](#about-the-author)

---

## Why missingly?

Missing data is ubiquitous in real-world datasets and handling it poorly can lead to biased results, reduced statistical power, and incorrect conclusions. Traditional approaches often involve ad-hoc deletion or simple imputation without understanding the underlying patterns of missingness.

`missingly` addresses these challenges by providing:

- **Comprehensive Analysis**: Statistical tests to understand missing data mechanisms (MCAR, MAR, MNAR)
- **Rich Visualizations**: Over 10 specialized plots to explore missing data patterns
- **Multiple Imputation Methods**: From simple mean imputation to advanced machine learning approaches — all supporting mixed numeric/categorical DataFrames
- **Automated Reporting**: Generate publication-ready HTML reports
- **Pandas Integration**: Seamless workflow via both functional API and the `df.miss.*` chaining accessor

## Key Features

### 🔍 **Missing Data Analysis**
- Statistical summaries by variable and observation
- Little's MCAR test for missing data mechanisms
- MAR vs MNAR likelihood ratio tests
- Custom missing value indicators support

### 📊 **Rich Visualizations**
- **Matrix plots**: Visualize missing data patterns across your dataset
- **Bar charts**: Missing value counts per variable/observation
- **Upset plots**: Intersection patterns of missing values
- **Dendrograms**: Hierarchical clustering of variables by missingness patterns
- **Scatter plots**: Relationship visualization with missing value highlighting
- **Distribution comparisons**: Before/after imputation analysis

### 🔧 **Advanced Imputation Methods**
- **Simple methods**: Mean, median, mode — all categorical-aware
- **Machine learning**: K-NN, Random Forest, Gradient Boosting — auto-encode/decode categoricals
- **Statistical**: Multiple Imputation by Chained Equations (MICE) via `sklearn` `IterativeImputer` + `BayesianRidge`
- **Comparison tools**: Evaluate and compare imputation performance across methods

### 🔗 **Pandas Chaining API**
- `df.miss.*` accessor exposes every function as a chainable method
- Manipulation and imputation methods return DataFrames for fluent pipelines
- Zero config — just `import missingly` to activate

### 📈 **Automated Reporting**
- Comprehensive HTML reports with embedded visualizations
- Customizable templates
- Summary statistics and recommendations

## Installation

Install `missingly` via pip:

```bash
pip install missingly
```

**Requirements**: Python 3.9+, pandas ≥ 2.0, numpy ≥ 1.24, and standard data-science stack (matplotlib, seaborn, scipy, scikit-learn, statsmodels, jinja2).

## Quick Start

```python
import pandas as pd
import numpy as np
import missingly as mi

# Mixed numeric + categorical dataset — no pre-processing needed
data = {
    'age':       [25, 30, np.nan, 45, 35, np.nan, 28],
    'income':    [50000, np.nan, 75000, np.nan, 65000, 80000, 55000],
    'education': ['HS', 'College', 'Graduate', np.nan, 'College', 'Graduate', 'HS'],
    'score':     [85, 92, np.nan, 78, np.nan, 95, 88]
}
df = pd.DataFrame(data)

# Quick overview
print(f"Missing values: {mi.n_miss(df)} ({mi.pct_miss(df):.1f}%)")
print(mi.miss_var_summary(df))

# Visualize missing patterns
mi.vis_miss(df)
mi.upset(df)

# Test missing data mechanisms
result = mi.mcar_test(df)
print(f"MCAR p-value: {result['p_value']:.4f}")

# Impute
df_imputed = mi.impute_knn(df, n_neighbors=3)

# Report
mi.create_report(df, "missing_data_analysis.html")
```

## Pandas Chaining API

After `import missingly`, every DataFrame gains a `.miss` accessor. Manipulation and imputation methods return a new DataFrame so calls chain naturally:

```python
import missingly  # registers df.miss — nothing else needed

# Full pipeline in one expression
result = (
    df
    .miss.replace_with_na({'age': -99, 'income': -1})
    .miss.remove_empty(thresh_col=0.9)
    .miss.coalesce_columns('income', 'fallback_income')
    .miss.clean_names()
    .miss.miss_as_feature()          # adds binary _NA indicator columns
    .miss.impute_mean()              # returns fully-observed DataFrame
)

# Summary — returns scalars / DataFrames, not chainable
df.miss.n_miss()                     # → int
df.miss.pct_miss()                   # → float
df.miss.miss_var_summary()           # → DataFrame
df.miss.bind_shadow()                # → DataFrame (doubled columns)

# Visualisation — returns Axes
df.miss.vis_miss()
df.miss.heatmap()
df.miss.miss_var_pct()
```

### Accessor method reference

| Category | Methods | Returns |
|---|---|---|
| Manipulation | `replace_with_na`, `replace_with_na_all`, `clean_names`, `remove_empty`, `coalesce_columns`, `miss_as_feature` | `DataFrame` (chainable) |
| Imputation | `impute_mean`, `impute_median`, `impute_mode`, `impute_knn`, `impute_mice`, `impute_rf`, `impute_gb` | `DataFrame` (chainable) |
| Summary | `n_miss`, `n_complete`, `pct_miss`, `pct_complete`, `miss_var_summary`, `miss_case_summary`, `bind_shadow` | scalar / `DataFrame` |
| Stats | `mcar_test`, `mar_mnar_test` | dict / `DataFrame` |
| Visualisation | `matrix`, `bar`, `upset`, `heatmap`, `vis_miss`, `miss_var_pct`, `miss_cluster`, `miss_which`, `scatter_miss`, `miss_case`, `vis_impute_dist`, `vis_miss_fct`, `vis_miss_cumsum_var`, `vis_miss_cumsum_case`, `vis_miss_span`, `vis_parallel_coords`, `dendrogram` | `Axes` |

## Comprehensive Feature Overview

### Statistical Analysis Functions

| Function | Description | Use Case |
|----------|-------------|----------|
| `n_miss()` | Count total missing values | Quick dataset overview |
| `pct_miss()` | Percentage of missing values | Assess missingness severity |
| `miss_var_summary()` | Per-variable missing statistics | Identify problematic variables |
| `miss_case_summary()` | Per-observation missing statistics | Find incomplete observations |
| `mcar_test()` | Little's MCAR test | Test randomness of missingness |
| `mar_mnar_test()` | MAR vs MNAR likelihood test | Distinguish missing mechanisms |

### Visualization Functions

| Function | Description | Best For |
|----------|-------------|----------|
| `vis_miss()` | Annotated missingness overview | Default first look |
| `matrix()` | Heatmap of missing patterns | Overall pattern visualization |
| `bar()` | Bar chart of missing counts per variable | Variable comparison |
| `miss_case()` | Bar chart of missing counts per row | Case-wise analysis |
| `dendrogram()` | Hierarchical clustering of variables | Variable grouping by missingness |
| `upset()` | Intersection plot of missing combinations | Complex pattern analysis |
| `scatter_miss()` | Scatterplot highlighting missing values | Bivariate relationships |
| `vis_impute_dist()` | Distribution comparison (before/after) | Imputation quality assessment |
| `heatmap()` | Nullity-correlation heatmap | Correlation of missingness |
| `miss_var_pct()` | % missing per variable (horizontal bar) | Quick variable ranking |
| `miss_cluster()` | Clustered missingness heatmap | Pattern grouping |
| `miss_which()` | Binary tile plot per column | Which columns have any missing |

### Data Manipulation Functions

| Function | Description |
|---|---|
| `replace_with_na(df, replace)` | Replace specific values with NaN |
| `replace_with_na_all(df, condition)` | Replace all values matching a predicate with NaN |
| `clean_names(df)` | Normalise column names to snake_case |
| `remove_empty(df, ...)` | Drop fully- or mostly-empty rows/columns |
| `coalesce_columns(df, target, *donors)` | SQL-style COALESCE across columns |
| `miss_as_feature(df, ...)` | Add binary missingness indicator columns |

### Imputation Methods

All imputation methods support mixed numeric/categorical DataFrames. Categorical columns are automatically ordinal-encoded before imputation and decoded back to their original categories.

| Method | Function | Strengths | Notes |
|--------|----------|-----------|-------|
| Mean | `impute_mean()` | Fast, simple | Categoricals filled with mode |
| Median | `impute_median()` | Robust to outliers | Categoricals filled with mode |
| Mode | `impute_mode()` | Works with all dtypes | Best for categorical-heavy data |
| K-NN | `impute_knn()` | Preserves local patterns | O(n²) — slow on large datasets |
| MICE | `impute_mice()` | Statistically rigorous | Uses BayesianRidge by default |
| Random Forest | `impute_rf()` | Handles non-linearity | Slower; accepts `**rf_kwargs` |
| Gradient Boosting | `impute_gb()` | High accuracy potential | Slower; accepts `**gb_kwargs` |

## Advanced Usage Examples

### Custom Missing Value Indicators

```python
# Specify them directly in functions
summary = mi.miss_var_summary(df, missing_values=[999, -99, 'unknown'])
mi.matrix(df, missing_values=[999, -99])
```

### Missing Data Mechanism Testing

```python
mcar_result = mi.mcar_test(df)
print(f"Chi-square: {mcar_result['chi_square']:.4f}")
print(f"p-value: {mcar_result['p_value']:.4f}")

if mcar_result['p_value'] > 0.05:
    print("Data appears to be MCAR")
else:
    print("Data is not MCAR — patterns in missingness detected")
```

### Imputation Comparison

```python
complete_df = df.dropna()
comparison = mi.compare_imputations(complete_df)
print(comparison)
best_method = comparison.index[0]
```

### Custom MICE Estimator

```python
from sklearn.ensemble import RandomForestRegressor
df_imputed = mi.impute_mice(df, estimator=RandomForestRegressor(n_estimators=50), max_iter=5)
```

### Dashboard

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
mi.matrix(df, ax=axes[0, 0])
mi.bar(df, ax=axes[0, 1])
mi.dendrogram(df, ax=axes[1, 0])
mi.miss_case(df, ax=axes[1, 1])
plt.tight_layout()
plt.show()
```

## API Reference

### Summary Statistics
- `n_miss(df, missing_values=None)` → int
- `n_complete(df, missing_values=None)` → int
- `pct_miss(df, missing_values=None)` → float
- `pct_complete(df, missing_values=None)` → float
- `miss_var_summary(df, missing_values=None)` → DataFrame
- `miss_case_summary(df, missing_values=None)` → DataFrame
- `bind_shadow(df, missing_values=None)` → DataFrame

### Statistical Tests
- `mcar_test(df)` → dict
- `mar_mnar_test(X, Y)` → list

### Data Manipulation
- `replace_with_na(df, replace)` → DataFrame
- `replace_with_na_all(df, condition)` → DataFrame
- `clean_names(df, case='lower', sep='_')` → DataFrame
- `remove_empty(df, axis='both', thresh_col=None, thresh_row=None)` → DataFrame
- `coalesce_columns(df, target, *donors, remove_donors=False)` → DataFrame
- `miss_as_feature(df, columns=None, suffix='_NA', keep_original=True)` → DataFrame

### Imputation
- `impute_mean(df)` → DataFrame
- `impute_median(df)` → DataFrame
- `impute_mode(df)` → DataFrame
- `impute_knn(df, n_neighbors=5)` → DataFrame
- `impute_mice(df, max_iter=10, random_state=0, estimator=None)` → DataFrame
- `impute_rf(df, max_iter=10, random_state=0, **rf_kwargs)` → DataFrame
- `impute_gb(df, max_iter=10, random_state=0, **gb_kwargs)` → DataFrame

### Utilities
- `compare_imputations(df, methods=None)` → DataFrame
- `create_report(df, output_path='missing_data_report.html')` → None

## Performance Considerations

- **Large datasets**: For datasets >100k rows, consider sampling for visualization
- **Memory usage**: MICE and tree-based imputation methods require more memory
- **Computational complexity**: K-NN imputation is O(n²) — consider RF or MICE for very large datasets
- **Categorical encoding**: ML-based imputers internally encode/decode categoricals via `OrdinalEncoder` — no manual pre-processing required
- **Statistical tests**: MCAR test performance degrades with >50% missingness
- **pandas version**: missingly requires pandas ≥ 2.0 — the `DataFrame.map()` API is used throughout

## Contributing

We welcome contributions! Here's how you can help:

1. **Bug reports**: Open an issue with reproducible examples
2. **Feature requests**: Propose new functionality with use cases
3. **Code contributions**: Fork, develop, test, and submit pull requests
4. **Documentation**: Improve examples, docstrings, and guides

See our [Contributing Guidelines](CONTRIBUTING.md) for detailed information.

### Development Setup

```bash
git clone https://github.com/alisadeghiaghili/missingly.git
cd missingly
pip install -e .[test]
pytest tests/ -v
```

## Citation

If you use `missingly` in your research, please cite:

```bibtex
@software{missingly,
  author = {Sadeghi Aghili, Ali},
  title = {missingly: Comprehensive Missing Data Analysis for Python},
  url = {https://github.com/alisadeghiaghili/missingly},
  version = {0.1.0},
  year = {2025}
}
```

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

**Note on Dependencies**: The MCAR test implementation is adapted from the XeroGraph library by Julhash Kazi, licensed under Apache License 2.0. See [XeroGraph_LICENSE](XeroGraph_LICENSE) for details.

## About the Author

**Ali Sadeghi Aghili** is a data scientist and software developer passionate about making statistical methods accessible to the broader data science community.

- **LinkedIn**: [https://www.linkedin.com/in/ali-sadeghi-aghili/](https://www.linkedin.com/in/ali-sadeghi-aghili/)
- **GitHub**: [https://github.com/alisadeghiaghili](https://github.com/alisadeghiaghili)
- **Email**: alisadeghiaghili@gmail.com

---

<p align="center">
  <strong>⭐ Star this repository if you find it useful!</strong><br>
  <em>Help others discover missingly by starring the project on GitHub</em>
</p>

---

*Built with ❤️ for the Python data science community*
