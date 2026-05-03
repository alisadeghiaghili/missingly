# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `coalesce_columns()` — SQL-style COALESCE across columns; fills target
  column from donor columns in priority order
- `clean_names()` — normalise column names to `snake_case` (or `upper` /
  `snake` variants); deduplicates collisions; preserves Persian/Unicode
- `remove_empty()` — drop fully- or mostly-empty rows/columns with
  `thresh_row` / `thresh_col` fraction thresholds and sentinel support
- `miss_as_feature()` — add binary `_NA` indicator columns for each column
  with missing values; configurable suffix, sentinel support, column ordering
- **Pandas chaining accessor** (`df.miss.*`) — every public missingly
  function exposed as a method; manipulation/imputation methods return
  `DataFrame` for fluent pipelines; registered automatically on `import missingly`

### Fixed
- `test_coalesce_does_not_mutate` — replaced broken `list.__eq__` NaN
  comparison with `np.testing.assert_array_equal` (IEEE 754: `nan != nan`)

### Changed
- **Minimum pandas version bumped to 2.0** (was 1.0).  The `DataFrame.map()`
  API used in manipulation functions requires pandas ≥ 2.0.
- Minimum versions of numpy (1.24), matplotlib (3.7), seaborn (0.12),
  scipy (1.10), scikit-learn (1.2), statsmodels (0.14), and jinja2 (3.1)
  aligned with the pandas 2.0 era.
- `pyproject.toml` classifiers now include Python 3.14.

## [0.1.0] - 2025-09-12

### Added
- Initial release of missingly package
- **Summary Functions**:
  - `bind_shadow()` - Bind shadow matrix to dataframe
  - `n_miss()` - Count total missing values
  - `n_complete()` - Count total non-missing values
  - `pct_miss()` - Calculate percentage of missing values
  - `pct_complete()` - Calculate percentage of non-missing values
  - `miss_var_summary()` - Summarize missingness by variable
  - `miss_case_summary()` - Summarize missingness by case (row)

- **Visualization Functions**:
  - `matrix()` - Nullity matrix plot
  - `bar()` - Bar plot of missing values per column
  - `miss_case()` - Bar plot of missing values per row
  - `dendrogram()` - Dendrogram to cluster variables by their nullity correlation
  - `upset()` - Upset plot for visualizing combinations of missingness
  - `scatter_miss()` - Scatter plot highlighting missing values
  - `vis_impute_dist()` - Compare distributions before and after imputation
  - `vis_miss_fct()` - Visualize missingness by factor variable
  - `vis_miss_cumsum_var()` - Cumulative sum of missing values per variable
  - `vis_miss_cumsum_case()` - Cumulative sum of missing values per case
  - `vis_miss_span()` - Missing values in repeating spans
  - `vis_parallel_coords()` - Parallel coordinates plot of missingness patterns
  - `heatmap()` - Nullity-correlation heatmap
  - `vis_miss()` - Annotated missingness overview matrix
  - `miss_var_pct()` - Horizontal bar chart of missingness % per variable
  - `miss_cluster()` - Clustered missingness heatmap
  - `miss_which()` - Binary tile plot showing which columns have missing data

- **Statistical Tests**:
  - `mcar_test()` - Little's MCAR test (adapted from XeroGraph library)
  - `mar_mnar_test()` - Test to distinguish between MAR and MNAR

- **Data Manipulation**:
  - `replace_with_na()` - Replace specified values with NA
  - `replace_with_na_all()` - Replace values meeting a condition with NA

- **Imputation Methods**:
  - `impute_mean()` - Mean imputation
  - `impute_median()` - Median imputation
  - `impute_mode()` - Mode imputation
  - `impute_knn()` - K-Nearest Neighbors imputation
  - `impute_mice()` - Multiple Imputation by Chained Equations
  - `impute_rf()` - Random Forest imputation
  - `impute_gb()` - Gradient Boosting imputation

- **Reporting**:
  - `create_report()` - Generate comprehensive HTML reports
  - `compare_imputations()` - Compare performance of different imputation methods

- **Core Features**:
  - Support for custom missing value indicators (e.g., -99, "N/A")
  - Pandas DataFrame integration
  - Matplotlib/Seaborn visualization backend
  - Comprehensive test suite
  - Sphinx documentation structure

### Dependencies
- pandas >= 1.0.0
- numpy >= 1.18.0
- matplotlib >= 3.0.0
- seaborn >= 0.10.0
- scipy >= 1.5.0
- upsetplot >= 0.6.0
- scikit-learn >= 1.0.0
- statsmodels >= 0.12.0
- jinja2 >= 3.0.0

### Attribution
- Little's MCAR test implementation adapted from XeroGraph library by Julhash Kazi (Apache License 2.0)
- Package inspired by R's naniar package

## Future Releases

### Planned for 0.2.0
- [ ] Interactive visualizations with plotly
- [ ] Time series missing data analysis
- [ ] Missing data simulation utilities
- [ ] Performance optimizations for large datasets
- [ ] Additional statistical tests for missing data mechanisms
