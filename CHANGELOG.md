# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

### Documentation
- Complete API documentation
- Usage examples and tutorials
- Contributing guidelines
- MIT License

### Attribution
- Little's MCAR test implementation adapted from XeroGraph library by Julhash Kazi (Apache License 2.0)
- Package inspired by R's naniar package

## Future Releases

### Planned for 0.2.0
- [ ] Additional visualization options
- [ ] More sophisticated missing data pattern analysis
- [ ] Integration with more imputation libraries
- [ ] Performance optimizations for large datasets
- [ ] Additional statistical tests for missing data mechanisms

### Planned for 0.3.0
- [ ] Interactive visualizations with plotly
- [ ] Time series missing data analysis
- [ ] Missing data simulation utilities
- [ ] Advanced reporting with customizable templates

---

**Note**: This project follows semantic versioning. Given the initial release nature:
- MAJOR version increments for incompatible API changes
- MINOR version increments for backwards-compatible functionality additions  
- PATCH version increments for backwards-compatible bug fixes
