# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - Unreleased

### Added
- **Defined v1 public API surface** (`df.miss.*`, `MissinglyImputer`,
  `create_report`, core visualisation & diagnosis functions). `__all__` now
  clearly separates the stable v1 surface from experimental utilities.
- `missingly/_deprecation.py` — `@deprecated_api` decorator that emits
  `FutureWarning` on every call to an experimental / legacy function.
- `tests/test_public_api.py` — tests verifying that every v1 symbol exists
  and is callable, and that legacy functions emit `FutureWarning`.
- `tests/test_impute_mice.py` — tests for `impute_mice` single-chain and
  multi-imputation behaviour (edge cases included).
- `tests/test_impute_knn.py` — tests for `impute_knn` correctness,
  heavy-categorical safety warning, and edge cases.
- `tests/test_timeseries_summary.py` — comprehensive tests for
  `miss_ts_summary`: zero-gap case, two-gap case (n_gaps, max_gap,
  mean_gap, longest-gap boundaries), and multi-column correctness.
- `tests/test_timeseries_visuals.py` — tests for `vis_ts_miss`,
  `vis_gap_lengths` (hist + box, FutureWarning, no-gap edge case), and
  `vis_miss_over_time` (axes type, ylim, custom window, existing axes).
- `tests/test_impute_ts.py` — tests for `impute_ts` covering all five
  strategies (ffill, bfill, linear, time, spline), `limit` behaviour on
  a long gap, columns-subset logic, invalid-method error, and the
  `method='time'` + non-DatetimeIndex ValueError.
- **README:** `Time-series missingness` subsection with a full worked
  example (temperature sensor dropouts → `miss_ts_summary` →
  `vis_ts_miss` / `vis_miss_over_time` → `impute_ts`) and a strategy
  reference table.

### Changed
- Marked advanced utilities as experimental with `FutureWarning`:
  `simulate_mcar/mar/mnar/mixed`, `chunk_apply`, `memory_usage_mb`,
  `optimize_dtypes`, `hotelling_test`, `pattern_monotone_test`,
  `missing_correlation_matrix`, `clean_names`, `remove_empty`,
  `coalesce_columns`, `miss_as_feature`, `gap_table`, `vis_gap_lengths`.
- **`impute_mice`** now exposes an `n_imputations` parameter.
  - `n_imputations=1` (default) — preserves existing behaviour, returns a
    single `pd.DataFrame`. Fully backward-compatible.
  - `n_imputations > 1` — runs `n_imputations` independent
    `IterativeImputer` chains (each with a distinct random seed) and returns
    a `List[pd.DataFrame]`. No pooling (Rubin's Rules) is performed yet;
    this is groundwork for full Multiple Imputation support in a future
    release.
  - Docstring updated to clarify that the current implementation is a
    *single chained imputation* using `sklearn.impute.IterativeImputer`,
    not a statistically complete MI procedure, and to explain the path
    towards real MI (independent chains → fit models → Rubin's Rules).
- **`impute_knn`** — documented limitations and added a runtime safety warning.
  - Documented in a new **Limitations** section of the docstring that
    KNN uses Euclidean distance over ordinal-encoded integer codes, which
    is not statistically meaningful for nominal categoricals and degrades
    imputation quality when categoricals dominate the feature space.
  - Added recommendation to prefer `impute_rf()`, `impute_gb()`, or
    `impute_mice()` for heavy-categorical datasets.
  - Now emits a `UserWarning` ("KNN imputation may be unreliable") when
    the number of categorical columns exceeds the number of numeric columns
    **and** `n_neighbors > 5`. No warning is raised for the default
    `n_neighbors=5` to avoid noise in common mixed-type use cases.
  - Planned future work noted in docstring: a ``metric='mixed'`` option
    backed by Gower distance for proper mixed numeric/categorical KNN.
- **Time-series API stabilised** (`miss_ts_summary`, `vis_ts_miss`,
  `vis_miss_over_time`, `impute_ts`) with dedicated tests and documentation.
  `gap_table` and `vis_gap_lengths` remain experimental (emit `FutureWarning`).
  No Kalman / ARIMA imputation yet — planned for 0.3.0.

## [0.1.0] - 2025-01-01

### Added
- Initial release with core missing data analysis functionality.
- `df.miss` pandas accessor (registered in `accessor.py`).
- `MissinglyImputer` sklearn-compatible transformer.
- HTML reporting via Jinja2 (`create_report`).
- Time series support (`miss_ts_summary`, `vis_ts_miss`, `vis_miss_over_time`,
  `impute_ts`).
- Simulation utilities (`simulate_mcar`, `simulate_mar`, `simulate_mnar`,
  `simulate_mixed`).
- MCAR/MAR/MNAR diagnostic tests (`mcar_test`, `mar_mnar_test`,
  `diagnose_missing`).
- Core visualisations: `matrix`, `bar`, `vis_miss`, `upset`, `heatmap`,
  `miss_patterns`, `miss_cooccurrence`, `miss_cluster`.
