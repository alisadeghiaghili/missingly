# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - Unreleased

### Added
- Defined v1 public API surface (`df.miss.*`, `MissinglyImputer`, `create_report`,
  core visualisation & diagnosis functions).
- Marked advanced utilities (`simulate_*`, performance helpers, extra statistical tests)
  as experimental with `FutureWarning` via new `@deprecated_api` decorator
  (`missingly/_deprecation.py`).
- Extended Plotly interactive mode to `bar`, `miss_case`, `upset`, and
  `miss_patterns`; static matplotlib behaviour remains unchanged.
- New test file `tests/test_visualise_interactive.py` covering all nine
  interactive-capable functions: return-type assertions, static fallback
  checks, `ImportError` checks via monkeypatching, and edge cases.
- Updated `README.md` with a full `Interactive Plots` section listing all
  nine functions that accept `interactive=True`.
- Pinned correlation heatmap implementation to
  `data_quality_toolkit.visualization.correlation_heatmap` with test
  coverage (`tests/test_dqt_integration.py`) and CI integration
  (`dqt-integration` job in `.github/workflows/ci.yml`).

## [0.1.0] - Initial release

### Added
- `df.miss.*` pandas accessor with summary, visualisation, and imputation helpers.
- `MissinglyImputer` sklearn-compatible transformer.
- Statistical tests: `mcar_test`, `mar_mnar_test`, `diagnose_missing`.
- Visualisation catalogue: `vis_miss`, `matrix`, `bar`, `miss_case`,
  `miss_var_pct`, `miss_which`, `upset`, `miss_patterns`, `miss_cooccurrence`,
  `heatmap`, `dendrogram`, `miss_cluster`.
- Interactive Plotly backends (Phase 1): `vis_miss`, `heatmap`, `matrix`,
  `miss_var_pct`, `miss_cooccurrence`.
- HTML report generation via `create_report`.
- Time-series support: `miss_ts_summary`, `vis_ts_miss`, `vis_miss_over_time`,
  `impute_ts`.
- Simulation utilities: `simulate_mcar`, `simulate_mar`, `simulate_mnar`,
  `simulate_mixed`.
