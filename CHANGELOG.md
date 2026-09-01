# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- Made predictor-free ``impute_polr`` calls use their safe observed modal
  baseline without importing ``statsmodels``. This valid no-estimator path is
  unchanged by ``strict_mode=True``.
- Reconciled the AI co-pilot context, README examples, legacy requirements,
  and repository line-ending policy with the supported public API. In
  particular, the documented Python floor is 3.9, time-series examples use
  the actual summary schema, and mixed-metric KNN is no longer presented as
  an inductive sklearn transformer mode.

- Reimplemented Little's MCAR test with an observed-data EM M-step, explicit
  convergence/error contracts, and an offline frozen `naniar` R oracle fixture.
- Prevented `MissinglyImputer.transform` from refitting on evaluation data or
  using evaluation rows as donors. Unsafe transductive transformer modes now
  fail explicitly pending a train-only implementation.
- Restored posterior stochasticity to MICE history chains and reject identical
  chain traces as invalid convergence evidence rather than reporting R-hat as
  acceptable.
- Made static visualisation offline-only: rendering no longer downloads fonts,
  writes a user font cache, or imports DQT. Empty MICE convergence plots now
  issue a warning instead of writing operational output to stdout.
- Aligned the static ``bar`` visualization with its public count contract and
  Plotly backend; bars now report missing-value counts rather than percentages.
- Corrected single-feature PMM to make deterministic random hot-deck draws
  from observed donors instead of producing an unsupported regression mean.
  PMM targets without observed donors now raise ``InsufficientDataError``.
- Made ``FittedImputer`` reject duplicate, missing, extra, and unknown schema
  labels before filling, while retaining support for reordered unique labels.
- Made public ``impute_mode`` preserve nullable extension dtypes and raise
  ``InsufficientDataError`` for all-missing columns instead of leaking backend
  failures.
- Repaired the observedness-association screen for one-feature, intercept-only
  inputs; duplicate labels and non-finite numeric outcomes now fail early, and
  recoverable logistic fit failures emit a feature-named warning. This remains
  an exploratory observed-data association screen and does not identify MAR/MNAR.

### Added
- Promoted the Little's MCAR R oracle to schema v2 with reviewed R/package,
  platform, and generator-script SHA-256 provenance from the locked workflow.
- Added a locked R 4.4.2/``naniar`` v1.1.0 workflow that regenerates and
  validates scalar Little's MCAR oracle evidence with source/data SHA-256
  identities, without publishing raw records.
- Added a protected GitHub Pages deployment workflow for the Sphinx site and
  linked the package metadata and README documentation badge to the public URL.
- Added typed, validated contracts for missingness schemas, imputation plans,
  FCS kernels, MI data, and MI diagnostic results; existing imputation APIs
  retain their return contracts pending a deliberate migration.
- Added a versioned, privacy-preserving ``BenchmarkManifest`` contract for
  reproducible cross-tool numerical evidence without raw records.
- Added a machine-readable manifest for the frozen Little's MCAR oracle; its
  inherited R-version/platform provenance gap is now explicit rather than
  being presented as parity evidence.
- Deterministic DataFrame and configured-analysis SHA-256 provenance manifests.
- Closed-form Rubin pooling conformance tests and an evidence-gated competitor roadmap.
- Branch reconciliation and statistical TDD policy documents.
- `missingness_association_test`, an explicitly limited observed-data association
  screen with named results.

### Changed

- Recorded the authoritative Python 3.12 CI coverage baseline as 89.64%
  (3589/4004) and its `coverage-py312` artifact, without changing the
  separately tracked coverage-enforcement target.

- `diagnose_missing` now reports MCAR evidence without claiming to identify MAR or
  MNAR. Report recommendations require documented assumptions and sensitivity analysis.
- Scalar MI pooling now distinguishes `lambda` from small-sample `fmi` and rejects
  non-finite estimates or invalid within-imputation variances.
- Mixed KNN uses Missingly's internal Gower implementation rather than an external
  runtime package.
- Runtime and distribution versions now share one metadata source.
- Build metadata uses the SPDX license format and explicit package-data configuration,
  eliminating setuptools deprecation and ambiguous-package warnings.

### Fixed
- Removed public MAR-versus-MNAR identification claims. `mar_mnar_test` remains as a
  warning-emitting compatibility wrapper and will be removed in a future major release.
- Reporting edge cases, localized recommendations, RTL font fallback, Little's MCAR
  handling, and pandas 3-compatible UpSet rendering.
- Stale Sphinx module references, invalid NumPy-doc sections, and undeclared Pandoc
  requirements that prevented warning-free API and notebook documentation builds.

### Removed
- Placeholder-only test modules that reported skipped work without testing behavior.

## [1.0.0] - 2026-07-29

### Added
- **ReadTheDocs deployment**: `.readthedocs.yaml` added; live docs at
  https://missingly.readthedocs.io
- `Documentation` URL added to `pyproject.toml`.
- `numpydoc` added to `[docs]` optional dependencies.

### Changed
- `docs/source/conf.py`: version bumped to `1.0.0`, `numpydoc` + `napoleon`
  extensions enabled, `intersphinx` expanded to pandas/numpy/sklearn/scipy/matplotlib,
  `autodoc_mock_imports` added for optional deps, copyright year updated to 2026.
- Public API frozen: all modules expose `__all__`; breaking changes now
  require a major version bump.

### Fixed
- `conf.py` version was stale at `0.1.0` — now reads `1.0.0` / `1.0`.

## [0.3.0] - 2026-07-22

### Changed
- Refactored visualisation layer into structured modules (`visualisation.static`,
  `visualisation.interactive`) and improved documentation/examples for all public plots.
- `missingly/visualise.py` is now a thin re-export facade; all implementation lives
  in `missingly/visualisation/`.
- Smoke test suite (`tests/test_visualise_smoke.py`) extended to cover all public
  visualisation functions including sentinel `missing_values`, shadow scatter,
  parallel coordinates, span, group breakdown, and imputation diagnostics.

## [0.2.0] - 2026-04-15

### Added
- **KNN with Gower distance** (`metric="mixed"`) for `impute_knn` and
  `MissinglyImputer(strategy="knn", metric="mixed")`: statistically sound
  imputation for mixed numeric + categorical datasets.
- `impute_knn` now selects the distance metric via the `metric` parameter
  (`"euclidean"` default, `"mixed"` for Gower).
- `MissinglyImputer` accepts `metric` and forwards it to `impute_knn`.
- New tests: `test_impute_knn_mixed.py` — covers Gower distance, edge cases
  (all-numeric, all-categorical, single-column), and sklearn `Pipeline`
  compatibility.

### Fixed
- `impute_knn` no longer crashes on DataFrames that contain only categorical
  columns when `metric="euclidean"` (ordinal encoding is applied first).

## [0.1.0] - 2025-01-10

### Added
- Initial public release.
- `df.miss.*` accessor (mirrors R `naniar`).
- `MissinglyImputer` sklearn-compatible transformer.
- Visualisation module: `matrix`, `vis_miss`, `bar`, `miss_case`,
  `miss_var_pct`, `miss_patterns`, `miss_cooccurrence`, `heatmap`,
  `upset`, `miss_cluster`, `dendrogram`, `shadow_scatter`,
  `vis_miss_fct`, `vis_miss_by_group`, `vis_impute_dist`,
  `miss_impute_compare`, `scatter_miss`, `vis_miss_cumsum_var`,
  `vis_miss_cumsum_case`, `vis_miss_span`, `vis_parallel_coords`.
- Statistical tests: `mcar_test`, `mar_mnar_test`, `diagnose_missing`.
- Time-series support: `miss_ts_summary`, `vis_ts_miss`, `impute_ts`, `gap_table`.
- Multiple imputation: `impute_mice` with Rubin's Rules pooling utilities.
- HTML report generation: `create_report`.
- `data_quality_toolkit` integration: `heatmap` delegates to
  `data_quality_toolkit.visualization.correlation_heatmap` when available.
