# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- Corrected typed MICE custom-estimator metadata to disclose ordinal encoding
  and the unvalidated categorical FCS approximation when categorical targets
  are supplied.
- Made typed MICE results disclose default numeric Bayesian-ridge and
  ordinal-encoded categorical approximation contracts, including the latter's
  multinomial/ordinal FCS limitation.
- Made MICE ``where`` and typed-result eligibility masks reject null boolean
  entries instead of accepting an ambiguous three-valued imputation policy.
- Made typed MICE results preserve ``where=False`` structural missingness
  while continuing to reject chains that leave imputation-eligible cells null.
- Made the sklearn transformer reject duplicate DataFrame column labels with
  stable, actionable errors before backend-specific validation.
- Made predictor-free transformer PMM draw reproducibly from observed training
  donors instead of producing a mean outside the donor support.
- Made ``miss_row_profile`` return an annotated axes for empty rows or columns
  instead of raising a histogram error or emitting an all-NaN runtime warning.
- Made ``DataFrame.miss.impute`` reject every unsupported strategy through
  ``InvalidStrategyError``, including unhashable Python values that previously
  leaked a mapping ``TypeError``.
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
- **Breaking:** `missingly.config.strict_mode` now defaults to `True`. A conditional
  model that fails to fit in `impute_rf`, `impute_gb`, `impute_logreg`,
  `impute_polyreg`, `impute_polr` or `impute_mice` raises `ImputationError`, naming
  the column, the method and the underlying error (chained). Pass
  `strict_mode=False` (per call or through `missingly.config`) to keep the mean/mode
  fallback; it now emits a `UserWarning` naming every column that fell back, instead
  of an INFO log record. A one-column frame has no predictors, so it counts as a
  failed fit.
- **Breaking:** `impute_mice` raises `ValueError` for `n_imputations > 1` when the
  estimator's `predict` has no `return_std` argument (for example `LinearRegression`
  or a random forest). Such an estimator returns identical imputations, so Rubin's
  between-imputation variance is zero (`df=inf`, `fmi=0`). An estimator that does
  accept `return_std`, including an explicit `BayesianRidge()`, now draws from its
  predictive distribution instead of returning mean predictions.
- `impute_mice` chain seeds are spawned with `numpy.random.SeedSequence(random_state)`
  instead of `random_state + i`. Chain `i` no longer depends on how many chains were
  requested, adjacent `random_state` values share no chain, and imputed values for a
  given seed differ from earlier unreleased builds. `random_state` may be `None`; a
  negative or non-integer value raises `ValueError`. The seeds used are recorded in
  `plan.seed_sequence` of a typed result.
- `impute_mice` runs one engine whatever optional arguments are passed (it used to
  switch between scikit-learn's `IterativeImputer` and a hand-written loop), so
  `return_history`, `return_result`, `visit_sequence`, `predictor_matrix`, `where` and
  `bounds` at their neutral values no longer change the imputed values. Custom
  estimators always run all `max_iter` sweeps (no tolerance-based early stop), and a
  cell kept missing by `where=False` is no longer used as a training row.
- `impute_rf` and `MissinglyImputer(strategy="rf")` fill incomplete predictors with
  training column means before every fit on every scikit-learn version, as the
  gradient-boosting imputers already did. Results on scikit-learn >= 1.4, which used
  to pass NaN to the forest, change slightly.
- `pool_scalar_estimates` emits a `UserWarning` when the between-imputation variance
  is exactly zero; its return values are unchanged.

### Fixed
- Removed public MAR-versus-MNAR identification claims. `mar_mnar_test` remains as a
  warning-emitting compatibility wrapper and will be removed in a future major release.
- Reporting edge cases, localized recommendations, RTL font fallback, Little's MCAR
  handling, and pandas 3-compatible UpSet rendering.
- Stale Sphinx module references, invalid NumPy-doc sections, and undeclared Pandoc
  requirements that prevented warning-free API and notebook documentation builds.
- `mcar_test` (and `diagnose_missing`) identified missingness patterns with a 64-bit
  integer code that overflowed from 64 columns, silently merging patterns and
  returning a wrong pattern count, degrees of freedom and p-value. Patterns are now
  identified exactly; results for fewer than 64 columns are unchanged.
- `impute_rf` returned the observed mean for every missing cell of a column whose
  predictors contained NaN on scikit-learn below 1.4 (inside the declared floor), and
  for every column when a predictor held `inf`, without any warning.
- `impute_rf`, `impute_gb`, `impute_logreg`, `impute_polyreg` and `impute_polr` reject
  `inf` / `-inf` and complex or timedelta columns up front with a `ValueError` that
  names the columns, instead of failing in every conditional fit.
- `impute_mice` replaced a column whose estimator failed by its mean without any
  message; it now follows `strict_mode` (new keyword argument) and a typed result
  records fallbacks in `warnings` and `validation["estimator_fallback_used"]`.

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
