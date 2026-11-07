# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Removed local working notes from the repository root; project rules now live in
  CONVENTIONS.md and CONTRIBUTING.md; documentation states the enforced coverage
  gate from pyproject.toml instead of a pinned measured figure (HYG-01, HYG-02,
  F-326, F-197).
- Reworded historical notes in the changelog, conventions and test docstrings so
  they describe project facts neutrally; no entries were removed (HYG-03).
- Raised the dependency floors to what the code uses: ``pandas>=2.1.0``
  (``DataFrame.map``) and ``scikit-learn>=1.4.0`` (random-forest imputation
  with missing feature values; ``OrdinalEncoder(encoded_missing_value=...)``
  alone needs 1.1). ``requirements.txt`` now mirrors the ``pyproject.toml``
  floors exactly and no longer carries a stray extras hint, and the module
  notes in ``compare.py``, ``performance.py`` and ``stats_extra.py`` state the
  pandas 2.1 minimum (WP-01; F-154, F-195, F-196, F-314, F-330, F-384).
- CI quality gates: the R-oracle workflow also runs when ``missingly/**``
  changes (pull requests and pushes to ``main``); the coverage threshold is
  now a single ``fail_under = 90`` in ``pyproject.toml`` with branch coverage
  enabled, replacing the 80% gate in ``ci.yml``; fixture CSVs are no longer
  git-ignored by a blanket ``*.csv`` rule; ``actionlint`` lints the workflows
  in the quality job; and third-party actions in ``ci.yml``, ``r-oracle.yml``
  and ``docs-pages.yml`` are pinned to commit SHAs (WP-02; F-062, F-319,
  F-320).

### Fixed

- Fixed the ``df.miss`` accessor (WP-07): ``dendrogram()`` and
  ``vis_miss_fct()`` no longer always raise ``TypeError`` (the accessor passed
  ``method='ward'`` into scipy and ``ax`` twice; F-005, F-006, F-047, F-048),
  ``vis_impute_dist()`` takes the required ``column`` (F-072) and
  ``vis_miss_span()`` accepts the default window, ``span_every`` and ``span``
  (F-209, F-210). Accessor methods now mirror the signature of the function
  they wrap, so parameters such as ``missing_values`` and the ``mcar_test``
  tuning arguments (``max_iter``, ``tol``, ``ridge``) are forwarded (F-205);
  every v1 positional call form (for example ``df.miss.matrix(ax)``) keeps
  working. ``DataFrame.miss.mar_mnar_test`` now raises for a partly
  missing target exactly like ``missingness_association_test`` instead of
  silently dropping rows (F-070, F-206, F-207). ``dendrogram`` gained an
  honoured ``method`` linkage argument (default ``"average"``, the previous
  module-function behaviour).
- Deprecated the accessor arguments that were silently ignored
  (``vis_miss(show_pct=...)`` for the matplotlib backend and
  ``miss_cluster(method=...)``): they now emit a ``FutureWarning`` (F-071,
  F-208). The accessor documentation no longer claims to expose every public
  function and lists what it omits (F-204). A signature-parity test and a
  contract test covering every accessor wrapper guard against future drift.
- R-oracle validators now enforce their own identities. The Little's MCAR validator
  hashes its generator script and compares it with the manifest, and checks the naniar
  version and the R version and platform. The scoped mice comparison rejects an empty
  tolerance set, computes the CSV SHA-256 itself instead of trusting a literal, and
  requires the Missingly subject version to match the running package. The mice README
  now states that the scoped Missingly comparison is run (WP-03; F-309, F-310, F-311,
  F-312).
- Importing missingly no longer switches the global matplotlib backend to Agg (WP-06, F-035).
- Restored the packaging metadata lost when the license field was updated:
  the ``docs``, ``test``, ``quality``, ``polars`` and ``upset`` extras that the
  CI and documentation workflows install, the dynamic version source, the
  explicit package list, the ruff and coverage configuration, the Python 3.13
  and 3.14 classifiers and the previous dependency floors (for example
  ``pandas>=2.0.0``). The license remains ``BUSL-1.1``; the redundant
  ``License ::`` classifier is removed because PEP 639 does not allow it next
  to a license expression. The unused ``dev`` extra introduced by the same
  edit is removed in favour of ``test`` and ``quality``. A packaging contract
  test now fails if a workflow installs an undeclared extra or the version
  source, tool configuration or Python classifiers drift.
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
- Reconciled the contributor guidance, README examples, legacy requirements,
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
- ``XeroGraph_LICENSE`` (the Apache-2.0 text for the adapted XeroGraph code)
  is now listed in ``license-files`` and ships in the wheel and the sdist next
  to ``LICENSE`` and ``NOTICE``. The sdist no longer contains a partial
  ``tests/`` tree that could not run without ``conftest.py`` and the test
  fixtures (``MANIFEST.in`` prunes it); run the tests from a git checkout. A
  packaging contract test builds both artefacts and checks their contents;
  the ``test`` extra therefore installs ``build`` and ``setuptools>=77.0.0``
  (WP-01; F-153, F-383).

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

- Hardened the release workflow: publishing now depends on a job that runs the
  test suite, builds the distributions once and fails unless the pushed tag,
  ``missingly.__version__`` and the built wheel version are identical. PyPI
  publishing uses trusted publishing from a dedicated job (the only one with
  ``id-token: write``) in the ``pypi`` environment, the GitHub Release is
  created in a separate job with only ``contents: write``, the workflow default
  is ``contents: read`` and every action is pinned to a full commit SHA
  (WP-01-R, F-050).
- Renamed the `.gitignore` section for local working files and moved local tool patterns out of `.gitignore` (HYG-04).

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
