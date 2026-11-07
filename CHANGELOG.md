# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Ruff lint configuration aligned with the SonarQube rule families (`E`, `W`,
  `F`, `B`, `BLE`, `S`, `N`, `C90`, `PL`, `SIM`, `ERA`, `UP`, `RUF`, `NPY`, `PD`,
  `PT`, `TRY`, `RET`, `ARG`, `ISC`, `DTZ`, `PGH`) with `max-complexity = 15`,
  `max-args = 13` and `max-positional-args = 13`. A ratchet script,
  `tools/ruff_ratchet.py`, compares per-file and per-rule counts and the
  unformatted file set with `tools/ruff_baseline.json`; CI fails on any new
  violation and the baseline records existing debt (SQ-02).
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

- Added one internal dtype normalisation layer (`missingly._dtypes`) and routed
  the imputers in `missingly.impute` through it (WP-04). Nullable `Int64` /
  `Float64`, `float32`, `int64`, `bool`, `boolean`, `string`, `category`,
  datetime (naive and time-zone aware), timedelta and mixed-type `object`
  columns no longer crash the imputers or silently change dtype (F-014,
  F-018, F-020, F-023, F-027, F-051, F-055, F-058, F-091, F-093, F-098, F-100,
  F-105, F-107, F-234, F-235, F-252, F-343):
  - nullable numeric columns are modelled as `float64` instead of crashing
    `impute_rf`, `impute_gb`, `impute_mice` (`return_history`, `where`,
    `bounds`, `visit_sequence`, `predictor_matrix`) and `impute_pmm`, and are
    restored to their original dtype; integer imputations from KNN, MICE, RF,
    GB and PMM are rounded to the nearest integer, so `Int64` stays `Int64`;
  - `impute_mean`, `impute_median` and `FittedImputer` rewrite only incomplete
    columns, treat `bool` / `boolean` / datetime / timedelta / `string` columns
    as non-numeric (mode fill) and keep a nullable integer dtype when the fill
    value is integral (otherwise `Float64`); `float32` stays `float32`;
  - `impute_mice(..., return_result=True)` accepts `int64`, `float32`, `Int64`
    and interleaved string columns because the completed frames now keep the
    original column order and dtypes;
  - the logistic imputers (`impute_logreg`, `impute_polyreg`, `impute_polr`)
    encode nullable numeric predictors by value instead of lexical rank and
    cast predicted labels to the column dtype;
  - an all-missing column raises `InsufficientDataError` in `impute_mean`,
    `impute_median` and `impute_knn` (both metrics), matching `impute_mode`
    and `impute_mice`; `impute_knn(metric="mixed")` uses the donor mode for
    boolean columns and rounds integer columns;
  - model-based imputers (`impute_knn`, `impute_mice`, `impute_rf`,
    `impute_gb`, `impute_polyreg`, `impute_polr`, `impute_logreg`) no longer
    treat complete datetime, timedelta or mixed-type `object` columns as
    numbers or ordinal categories: they pass through unchanged and are not
    predictors. Incomplete ones that the previous release imputed without
    error (datetime in KNN, MICE, RF, GB and the logistic imputers;
    timedelta and mixed-type object in `impute_knn(metric="mixed")` and the
    logistic imputers) now emit a `FutureWarning` and keep working; every
    other incomplete case raises the new
    `missingly.exceptions.UnsupportedDtypeError`;
  - `impute_mice(bounds=...)` on a boolean column emits a `FutureWarning`
    (the bound is ignored; it will become an error).
  - column order of the encoding-based imputers is unchanged (numeric columns
    first); `impute_knn(metric="mixed")` keeps the input order.
- Unified `random_state` handling for stochastic imputers and argument
  validation (WP-05, F-033, F-111, F-203, F-231, F-244):
  - every stochastic function (`impute_hotdeck_random`,
    `impute_hotdeck_weighted`, `impute_pmm`, `impute_logreg`,
    `impute_polyreg`, `impute_polr`, `impute_rf`, `impute_gb`, `impute_mice`)
    accepts `None`, non-negative integers of any size (Python or numpy), a
    `numpy.random.Generator` and a `numpy.random.RandomState`; integer seeds
    give the same output as before. Other values raise
    `missingly.exceptions.InvalidSeedError` before any draw (F-231). It is a
    `ConfigurationError` and is also a `ValueError` and a `TypeError`, so
    code that caught those builtins for invalid seeds keeps working. A
    `impute_polyreg`, `impute_polr`, `impute_rf` and `impute_gb` used to
    return a silent mean or mode fallback for an invalid seed; they now raise
    (F-033);
  - a `bool` seed (`True` / `False`) is deprecated, not rejected: it still
    gives exactly the output of `1` / `0` as before, and emits a
    `FutureWarning`. A future release will raise `InvalidSeedError` for it;
  - `impute_rf` and `impute_gb` no longer turn a `Generator` seed, an integer
    of 2**32 or more, or a negative seed into mean imputation; these seeds
    now fit the model, and invalid seeds raise (F-033);
  - `impute_mice(return_result=True)` accepts numpy integers for
    `random_state`, `max_iter` and `n_imputations` (F-111);
  - `validate_positive_int` (counters such as `n_neighbors`, `n_donors`,
    `n_imputations`, `max_iter`, `n_nearest_donors`) treats a `bool` as
    `1` / `0`, as before, and emits a `FutureWarning`; `False` is still
    rejected where zero is not allowed. A future release will reject booleans
    with `TypeError` (F-203, deprecated rather than rejected, so existing
    code keeps its output);
  - `impute_mice(return_result=True, n_imputations=True)` now works; it used
    to fail with `ValueError` after all chains had run (F-111);
  - `impute_mice` accepts numpy-integer `bounds` and non-string column labels
    in `visit_sequence` (F-244).
  - Not changed here (WP-05b, needs owner approval): chain-seed derivation
    `random_state + i` (F-246) and the default seed policy (`random_state=0`
    versus `None`, F-247). Defaults are unchanged.
  - Not changed here: `simulate_*` and `compare_imputations` still use the
    previous seed handling (F-231 follow-up, outside this work package).
- Fixed dtype handling in hot-deck imputation and missingness simulation
  (WP-04, F-012, F-232, F-233, F-358):
  - `impute_hotdeck_weighted` accepts bool and nullable `boolean` columns,
    matching them as categories instead of crashing (F-012);
  - `impute_hotdeck_weighted` copies donor values exactly, so `Int64`
    integers beyond 2**53 keep their precision (F-233);
  - `impute_hotdeck_random` copies donor values without the naive
    NumPy round trip, so tz-aware datetime columns keep their time zone and
    no longer emit a pandas FutureWarning or fail (F-232);
  - `simulate_mcar`, `simulate_mar`, `simulate_mnar` and `simulate_mixed`
    return bool columns they mask as nullable `boolean` instead of failing
    or emitting a pandas FutureWarning (F-358). int64 columns still become
    float64 when masked, as before.
- Static-plot figure lifetime and degenerate inputs (WP-06): a static plot
  that creates its own figure now closes it when it raises afterwards, for
  example after a misspelled column name (F-375). A call given `ax` never
  creates a figure (F-366). A frame with no rows or no columns makes
  `matrix` and `miss_cooccurrence` (no columns) and the column-based plots
  raise a clear `ValueError` instead of a `TypeError`, `KeyError` or
  matplotlib error (F-367, F-302). Other degenerate frames (no rows, no
  missing values, all values missing) are drawn as an annotated empty state
  without runtime warnings, and `miss_row_profile` no longer shows a
  completeness axis beyond 100% (F-302, F-374). `vis_miss_cumsum_var` and
  `vis_miss_cumsum_case` no longer divide by zero when nothing is missing
  (F-380).
- Static-plot API hygiene (WP-06): unsupported keyword arguments of the static
  plot functions, `interactive=True` on functions without a Plotly backend and
  an unknown `backend` value are no longer swallowed silently; each emits a
  `FutureWarning` naming the argument (a future release will raise
  `TypeError`; F-147, F-365, F-303). Styling arguments the Plotly backend
  cannot honour (`color`, `figsize`, `cmap`, ...) now emit a `UserWarning`
  (F-303). Interactive calls without Plotly now raise the `ImportError` that
  names `missingly[interactive]` (F-299). `matrix(color=...)` accepts any
  Matplotlib colour string (F-301). `fontsize` is honoured by `heatmap`,
  `miss_cooccurrence`, `vis_miss_by_group` and by `dendrogram` for the
  `left`/`right` orientations (F-370, F-371). `miss_case` labels its x axis
  "Row (sorted by # missing)" when sorted and `vis_parallel_coords` no longer
  hard-codes a colour in its title (F-368, F-382). The module, README and API
  documentation now state which functions lack `ax`, return a non-`Axes`
  value, or have a Plotly backend, and that `heatmap` masking is on by
  default (F-298, F-364).
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
