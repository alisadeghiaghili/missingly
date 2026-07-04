# missingly — Missing Data Toolkit (CONVENTIONS)

- `missingly` — Missing Data Toolkit (missingness analysis, imputation, comparison)
- `DQT` — SQL DB Data Quality Toolkit (DBA-focused, sister package)

**Relationship:**
- Independent sister packages.
- `DQT` may optionally bridge to `missingly` for missingness reporting.
- No hard dependency in either direction. `missingly` never imports `DQT`.

**Scope: Missing Data Only** — missingness diagnostics, imputation strategies, imputation
comparison, and reporting on those.
- No general-purpose data cleaning beyond missingness (belongs in DQT)
- No SQL/DB profiling or rules engine (belongs in DQT)
- No MDM/golden record, no compliance/masking

---

## Session Log

### [2026-07-04] Initial missingly Critical Review

**Critical Gaps Found:**
- Broad `except Exception:` in `impute.py` (~lines 2044–2048) silently falls back to
  `np.nanmean(y_train)`. This hides real errors and makes imputation failures
  undebuggable in production.
- Package combines too many concerns in one flat namespace: analysis, imputation,
  comparison, reporting, visualization, sklearn transformer, and workflow orchestration.
  Boundaries between these are blurred ("god package" risk).
- Accessor exposes thin wrappers that only `return func(self._df, **kwargs)` with no
  added value, doubling the maintenance surface.
- Inconsistent public naming: some methods use a `miss_` prefix (`miss_var_summary`,
  `miss_as_feature`), others are bare verbs (`clean_names`, `remove_empty`),
  others use `impute_` — no single rule.
- Imputation wrappers accept loose `**kwargs`, killing discoverability and static typing.
- Hardcoded thresholds/fallbacks (`_LARGE_DF_ROW_THRESHOLD = 50_000`,
  `_KNN_CAT_NEIGHBORS_THRESHOLD = 5`, all-NaN fill `0.0`) are not configurable.
- `collect_ignore_glob = ["*.py"]` and the comment "Prevent pytest from treating
  decorated functions as test items" indicate test-discovery workarounds and
  non-standard test organization. Real coverage is unknown.
- Custom `_split_encode` / `_decode` categorical pipeline is hand-rolled and can drift
  from sklearn semantics or carry subtle round-trip bugs.
- `stats.py` + `stats_extra.py` + `summary.py` are three files doing the job of one
  `diagnostics.py`. This split has no architectural justification; it is technical debt.
- `stats_extra.py` and `performance.py` are deprecated shim files that import from
  `data_quality_toolkit` at call time. They will break at runtime if DQT is not
  installed, with no clear error. They must be removed after the deprecation window.
- `manipulation.py` (`clean_names`, `remove_empty`) is out of scope per the charter
  but lives in the core package with no explanation.
- `manual_tests/` directory exists alongside `tests/`. This signals untestable or
  untested code paths. All manual tests must be automated or deleted.
- `_distance.py` is an internal helper for KNN distance calculation. Its fate depends
  on C5 (encode/decode pipeline): if sklearn encoders replace the hand-rolled pipeline,
  this file likely becomes redundant.
- Python version matrix in CONVENTIONS (3.9, 3.11, 3.12) conflicts with
  `pyproject.toml` target of Python 3.8+. This must be resolved to a single policy.

**Decisions Made:**
- Keep scope tight: missingness analysis, imputation, comparison, and reporting on those.
  Anything else moves to an optional module or a sister package.
- Error handling must be explicit. No broad `except Exception` without re-raise.
- Enforce one public API naming convention and one docstring style in CI.
- Externalize all hardcoded thresholds into a single config object.
- Make tests, clear English docstrings, and CI first-class requirements before v0.1.0.
- Preserve backward compatibility via `_deprecation.py` aliases for any renames.
- **Minimum Python version is 3.9** (not 3.8). `pyproject.toml` must be updated to match.
- `stats.py` + `stats_extra.py` + `summary.py` merge into `diagnostics.py` as part of A1.
  This must happen before C3 (naming cleanup), not after.
- `performance.py` and `stats_extra.py` are deprecated shims; both are deleted after
  the two-minor-version window closes (target: v0.3.0 removal).
- `manipulation.py` is out of scope. It moves to a clearly-labelled `utils/` subpackage
  with a deprecation notice pointing users to DQT, or is deleted if unused.
- `manual_tests/` is deleted. Every test case either moves to `tests/integration/` or
  is dropped.
- `_distance.py` fate is decided as part of C5. If the sklearn encoder path makes it
  redundant, delete it. If it is still needed, it stays private and gets full unit tests.

**Corrected Execution Order:**
C1 → C2 → **A1** → C3 → C4 → C5 → A2 → A3 → P2 → P3 → P4

Rationale: A1 (module restructure) must precede C3 (naming cleanup) because renaming
methods across a broken module layout forces you to redo deprecation aliases twice.
The structural boundary comes first; naming follows.

---

## Status Legend

- `[ ] TODO` — not started / not yet in code
- `[~] IN PROGRESS` — partially implemented / under refactor
- `[x] DONE` — implemented and verified in source

---

## P0 — Critical (correctness & production safety)

### [ ] C1. Explicit error handling
- Replace every broad `except Exception:` with specific exception types.
  - In `impute.py` (~2044–2048), catch only expected errors
    (`ValueError`, `np.linalg.LinAlgError`, sklearn fit errors).
  - Log every fallback before it happens; never fail silently.
- Add domain exceptions in a new `exceptions.py`:
  - `ImputationError`
  - `InsufficientDataError`
  - `InvalidStrategyError`
  - `MissingColumnError`
  - `ConfigurationError`
- Add a `strict_mode` switch: raises instead of falling back when enabled.
- All fallbacks must be logged, disableable, and documented in the docstring.
- **Acceptance:**
  - No bare `except` / `except Exception` without re-raise anywhere in the codebase.
  - Unit test per exception path.
  - A test that proves `strict_mode=True` raises instead of falling back.
  - `ruff` rule `BLE001` (blind exception) enabled and passing.

### [ ] C2. Input validation layer
- Add a new `_validation.py` module with shared validators:
  - `_validate_dataframe(df, allow_empty=False, min_rows=None)`
  - `_validate_columns(df, columns, require_numeric=False)`
  - `_validate_strategy(strategy, allowed)`
  - `_validate_positive_int(value, name)`
- Every public function validates:
  - Type is `DataFrame` (not Series / array / dict).
  - Shape is non-empty unless `allow_empty=True`.
  - Requested columns exist and have the required dtype.
  - Strategy and numeric params are within allowed range.
- Error messages must: name the offending parameter, state the expected value, and
  give a valid example. Example:
  ```
  ValueError: `strategy` must be one of {'mean', 'median', 'knn'}; got 'avg'.
  Try: strategy='mean'
  ```
- **Acceptance:**
  - `pytest.raises` test for each invalid-input path.
  - Error messages reviewed manually against the template above.
  - No public function that accepts a DataFrame skips validation.

---

## P1 — Architecture (structural; must precede C3)

### [ ] A1. Facets-based module layout

**This item is promoted to P1 and must be completed before C3.**

Merge and reorganise the current flat layout into:

```
missingly/
├── __init__.py          # public surface only; registers accessor
├── exceptions.py        # ImputationError, InsufficientDataError, … (new, from C1)
├── _validation.py       # shared validators (new, from C2)
├── config.py            # MissinglyConfig, get_config, set_config, config_context (new, from C4)
├── diagnostics.py       # MERGE of: stats.py + stats_extra.py + summary.py
│                        #   → all miss_* analysis functions live here
├── impute.py            # imputation strategies only (keep, refactor internals)
├── compare.py           # imputation comparison and CV (keep)
├── mi.py                # multiple imputation core (keep)
├── mi_workflow.py       # multiple imputation orchestration (keep)
├── accessor.py          # pandas .miss accessor (keep, thin and consistent)
├── report.py            # HTML reporting (keep)
├── simulate.py          # missingness simulation (keep — in scope)
├── timeseries.py        # time-series specific diagnostics/imputation (keep)
├── transformer.py       # MissinglyImputer sklearn transformer (keep)
├── _deprecation.py      # deprecation helpers (keep)
├── _distance.py         # DECIDE during C5: delete if sklearn encoders make it redundant
├── visualise.py         # thin dispatcher to visualisation/ (keep)
├── visualisation/       # plot modules (keep)
└── templates/           # Jinja2 templates (keep)
```

**Files to delete or migrate:**
- `stats.py` → content merged into `diagnostics.py`
- `stats_extra.py` → `pattern_monotone_test` and `missing_correlation_matrix` move to
  `diagnostics.py` with their deprecation warnings removed (they are in-scope);
  `hotelling_test` shim stays until v0.3.0 then is deleted.
- `summary.py` → content merged into `diagnostics.py`
- `performance.py` → delete after v0.3.0 (two minor versions from 0.2.0); add a
  deletion milestone to CHANGELOG.
- `manipulation.py` → move to `missingly/utils/manipulation.py` with a module-level
  deprecation warning: "These utilities are out of scope for missingly and will be
  removed in v0.4.0. Use DQT instead." Do not expose in `__init__.py`.
- `manual_tests/` → each script is either converted to `tests/integration/` or deleted.
  No manual test scripts in the repository after A1.

**No cross-facet import cycles.** Draw the dependency graph before merging:
```
exceptions ← _validation ← diagnostics
exceptions ← _validation ← impute
exceptions ← _validation ← compare
config ← impute
config ← compare
```
`diagnostics` must not import `impute`. `impute` must not import `diagnostics`.

- **Acceptance:**
  - `stats.py`, `stats_extra.py`, `summary.py` no longer exist as top-level files.
  - `performance.py` is either deleted or marked with a removal milestone ≤ v0.3.0.
  - `manipulation.py` no longer in `missingly/` root; moved or deleted.
  - `manual_tests/` directory is gone.
  - `import missingly` passes; all existing public symbols still importable (via aliases
    in `__init__.py` if needed for backward compat).
  - `pytest --import-mode=importlib` discovers tests correctly with no `conftest.py`
    workarounds.

### [ ] A2. Public API surface
- `missingly/__init__.py` explicitly exports the intended public surface only:
  - Accessor registration, top-level imputers, `MissinglyImputer`, comparison entry
    points, config helpers, and exceptions.
- `__all__` is defined and matches the API reference docs.
- Everything else is `_`-prefixed or documented as internal.
- `py.typed` marker file added (enables mypy for downstream users).
- **Acceptance:**
  - `__all__` defined.
  - A test imports every name in `__all__` and asserts it is not `None`.
  - No accidental leakage: `[x for x in dir(missingly) if not x.startswith('_')]`
    returns only names in `__all__`.

### [ ] A3. sklearn interoperability contract
- `MissinglyImputer(BaseEstimator, TransformerMixin)` must fully honour the sklearn API:
  `get_params` / `set_params`, `fit` / `transform` / `fit_transform`, `feature_names_out`.
- Run `sklearn.utils.estimator_checks.check_estimator` and document **each skipped check
  by name with the reason** (e.g. "skipped: check_n_features_in — we accept DataFrames,
  not numpy arrays by design").
- **Acceptance:**
  - Integration test: `MissinglyImputer` inside a `Pipeline` with `GridSearchCV`.
  - `get_params` / `set_params` round-trip test.
  - Each skipped estimator check is listed in a comment block in `test_transformer.py`.

---

## P2 (was P0 C3–C5) — API & Config cleanup

### [ ] C3. API surface cleanup & naming consistency

**Prerequisite: A1 must be complete.**

- One naming rule, enforced:
  - Analysis / diagnostic methods → `miss_*`
    (e.g. `miss_var_summary`, `miss_case_summary`, `miss_pattern`)
  - Manipulation methods → bare verbs
    (e.g. `clean_names`, `remove_empty`) — these now live in `utils/` only
  - Imputation methods → `impute_*`
- Accessor refactor:
  - Remove thin wrappers that only delegate; keep only methods with real value-add
    (chaining, state, or a distinct signature that cannot be replicated by calling the
    module function directly).
  - Replace `**kwargs` on all public imputation methods with explicit typed signatures.
- Deprecation path: every rename ships a deprecated alias via `_deprecation.py` with a
  `DeprecationWarning`, kept for at least two minor versions. Document in CHANGELOG.
- **Acceptance:**
  - All public methods follow the naming rule.
  - `ruff` rule `N802` (function name should be lowercase) passing.
  - Deprecation warnings fire and are verified in tests.
  - Migration notes in CHANGELOG.

### [ ] C4. Configuration externalization
- Introduce `config.py` with a `MissinglyConfig` dataclass:
  ```python
  @dataclass
  class MissinglyConfig:
      large_df_threshold: int = 50_000
      knn_cat_neighbors_threshold: int = 5
      all_nan_fillvalue: float = 0.0
      warn_on_large: bool = True
      strict_mode: bool = False
  ```
- Public interface: `get_config()`, `set_config(**kwargs)`, and a
  `config_context(...)` context manager that restores the previous config on exit.
- All imputation and diagnostic logic reads thresholds from config, not literals.
- **Acceptance:**
  - No hardcoded threshold left in imputation logic (`ruff` + grep confirm this).
  - Tests for override (`set_config`) and context manager isolation.
  - Config is documented in the API reference.

### [ ] C5. Encode/decode pipeline correctness
- Replace hand-rolled `_split_encode` / `_decode` with a sklearn-standard pattern
  (`ColumnTransformer` / `OrdinalEncoder` as appropriate).
- Guarantee lossless round-trip for categorical columns.
- After replacing: evaluate whether `_distance.py` is still needed. Delete if redundant.
- **Acceptance (expanded from original — these edge cases are not optional):**
  - Round-trip tests: `encode → decode == original` for:
    - Columns with `NaN` at start, middle, and end
    - A category column with exactly one unique value
    - A category column with 100+ cardinality
    - Unseen categories at `transform` time raise `InvalidStrategyError`, not `KeyError`
  - Benchmark confirms no meaningful performance regression vs. old pipeline
    (±10% on a 100k-row frame is acceptable; document the number).

---

## P3 — Testing, Docstrings, and CI (gate for v0.1.0)

### [ ] T1. Test layout and coverage
- Restructure tests:
  ```
  tests/
  ├── unit/
  │   ├── test_diagnostics.py
  │   ├── test_impute.py
  │   ├── test_compare.py
  │   ├── test_mi.py
  │   ├── test_accessor.py
  │   ├── test_config.py
  │   ├── test_exceptions.py
  │   └── test_validation.py
  ├── integration/
  │   ├── test_sklearn_pipeline.py
  │   └── test_full_workflow.py
  └── fixtures/
      ├── conftest.py          # shared fixtures
      └── datasets.py          # controlled-missingness DataFrames
  ```
- Required fixtures (must exist in `fixtures/conftest.py`):
  - `df_no_missing` — clean DataFrame, no NaN anywhere
  - `df_all_missing_col` — at least one column is 100% NaN
  - `df_single_row` — exactly one row
  - `df_mixed_dtypes` — numeric + categorical + datetime columns with missing
  - `df_large` — 100k rows, generated lazily (not stored as a file)
  - `df_high_cardinality_cat` — categorical column with 200+ unique values
- Remove `collect_ignore_glob = ["*.py"]` from both `conftest.py` files. Fix the
  underlying cause: rename any decorated helper that starts with `test_` or move it
  to a non-test module.
- Coverage target: **≥ 85%**. No public API without at least one test.
- **Acceptance:**
  - `pytest tests/` runs cleanly with no collection errors.
  - `pytest --cov=missingly --cov-report=term-missing` reports ≥ 85%.
  - `manual_tests/` is gone (enforced: CI fails if the directory exists).

### [ ] T2. Docstring and typing policy
- Every public function/class must have: one-line summary, `Args` section, `Returns`
  section, and one minimal `Example` in a `>>> ` doctest block.
- **Documentation language: English only.** No mixed-language comments in source.
- Docstring style: **Google-style** throughout. Enforced by `pydoclint` in CI.
- Complete type hints on all public and internal functions.
- `py.typed` marker file present (added in A2).
- `mypy --strict` passes with no blanket `type: ignore` comments (individual ignores
  with a reason comment are allowed, but must be justified).
- **Acceptance:**
  - `pydoclint --style=google missingly/` exits 0.
  - `mypy --strict missingly/` exits 0.
  - Every name in `__all__` has a docstring (checked in CI).

### [ ] T3. CI pipeline
- GitHub Actions workflow at `.github/workflows/ci.yml`:
  - Jobs: `lint`, `typecheck`, `test`
  - Matrix: **Python 3.9, 3.11, 3.12** (3.8 is dropped — see Decisions Made)
  - `lint`: `ruff check` + `black --check` + `pydoclint`
  - `typecheck`: `mypy --strict`
  - `test`: `pytest --cov=missingly --cov-fail-under=85`
- Fail conditions: lint violations, mypy errors, coverage < 85%.
- Optional job: docs build (`mkdocs build --strict`) — does not block merge but must
  not silently fail.
- **Acceptance:**
  - All three jobs pass on a clean branch before any PR is merged.
  - `pyproject.toml` `requires-python` is updated to `>=3.9`.

### [ ] T4. Logging infrastructure
- Use the standard `logging` module. No `print()` for operational events. No bare
  `warnings.warn()` for runtime events (use `logging.warning` instead;
  `warnings.warn` is reserved for deprecation notices only).
- Logger naming follows the module hierarchy:
  `missingly`, `missingly.impute`, `missingly.compare`, `missingly.diagnostics`, etc.
- Log levels:
  - `INFO`: large-data sampling notices, fallback activations (when `strict_mode=False`)
  - `WARNING`: validation edge cases, deprecation notices
  - `ERROR`: unrecoverable states before raising
- **Acceptance:**
  - `grep -r "print(" missingly/` returns no results (checked in CI via `ruff T201`).
  - Every fallback in `impute.py` logs at `INFO` before executing.

---

## P4 — Bridges & Sister Integration

### [ ] B1. Missingness report as a portable object
- Define a package-neutral `MissingnessReport` dataclass in `missingly/report_model.py`:
  - Must be serialisable to `dict` / JSON with no missingly-internal types.
  - Schema is versioned with a `schema_version: str` field.
- A sister package (e.g. DQT) can consume this dict without importing missingly internals.
- **Acceptance:**
  - `MissingnessReport` can be constructed and serialised to JSON in a test that does
    not import any other missingly module.
  - Schema version documented in `CONVENTIONS-missingly-data-model.md`.

### [ ] B2. DQT-facing contract
- `missingly` never imports `DQT`. The bridge lives entirely on the DQT side
  (`dqt.bridges.missingly`).
- `missingly` guarantees a stable entry point:
  `missingly.run(df, config=None) -> MissingnessReport`
- **Acceptance:**
  - Contract is documented in API reference.
  - `MissingnessReport` schema is versioned and any breaking change bumps `schema_version`.

---

## P5 — Documentation & Ecosystem (after v0.1.0)

### [ ] D1. README consolidation
- **Do not work on this before P0–P3 are complete.** Multilingual READMEs that are
  out of sync with the actual API are worse than no README.
- Canonical README: `README.md` (English). Single source of truth.
- After v0.1.0: generate or manually sync `README_FA.md`. Remove `README_DE.md` unless
  a German-speaking contributor commits to maintaining it. Stale translated docs
  mislead users.
- **Acceptance:** All READMEs reflect the same API version. CI checks that the
  version string in all READMEs matches `pyproject.toml`.

### [ ] D2. Architecture & API docs
- Architecture doc: package structure, design decisions, extension points.
- API reference: every public function with parameters, return spec, and example.
- Tutorials:
  - Basic missingness analysis
  - Imputation strategies comparison
  - sklearn pipeline integration
  - Custom imputation strategies
- Migration guide for every API change introduced in C3.
- Tooling: MkDocs + `mkdocstrings` (pulls from Google-style docstrings directly).

### [ ] D3. Extensibility — plugin strategy registration
- **Do not implement before C3 and A2 are stable.** A plugin interface on top of an
  unstable API creates breaking changes that compound.
- Plugin registration for custom strategies:
  `@missingly.register_strategy("my_custom")`
- Public base classes: `BaseImputer` (abstract), usable by external packages.
- **Acceptance:**
  - A third-party imputer registered via `register_strategy` works inside
    `MissinglyImputer` and `compare`.
  - Base classes are documented with a "Extending missingly" tutorial.

---

## Deletion Milestones (track in CHANGELOG)

| File / Symbol              | Deprecated in | Delete in | Reason                              |
|----------------------------|---------------|-----------|-------------------------------------|
| `performance.py`           | v0.2.0        | v0.3.0    | Moved to DQT                        |
| `stats_extra.hotelling_test` | v0.2.0      | v0.3.0    | Moved to DQT                        |
| `stats_extra.py` (shim)    | v0.2.0        | v0.3.0    | Merged into diagnostics.py          |
| `summary.py`               | (A1)          | (A1)      | Merged into diagnostics.py          |
| `stats.py`                 | (A1)          | (A1)      | Merged into diagnostics.py          |
| `manipulation.py` (root)   | (A1)          | v0.4.0    | Out of scope; moved to utils/       |
| `README_DE.md`             | —             | D1        | Unmaintained translation            |

---

## Source of Truth
- Data structures (`MissingnessReport`, config schema, exception hierarchy) are defined
  in `CONVENTIONS-missingly-data-model.md` (to be created as part of B1).
- This file is the single source of truth for task status. Update it when a task is
  completed, not retroactively.
- Every PR that completes a task must update the status marker in this file as part of
  the same commit.
