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

### [2026-07-06] Second Critical Review — Code-vs-Convention Reconciliation

**This session corrects the record. The previous session's status markers did not
match the actual source code. The following reflects the VERIFIED state of the code
as read directly from the repository.**

#### Status corrections (previously mis-marked)

- **C1 (explicit error handling): `[~] IN PROGRESS`**, not `[ ] TODO`.
  `impute.py` already catches `(ValueError, np.linalg.LinAlgError, NotFittedError)`,
  logs fallbacks at INFO, and honours `strict_mode`. BUT `report.py` (3×),
  `compare.py` (2×), and `diagnostics.py` (1×) still use `except Exception` with
  `# noqa: BLE001`. C1 is NOT done until these are specific and BLE001 passes without noqa.

- **C2 (validation layer): `[x] DONE`**.
  `_validation.py` and `exceptions.py` exist, are complete, and are wired into
  `diagnostics.py` and `impute.py`. All five exception classes confirmed present:
  `ImputationError`, `InsufficientDataError`, `InvalidStrategyError`,
  `MissingColumnError`, `ConfigurationError`.

- **A1 (module layout): `[~] IN PROGRESS`**.
  `diagnostics.py` has merged stats + summary logic. BUT `stats.py` and `summary.py`
  survive as `__getattr__` shims, and BOTH `__init__.py` and `accessor.py` still import
  the doomed modules (`summary`, `stats`, `stats_extra`, `performance`, `manipulation`).
  A1 is not done until accessor and __init__ no longer depend on shim modules.

- **C4 (config): `[ ] TODO`** — but `impute.py` docstrings already **reference**
  a non-existent `missingly.config` module. These references are lies. Either create
  `config.py` (the real fix) or remove the references from docstrings first (the fast
  fix). Do not leave documentation that references code that does not exist.

#### Confirmed exceptions.py signatures (verified from source)

```python
InvalidStrategyError(param: str, got: Any, allowed: List[Any], example: Optional[str] = None)
ImputationError(column: str, strategy: str, original: Exception)
InsufficientDataError(column: str, n_observed: int, n_required: int)
MissingColumnError(columns: List[str], available: List[str])
ConfigurationError(param: str, got: Any, reason: str)
```

#### Confirmed conftest.py root cause (verified from source)

`missingly/conftest.py` contains exactly:
```python
"""Prevent pytest from collecting this source package as a test directory."""
collect_ignore_glob = ["*.py"]
```
This workaround exists because `test_hotelling` and `test_pattern_monotone` are
exported in `__all__`, causing pytest to collect them as test functions. Fix `__all__`
(A2 sub-item below), then delete this file entirely.

---

#### NEW P0 items — Correctness bugs (blockers; higher priority than any refactor)

These are silent-wrong-result or crash bugs found in the actual source. They outrank
every structural or naming task. A package that returns mean-imputed data when asked
for KNN is not shippable at any version number.

**Revised execution order:**
```
X1 → X2 → X3 → X4 → (finish C1: kill remaining except Exception) →
A2 (__all__ hygiene, unblocks T1) → A1 (finish module deletions) →
C4 → C3 → C5 → A3 → rest
```

##### [ ] X1. FittedImputer.transform ignores strategy — SILENT WRONG RESULTS

**File:** `missingly/impute.py` | **Class:** `FittedImputer`

Verified from source:
- `fit()` calls `_dispatch_strategy(self.strategy, df, **self.kwargs)` and stores the
  result in `self._fitted_train` — which is **never used again**.
- `transform()` always applies `np.nanmean` for numeric and `mode` for categorical,
  regardless of `self.strategy`.
- `make_imputer('knn').fit(train).transform(test)` returns **mean-imputed data** with
  no warning and no error.
- `fit_transform()` calls `_dispatch_strategy` directly and works correctly — but is
  inconsistent with `fit` + `transform`.

**Decision:** Restrict `FittedImputer` to `{mean, median, mode}` (strategies whose
fit-state is a per-column fill value). Raise `InvalidStrategyError` for all other
strategies with a clear message pointing to `MissinglyImputer`. Do NOT duplicate
the knn/rf/gb/mice fit logic — `MissinglyImputer` already implements it correctly.

**Implementation:**
1. Add constant after `_SUPPORTED_KNN_METRICS`:
   ```python
   _SIMPLE_FIT_STRATEGIES = frozenset({"mean", "median", "mode"})
   ```
2. In `__init__`, after `validate_strategy(...)`, add guard:
   ```python
   if strategy not in _SIMPLE_FIT_STRATEGIES:
       raise InvalidStrategyError(
           param="strategy",
           got=strategy,
           allowed=sorted(_SIMPLE_FIT_STRATEGIES),
       )
   ```
3. In `fit()`, remove: `self._fitted_train = _dispatch_strategy(self.strategy, df, **self.kwargs)`
   Keep: `self._fit_df = df.copy()` and `self._is_fitted = True`.
4. In `transform()`, replace hardcoded mean with strategy-aware fill:
   ```python
   if self.strategy == "mode":
       fill_val = train_col.dropna().mode().iloc[0]
   elif pd.api.types.is_numeric_dtype(train_col):
       if self.strategy == "median":
           fill_val = float(np.nanmedian(train_col.values))
       else:  # mean
           fill_val = float(np.nanmean(train_col.values))
   else:
       fill_val = train_col.dropna().mode().iloc[0]
   ```
5. In `fit_transform()`, replace `return _dispatch_strategy(...)` with
   `return self.transform(df)`.
6. Update class docstring: state only `{mean, median, mode}` are accepted; add note
   "For model-based strategies use `MissinglyImputer`."

**Acceptance:**
- `make_imputer('median').fit(train).transform(test)` fills with the TRAIN median.
- `make_imputer('knn')` raises `InvalidStrategyError` immediately.
- `make_imputer('mean').fit(train).transform(test)` == `make_imputer('mean').fit_transform(test)`
  is FALSE (intentional — fit uses train stats, transform applies to test).

##### [ ] X2. MissinglyImputer breaks the sklearn parameter contract

**File:** `missingly/transformer.py` | **Class:** `MissinglyImputer`

Verified from source:
- `__init__(self, ..., **estimator_kwargs)` — `**kwargs` in `__init__` is forbidden by
  sklearn. `get_params()` / `set_params()` / `clone()` / `check_estimator` all fail.
- Fitted-state attributes (`numeric_cols_`, `rf_reg_models_`, etc.) are initialised
  in `__init__` with trailing underscores — violates sklearn convention that `_`-suffix
  attributes exist only after `fit`.

**Implementation:**
1. Change `__init__` signature:
   ```python
   # Before
   def __init__(self, ..., **estimator_kwargs) -> None:
   # After
   def __init__(self, ..., estimator_kwargs: Optional[dict] = None) -> None:
   ```
2. Store verbatim (unchanged): `self.estimator_kwargs = estimator_kwargs`
   (Do NOT coerce to `{}` in `__init__` — sklearn requires storing the value as-is.)
3. In `_fit_tree`, at the top: `est_kwargs = self.estimator_kwargs or {}`
   Replace all four `**self.estimator_kwargs` with `**est_kwargs`.
4. Move `numeric_cols_`, `rf_reg_models_`, `rf_cls_models_`, `gb_reg_models_`,
   `gb_cls_models_`, `encoders_`, `cat_cols_`, `num_cols_` out of `__init__`.
   Set them only at the top of `fit()`.
5. Update docstring: replace `**estimator_kwargs` entry with
   `estimator_kwargs : dict or None, optional`.

**Acceptance:**
- `MissinglyImputer().get_params()` includes key `'estimator_kwargs'`.
- `MissinglyImputer(estimator_kwargs={'n_estimators': 50}).get_params()` round-trips.
- `sklearn.utils.estimator_checks.check_estimator(MissinglyImputer())` lists only
  intentional, named skips.

##### [ ] X3. Accessor mar_mnar_test — target_col silently discarded, signature crash

**File:** `missingly/accessor.py` | **Method:** `MissinglyAccessor.mar_mnar_test`

Verified from source:
- Method signature accepts `target_col: Optional[str] = None`.
- Body calls `stats.mar_mnar_test(self._df)` — `target_col` is never used.
- The real `diagnostics.mar_mnar_test(X, Y, ...)` requires `Y` as an outcome array.
  Passing a column-name string as `Y` crashes inside `LogisticRegression`.

**Implementation:**
1. Fix imports in `accessor.py`: remove `stats` from `from . import (...)`,
   add `from . import diagnostics` and `from .exceptions import MissingColumnError`.
2. Rewrite method:
   ```python
   def mar_mnar_test(self, target: str) -> pd.DataFrame:
       if target not in self._df.columns:
           raise MissingColumnError(
               columns=[target], available=list(self._df.columns)
           )
       X = self._df.drop(columns=[target])
       Y = self._df[target].to_numpy()
       return diagnostics.mar_mnar_test(X, Y)
   ```
3. Replace all `stats.*` and `summary.*` calls in the file with `diagnostics.*`.
4. Fix `mcar_test` docstring: correct return keys to
   `chi_square, df, p_value, missing_patterns, amount_missing`.

**Acceptance:**
- `df.miss.mar_mnar_test(target="income")` returns a valid result.
- `df.miss.mar_mnar_test(target="nonexistent")` raises `MissingColumnError`.
- Old call `df.miss.mar_mnar_test(target_col=...)` raises `TypeError` with a helpful
  message (parameter renamed intentionally).

##### [ ] X4. Categorical estimator fallback uses mean of ordinal codes

**File:** `missingly/impute.py` | **Function:** `_impute_column_by_column`

Verified from source:
- The except branch computes `fallback = float(np.nanmean(y_train))` for ALL columns.
- At this point `y_train` is already ordinal-encoded (float codes from `_split_encode`).
  For a categorical column, this fills with a fractional code mean — statistically wrong.
- The existing log message claims "column mean (numeric) or mode (categorical)" but
  the code does NOT branch on column type.

**Implementation:**
Replace the single fallback line with:
```python
if col in cat_cols:
    valid_codes = y_train[~np.isnan(y_train)].astype(int)
    valid_codes = valid_codes[valid_codes >= 0]  # guard: OrdinalEncoder uses -1 for unknown
    fallback = float(np.bincount(valid_codes).argmax()) if len(valid_codes) > 0 else 0.0
    logger.info(
        "Categorical fallback for %r (strategy=%r): mode code %d. "
        "Set strict_mode=True to raise instead.",
        col, strategy_name, int(fallback),
    )
else:
    fallback = float(np.nanmean(y_train))
    logger.info(
        "Numeric fallback for %r (strategy=%r): mean %.4f. "
        "Set strict_mode=True to raise instead.",
        col, strategy_name, fallback,
    )
```

**Acceptance:**
- Unit test forces an estimator failure on a categorical column; asserts the fill value
  is a valid existing category code (integer), not a rounded fractional mean.
- Unit test forces an estimator failure on a numeric column; asserts fill is the mean.

---

#### NEW A2 sub-item — __all__ hygiene (root cause of conftest workaround)

`__all__` currently exports `test_hotelling`, `test_pattern_monotone` — these are the
ROOT CAUSE of `collect_ignore_glob = ["*.py"]` in `missingly/conftest.py`.
It also exports out-of-scope `chunk_apply`, `memory_usage_mb`, `optimize_dtypes` and
deprecated manipulation symbols.

**Fix ordering:**
1. Remove `test_hotelling` / `test_pattern_monotone` from `__all__` AND the module
   namespace. Rename to `_test_hotelling` / `_test_pattern_monotone` if still needed
   internally.
2. Remove `chunk_apply`, `memory_usage_mb`, `optimize_dtypes` from `__all__`
   (DQT scope; keep importable via deprecated aliases only).
3. Remove deprecated manipulation aliases from `__all__` (keep importable, not public).
4. **Delete `missingly/conftest.py` entirely** once step 1 is done.

**Acceptance:**
- `[x for x in dir(missingly) if x.startswith('test_')] == []`
- `missingly/conftest.py` does not exist.
- `pytest tests/` collects correctly with no workarounds.

#### Doc-vs-code integrity rule (new, permanent)

No docstring may reference a module, function, or parameter that does not exist in
source. `impute.py` currently references `missingly.config` which does not exist.

**Immediate action:** Either create `config.py` (the C4 task) OR remove the references
from `impute.py` docstrings. Do not leave lying documentation.

**Acceptance:** `grep -r "missingly.config" missingly/` returns matches ONLY after
`config.py` actually exists.

---

## Status Legend

- `[ ] TODO` — not started / not yet in code
- `[~] IN PROGRESS` — partially implemented / under refactor
- `[x] DONE` — implemented and verified in source

---

## P0 — Critical (correctness & production safety)

### [~] C1. Explicit error handling
- Replace every broad `except Exception:` with specific exception types.
  - **DONE in `impute.py`**: catches `(ValueError, np.linalg.LinAlgError, NotFittedError)`,
    logs fallbacks at INFO, honours `strict_mode`.
  - **REMAINING**: `report.py` (3 sites), `compare.py` (2 sites), `diagnostics.py` (1 site)
    still use `except Exception` with `# noqa: BLE001`. These must be narrowed.
- All fallbacks must be logged, disableable, and documented in the docstring.
- **Acceptance:**
  - No bare `except` / `except Exception` without re-raise anywhere in the codebase.
  - Unit test per exception path.
  - A test that proves `strict_mode=True` raises instead of falling back.
  - `ruff` rule `BLE001` (blind exception) enabled and passing with **no** `# noqa` suppression.

### [x] C2. Input validation layer
- `_validation.py` and `exceptions.py` exist, are complete, and are wired into the
  imputation and diagnostics layers. All five exception classes present and tested.
- **No further work required for C2 itself.**

---

## P1 — Architecture (structural; must precede C3)

### [~] A1. Facets-based module layout

**This item is promoted to P1 and must be completed before C3.**

**Current state:** `diagnostics.py` has merged stats + summary logic. `stats.py` and
`summary.py` survive as `__getattr__` shims. Both `__init__.py` and `accessor.py` still
import the doomed shim modules.

Merge and reorganise the current flat layout into:

```
missingly/
├── __init__.py          # public surface only; registers accessor
├── exceptions.py        # [x] DONE — ImputationError, InsufficientDataError, …
├── _validation.py       # [x] DONE — shared validators
├── config.py            # [ ] TODO — MissinglyConfig, get_config, set_config (C4)
├── diagnostics.py       # [~] IN PROGRESS — MERGE of: stats.py + stats_extra.py + summary.py
├── impute.py            # [~] IN PROGRESS — imputation strategies (X1, X4 pending)
├── compare.py           # [~] IN PROGRESS — C1 remaining sites
├── mi.py                # keep
├── mi_workflow.py       # keep
├── accessor.py          # [~] IN PROGRESS — X3 pending
├── report.py            # [~] IN PROGRESS — C1 remaining sites
├── simulate.py          # keep
├── timeseries.py        # keep
├── transformer.py       # [~] IN PROGRESS — X2 pending
├── _deprecation.py      # keep
├── _distance.py         # DECIDE during C5
├── visualise.py         # keep
├── visualisation/       # keep
└── templates/           # keep
```

**Files to delete or migrate:**
- `stats.py` → merge into `diagnostics.py`, then delete
- `stats_extra.py` → merge in-scope functions into `diagnostics.py`, then delete
- `summary.py` → merge into `diagnostics.py`, then delete
- `performance.py` → delete after v0.3.0
- `manipulation.py` → move to `missingly/utils/manipulation.py` with deprecation notice
- `manual_tests/` → convert to `tests/integration/` or delete; directory must not exist
- `missingly/conftest.py` → **delete** after A2 __all__ hygiene removes `test_*` symbols

**Dependency graph (no cross-facet import cycles):**
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
  - `accessor.py` imports from `diagnostics`, not `stats` or `summary`.
  - `__init__.py` imports from `diagnostics`, not shim modules.
  - `missingly/conftest.py` deleted.
  - `import missingly` passes; all existing public symbols still importable.

### [ ] A2. Public API surface
- `__all__` is defined and matches the API reference docs.
- **Sub-item (unblocks T1):** Remove `test_hotelling`, `test_pattern_monotone` from
  `__all__` and public namespace. Remove `chunk_apply`, `memory_usage_mb`,
  `optimize_dtypes` from `__all__`. Remove deprecated manipulation aliases from `__all__`.
  After this, delete `missingly/conftest.py`.
- `py.typed` marker file added.
- **Acceptance:**
  - `[x for x in dir(missingly) if x.startswith('test_')] == []`
  - `__all__` defined and tested.
  - `missingly/conftest.py` does not exist.

### [ ] A3. sklearn interoperability contract
- `MissinglyImputer` must fully honour the sklearn API. **Prerequisite: X2 must be
  complete before A3 can pass `check_estimator`.**
- Run `check_estimator` and document each skipped check by name with reason.
- **Acceptance:**
  - Integration test: `MissinglyImputer` inside a `Pipeline` with `GridSearchCV`.
  - `get_params` / `set_params` round-trip test.
  - Each skipped estimator check listed in `test_transformer.py`.

---

## P2 (was P0 C3–C5) — API & Config cleanup

### [ ] C3. API surface cleanup & naming consistency

**Prerequisite: A1 must be complete.**

- One naming rule: `miss_*` for analysis, `impute_*` for imputation, bare verbs in
  `utils/` only.
- Accessor: remove thin wrappers with no value-add; replace `**kwargs` with typed sigs.
- **Also:** Add `df.miss.impute(strategy=...)` method to match what both READMEs
  advertise. This method does not currently exist — the READMEs are lying. Either add
  it here or fix the READMEs in D1. Do not leave both broken.
- Deprecation path: every rename ships via `_deprecation.py` with `DeprecationWarning`.
- **Acceptance:**
  - All public methods follow the naming rule.
  - `df.miss.impute(strategy="mean")` works as documented.
  - Deprecation warnings fire and are verified in tests.

### [ ] C4. Configuration externalization
- Create `config.py` with `MissinglyConfig` dataclass.
- **First action:** Remove all `missingly.config` references from `impute.py` docstrings
  until `config.py` actually exists. Do not leave lying documentation.
- All imputation and diagnostic logic reads thresholds from config, not literals.
- **Acceptance:**
  - `config.py` exists and is importable.
  - No hardcoded threshold left in imputation logic.
  - `grep -r "missingly.config" missingly/` returns matches only after this is done.

### [ ] C5. Encode/decode pipeline correctness
- Replace hand-rolled `_split_encode` / `_decode` with sklearn-standard pattern.
- Guarantee lossless round-trip for categorical columns.
- After replacing: evaluate `_distance.py`; delete if redundant.
- **Acceptance (edge cases are not optional):**
  - Round-trip tests for NaN at start/middle/end, single unique value, 100+ cardinality,
    unseen categories raise `InvalidStrategyError` (not `KeyError`).
  - Benchmark: ±10% vs old pipeline on 100k-row frame; document the number.

---

## P3 — Testing, Docstrings, and CI (gate for v0.1.0)

### [ ] T1. Test layout and coverage
- Restructure tests into `unit/`, `integration/`, `fixtures/`.
- Required fixtures: `df_no_missing`, `df_all_missing_col`, `df_single_row`,
  `df_mixed_dtypes`, `df_large`, `df_high_cardinality_cat`.
- **Prerequisite: A2 __all__ hygiene must be done first** (removes root cause of
  `collect_ignore_glob` workaround).
- Coverage target: ≥ 85%.
- **Acceptance:**
  - `pytest tests/` runs cleanly with no collection errors.
  - `pytest --cov=missingly --cov-report=term-missing` reports ≥ 85%.
  - `manual_tests/` does not exist.

### [ ] T2. Docstring and typing policy
- Every public function/class: one-line summary, `Args`, `Returns`, one `Example`.
- **Documentation language: English only.**
- Docstring style: **Google-style** throughout.
- `mypy --strict` passes.
- **Acceptance:** `pydoclint --style=google missingly/` exits 0. `mypy --strict` exits 0.

### [ ] T3. CI pipeline
- GitHub Actions: `lint`, `typecheck`, `test` jobs.
- Matrix: Python 3.9, 3.11, 3.12.
- Coverage gate: `--cov-fail-under=85`.
- **Acceptance:** All three jobs pass on a clean branch before any PR merges.

### [ ] T4. Logging infrastructure
- No `print()` for operational events. No bare `warnings.warn()` for runtime events.
- Logger naming follows module hierarchy.
- Every fallback logs at `INFO` before executing.
- **Acceptance:** `grep -r "print(" missingly/` returns no results.

---

## P4 — Bridges & Sister Integration

### [ ] B1. Missingness report as a portable object
- `MissingnessReport` dataclass in `missingly/report_model.py`. Serialisable to JSON.
  Schema versioned with `schema_version: str`.
- **Acceptance:** Constructed and serialised in a test that imports no other missingly module.

### [ ] B2. DQT-facing contract
- `missingly` never imports `DQT`.
- Stable entry point: `missingly.run(df, config=None) -> MissingnessReport`.
- **Acceptance:** Contract documented. Schema versioned.

---

## P5 — Documentation & Ecosystem (after v0.1.0)

### [ ] D1. README consolidation
- **Do not work on this before P0–P3 are complete.**
- Canonical README: `README.md` (English). Single source of truth.
- **Known issue:** Both `README_FA.md` and `README_DE.md` advertise `df.miss.impute(strategy=...)`
  which does not exist. This must be resolved in C3 (add the method) or D1 (fix the docs).
  Do not leave this unresolved past v0.1.0.
- **Acceptance:** All READMEs reflect the same API version. CI checks version string match.

### [ ] D2. Architecture & API docs
- Architecture doc, API reference, tutorials (basic analysis, imputation comparison,
  sklearn pipeline, custom strategies), migration guide.
- Tooling: MkDocs + `mkdocstrings`.

### [ ] D3. Extensibility — plugin strategy registration
- **Do not implement before C3 and A2 are stable.**
- `@missingly.register_strategy("my_custom")` plugin hook.
- `BaseImputer` abstract base class.

---

## Deletion Milestones (track in CHANGELOG)

| File / Symbol                   | Deprecated in | Delete in | Reason                              |
|---------------------------------|---------------|-----------|-------------------------------------|
| `performance.py`                | v0.2.0        | v0.3.0    | Moved to DQT                        |
| `stats_extra.hotelling_test`    | v0.2.0        | v0.3.0    | Moved to DQT                        |
| `stats_extra.py` (shim)         | v0.2.0        | v0.3.0    | Merged into diagnostics.py          |
| `summary.py`                    | (A1)          | (A1)      | Merged into diagnostics.py          |
| `stats.py`                      | (A1)          | (A1)      | Merged into diagnostics.py          |
| `manipulation.py` (root)        | (A1)          | v0.4.0    | Out of scope; moved to utils/       |
| `missingly/conftest.py`         | —             | (A2)      | Workaround; root cause fixed in A2  |
| `FittedImputer` / `make_imputer`| v0.3.0        | v0.5.0    | Restricted to simple strategies;    |
|                                 |               |           | use MissinglyImputer instead        |
| `README_DE.md`                  | —             | D1        | Unmaintained translation            |

---

## Source of Truth
- Data structures (`MissingnessReport`, config schema, exception hierarchy) are defined
  in `CONVENTIONS-missingly-data-model.md` (to be created as part of B1).
- **This file is the single source of truth for task status.** Update it when a task is
  completed, not retroactively.
- Every PR that completes a task must update the status marker in this file as part of
  the same commit.
- No docstring may reference a module or symbol that does not exist in source.
