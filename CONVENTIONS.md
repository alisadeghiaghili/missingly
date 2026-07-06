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

##### [x] X1. FittedImputer.transform ignores strategy — SILENT WRONG RESULTS

**File:** `missingly/impute.py` | **Class:** `FittedImputer`
**Fixed in:** commit `fcfdc4c` (2026-07-06)

Verified from source:
- `fit()` called `_dispatch_strategy(self.strategy, df, **self.kwargs)` and stored
  result in `self._fitted_train` which was **never used again**.
- `transform()` always applied `np.nanmean` for numeric and `mode` for categorical,
  regardless of `self.strategy`.

**What was done:**
1. Added `_SIMPLE_FIT_STRATEGIES = frozenset({"mean", "median", "mode"})` constant.
2. `__init__` raises `InvalidStrategyError` immediately for knn/mice/rf/gb.
3. `fit()` stores only `self._fit_df` — no more stale `_dispatch_strategy` call.
4. `transform()` is now strategy-aware: separate mean / median / mode branches.
5. `fit_transform()` delegates to `self.fit(df).transform(df)`.
6. Class docstring updated.

##### [x] X2. MissinglyImputer breaks the sklearn parameter contract

**File:** `missingly/transformer.py` | **Class:** `MissinglyImputer`
**Fixed in:** commit `fcfdc4c` (2026-07-06)

**What was done:**
1. `**estimator_kwargs` → `estimator_kwargs: Optional[dict] = None`.
2. Fitted-state attrs moved OUT of `__init__` — set only inside `fit()`.

##### [x] X3. Accessor mar_mnar_test — target_col silently discarded, signature crash

**File:** `missingly/accessor.py` | **Method:** `MissinglyAccessor.mar_mnar_test`
**Fixed in:** commit `fcfdc4c` (2026-07-06)

**What was done:**
1. Rewrote method with required `target: str` param and correct X/Y split.
2. Raises `MissingColumnError` on invalid column.
3. Delegates to `diagnostics.mar_mnar_test(X, Y)`.

##### [x] X4. Categorical estimator fallback uses mean of ordinal codes

**File:** `missingly/impute.py` | **Function:** `_impute_column_by_column`
**Fixed in:** commit `fcfdc4c` (2026-07-06)

**What was done:**
1. Replaced single fallback line with `if col in cat_cols:` / `else:` branch.
2. Categorical: `bincount(valid_codes).argmax()` with `valid_codes >= 0` guard.
3. Two separate `logger.info(...)` messages per branch.

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

### [2026-07-06] Session 3 — X1–X4 Implementation

**Commit:** `fcfdc4c` — pushed directly to `main`.

All four P0 correctness bugs resolved:
- X1: `FittedImputer` restricted to `{mean, median, mode}`; strategy-aware `transform`.
- X2: `MissinglyImputer` sklearn contract repaired.
- X3: `accessor.mar_mnar_test` rewritten with required `target` param.
- X4: categorical fallback uses `bincount` mode, not mean of ordinal codes.

---

### [2026-07-06] Session 4 — C1 Verification

**Finding:** C1 was already complete in the repository. Direct code inspection of all
three previously-flagged files confirmed no `except Exception` / `# noqa: BLE001`
sites remain:

- **`report.py`** — `_safe_plot` catches
  `(TypeError, ValueError, RuntimeError, MemoryError)`.
  `create_report` MCAR block catches
  `(ValueError, TypeError, np.linalg.LinAlgError, ArithmeticError)`.
  Zero broad catches.

- **`compare.py`** — `compare_imputations` method loop catches
  `(ValueError, TypeError, RuntimeError, MemoryError, np.linalg.LinAlgError)`.
  `cv_compare_imputations` contains no try/except at all.
  Zero broad catches.

- **`diagnostics.py`** — `diagnose_missing` inner call catches
  `(ValueError, np.linalg.LinAlgError, ArithmeticError)` with a `# pragma: no cover`
  comment (numerical edge-case, not a BLE001 suppression).
  Zero broad catches.

**Status corrected:** C1 is `[x] DONE` — all exception handling is already specific
throughout the codebase. The session-2 assessment was based on an earlier code state
that had since been fixed.

**Updated execution order going forward:**
```
[x] C1 → [x] C2 → [x] X1 → [x] X2 → [x] X3 → [x] X4 →
[ ] A2 (__all__ hygiene + delete conftest.py) →
[ ] A1 (finish: delete stats.py / summary.py shims, fix __init__.py + accessor.py imports) →
[ ] C4 (create config.py OR scrub missingly.config refs from docstrings) →
[ ] T1 (test layout restructure + coverage ≥ 85%) →
[ ] C3 → [ ] C5 → [ ] A3 → [ ] T2 → [ ] T3 → [ ] T4
```

**Next recommended step: A2 (`__all__` hygiene)**
The `conftest.py` workaround actively harms pytest collection. Fixing `__all__` unblocks
both T1 (test restructure) and removes the last source-of-confusion in the public
namespace. It is a small, self-contained change with high leverage.

---

## Status Legend

- `[ ] TODO` — not started / not yet in code
- `[~] IN PROGRESS` — partially implemented / under refactor
- `[x] DONE` — implemented and verified in source

---

## P0 — Critical (correctness & production safety)

### [x] C1. Explicit error handling
- All `except` blocks throughout the codebase use specific exception types.
  - `impute.py`: catches `(ValueError, np.linalg.LinAlgError, NotFittedError)`,
    logs fallbacks at INFO, honours `strict_mode`.
  - `report.py`: `_safe_plot` catches `(TypeError, ValueError, RuntimeError, MemoryError)`;
    MCAR block catches `(ValueError, TypeError, np.linalg.LinAlgError, ArithmeticError)`.
  - `compare.py`: method loop catches
    `(ValueError, TypeError, RuntimeError, MemoryError, np.linalg.LinAlgError)`.
  - `diagnostics.py`: `diagnose_missing` catches
    `(ValueError, np.linalg.LinAlgError, ArithmeticError)` for numerical edge-cases.
- **No broad `except Exception` / `except BaseException` anywhere in the codebase.**
- **Acceptance:**
  - `ruff` rule `BLE001` enabled in `pyproject.toml` and passing with no `# noqa`.
  - Unit test per exception path (tracked under T1).

### [x] C2. Input validation layer
- `_validation.py` and `exceptions.py` exist, are complete, and are wired into the
  imputation and diagnostics layers. All five exception classes present and tested.
- **No further work required for C2 itself.**

### [x] X1. FittedImputer strategy — FIXED (see session 3 log)
### [x] X2. MissinglyImputer sklearn contract — FIXED (see session 3 log)
### [x] X3. Accessor mar_mnar_test — FIXED (see session 3 log)
### [x] X4. Categorical fallback in _impute_column_by_column — FIXED (see session 3 log)

---

## P1 — Architecture (structural; must precede C3)

### [~] A1. Facets-based module layout

**This item is promoted to P1 and must be completed before C3.**

**Current state:** `diagnostics.py` has merged stats + summary logic. `stats.py` and
`summary.py` survive as `__getattr__` shims. Both `__init__.py` and `accessor.py` still
import the doomed shim modules.

Target layout:

```
missingly/
├── __init__.py          # public surface only; registers accessor
├── exceptions.py        # [x] DONE — ImputationError, InsufficientDataError, …
├── _validation.py       # [x] DONE — shared validators
├── config.py            # [ ] TODO — MissinglyConfig, get_config, set_config (C4)
├── diagnostics.py       # [x] DONE — merged stats + summary logic
├── impute.py            # [x] DONE — X1, X4 fixed
├── compare.py           # [x] DONE — specific exception handling
├── mi.py                # keep
├── mi_workflow.py       # keep
├── accessor.py          # [x] DONE — X3 fixed
├── report.py            # [x] DONE — specific exception handling
├── simulate.py          # keep
├── timeseries.py        # keep
├── transformer.py       # [x] DONE — X2 fixed
├── _deprecation.py      # keep
├── _distance.py         # DECIDE during C5
├── visualise.py         # keep
├── visualisation/       # keep
└── templates/           # keep
```

**Files to delete or migrate (remaining work for A1):**
- `stats.py` → merge into `diagnostics.py`, then delete
- `stats_extra.py` → merge in-scope functions into `diagnostics.py`, then delete
- `summary.py` → merge into `diagnostics.py`, then delete
- `performance.py` → delete after v0.3.0
- `manipulation.py` → move to `missingly/utils/manipulation.py` with deprecation notice
- `manual_tests/` → convert to `tests/integration/` or delete; directory must not exist
- `missingly/conftest.py` → **delete** after A2 __all__ hygiene removes `test_*` symbols
- Fix `__init__.py` and `accessor.py` imports: remove all references to shim modules

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
- `MissinglyImputer` must fully honour the sklearn API. **Prerequisite: X2 is now
  complete. A3 can proceed once A2 is done.**
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
