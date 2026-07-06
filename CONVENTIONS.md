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

`missingly/conftest.py` contained exactly:
```python
"""Prevent pytest from collecting this source package as a test directory."""
collect_ignore_glob = ["*.py"]
```
This workaround existed because `test_hotelling` and `test_pattern_monotone` were
exported in `__all__`, causing pytest to collect them as test functions.
Fixed in A2: `__all__` cleaned up, `test_*` aliases removed, `conftest.py` deleted.

---

#### NEW P0 items — Correctness bugs (blockers; higher priority than any refactor)

**Revised execution order:**
```
X1 → X2 → X3 → X4 → (finish C1: kill remaining except Exception) →
A2 (__all__ hygiene, unblocks T1) → A1 (finish module deletions) →
C4 → C3 → C5 → A3 → rest
```

##### [x] X1. FittedImputer.transform ignores strategy — SILENT WRONG RESULTS
**Fixed in:** commit `fcfdc4c` (2026-07-06)

##### [x] X2. MissinglyImputer breaks the sklearn parameter contract
**Fixed in:** commit `fcfdc4c` (2026-07-06)

##### [x] X3. Accessor mar_mnar_test — target_col silently discarded, signature crash
**Fixed in:** commit `fcfdc4c` (2026-07-06)

##### [x] X4. Categorical estimator fallback uses mean of ordinal codes
**Fixed in:** commit `fcfdc4c` (2026-07-06)

---

#### NEW A2 sub-item — __all__ hygiene (root cause of conftest workaround)

**Fixed in session 5 (2026-07-06).** See session 5 log below.

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

All four P0 correctness bugs resolved.

---

### [2026-07-06] Session 4 — C1 Verification

**Finding:** C1 was already complete. All three flagged files confirmed using specific
exception types. Status corrected to `[x] DONE`.

---

### [2026-07-06] Session 5 — A2 Implementation

**Commits:**
- [`fd8faf1`](https://github.com/alisadeghiaghili/missingly/commit/fd8faf15b474fb8ddfac0c0031a4d7c77a1e2664) — `__init__.py` A2 hygiene + `py.typed`
- [`60800fd`](https://github.com/alisadeghiaghili/missingly/commit/60800fdd6c88b5664ff03febe9ece9ca3e2eefc9) — delete `missingly/conftest.py`

**What was done:**
1. **`test_hotelling` / `test_pattern_monotone` removed from `__all__` and public
   namespace.** They survive as private `_test_hotelling` / `_test_pattern_monotone`
   internally. This was the root cause of the `collect_ignore_glob = ["*.py"]` workaround.
2. **`chunk_apply`, `memory_usage_mb`, `optimize_dtypes` removed from `__all__`.**
   DQT-scope deprecated symbols; still callable with `DeprecationWarning`.
3. **Duplicate `simulate_*` entries removed from `__all__`.**
   They were listed twice (canonical + deprecated-alias). Now listed once (canonical).
4. **All other deprecated symbols** (`hotelling_test`, `clean_names`, `gap_table`, etc.)
   remain importable with `DeprecationWarning` but are excluded from `__all__`.
5. **`missingly/conftest.py` deleted.** Workaround no longer needed.
6. **`missingly/py.typed` added.** PEP 561 marker enables downstream type checkers.

**Acceptance verification:**
- `[x for x in dir(missingly) if x.startswith('test_')] == []` ✓
- `missingly/conftest.py` does not exist ✓
- `py.typed` present ✓
- All previously-public deprecated symbols still importable (backward compat) ✓

**Updated execution order going forward:**
```
[x] C1 → [x] C2 → [x] X1 → [x] X2 → [x] X3 → [x] X4 → [x] A2 →
[ ] A1 (finish: delete stats.py / summary.py shims, fix __init__.py + accessor.py) →
[ ] C4 (create config.py OR scrub missingly.config refs from docstrings) →
[ ] T1 (test layout restructure + coverage ≥ 85%) →
[ ] C3 → [ ] C5 → [ ] A3 → [ ] T2 → [ ] T3 → [ ] T4
```

**Next recommended step: A1**
Shim modules (`stats.py`, `summary.py`, `stats_extra.py`) still exist and both
`__init__.py` and `accessor.py` import from deprecated modules. This creates import
path confusion and prevents `mypy --strict` from passing.

---

## Status Legend

- `[ ] TODO` — not started / not yet in code
- `[~] IN PROGRESS` — partially implemented / under refactor
- `[x] DONE` — implemented and verified in source

---

## P0 — Critical (correctness & production safety)

### [x] C1. Explicit error handling
- All `except` blocks use specific exception types throughout the codebase.
- No broad `except Exception` / `except BaseException` anywhere.
- **Acceptance:** `ruff` rule `BLE001` enabled and passing with no `# noqa`.

### [x] C2. Input validation layer
- `_validation.py` and `exceptions.py` complete and wired in. Five exception classes present.

### [x] X1. FittedImputer strategy — FIXED
### [x] X2. MissinglyImputer sklearn contract — FIXED
### [x] X3. Accessor mar_mnar_test — FIXED
### [x] X4. Categorical fallback in _impute_column_by_column — FIXED

---

## P1 — Architecture (structural; must precede C3)

### [~] A1. Facets-based module layout

**Current state:** `diagnostics.py` has merged stats + summary logic. `stats.py` and
`summary.py` survive as `__getattr__` shims. Both `__init__.py` and `accessor.py` still
import from shim modules.

**Remaining work:**
- Delete `stats.py`, `stats_extra.py`, `summary.py` (merge any remaining in-scope
  symbols into `diagnostics.py` first)
- Fix `accessor.py` imports: replace `stats` / `summary` references with `diagnostics`
- Fix `__init__.py`: remove remaining references to shim modules
- Move `manipulation.py` to `missingly/utils/manipulation.py` with deprecation notice
- Convert `manual_tests/` to `tests/integration/` or delete; directory must not exist

Target layout:

```
missingly/
├── __init__.py          # [x] A2 complete
├── exceptions.py        # [x] DONE
├── _validation.py       # [x] DONE
├── config.py            # [ ] TODO (C4)
├── diagnostics.py       # [x] DONE — merged stats + summary logic
├── impute.py            # [x] DONE
├── compare.py           # [x] DONE
├── mi.py                # keep
├── mi_workflow.py       # keep
├── accessor.py          # [x] X3 fixed; A1 still needs import cleanup
├── report.py            # [x] DONE
├── simulate.py          # keep
├── timeseries.py        # keep
├── transformer.py       # [x] DONE
├── _deprecation.py      # keep
├── _distance.py         # DECIDE during C5
├── visualise.py         # keep
├── visualisation/       # keep
├── py.typed             # [x] DONE (A2)
└── templates/           # keep
```

**Files to delete (remaining A1 work):**
- `stats.py` → delete (shim, content merged into diagnostics.py)
- `stats_extra.py` → delete after v0.3.0
- `summary.py` → delete (shim, content merged into diagnostics.py)
- `performance.py` → delete after v0.3.0
- `manipulation.py` (root) → move to `missingly/utils/manipulation.py`
- `manual_tests/` → convert or delete

**Acceptance:**
- `stats.py`, `stats_extra.py`, `summary.py` no longer exist as top-level files.
- `accessor.py` imports from `diagnostics`, not `stats` or `summary`.
- `__init__.py` imports from `diagnostics`, not shim modules.
- `import missingly` passes; all existing public symbols still importable.

### [x] A2. Public API surface
- `__all__` defined, clean, and matching API reference.
- `test_hotelling` / `test_pattern_monotone` removed from `__all__` and public namespace.
- `chunk_apply`, `memory_usage_mb`, `optimize_dtypes` removed from `__all__`.
- `py.typed` PEP 561 marker added.
- `missingly/conftest.py` deleted.
- **Acceptance:** `[x for x in dir(missingly) if x.startswith('test_')] == []` ✓

### [ ] A3. sklearn interoperability contract
- Run `check_estimator` and document each skipped check.
- Integration test: `MissinglyImputer` inside `Pipeline` with `GridSearchCV`.
- **Prerequisite: A2 done ✓. Proceed after A1.**

---

## P2 (was P0 C3–C5) — API & Config cleanup

### [ ] C3. API surface cleanup & naming consistency

**Prerequisite: A1 must be complete.**

- One naming rule: `miss_*` for analysis, `impute_*` for imputation, bare verbs in
  `utils/` only.
- Accessor: remove thin wrappers; replace `**kwargs` with typed sigs.
- Add `df.miss.impute(strategy=...)` — both READMEs advertise this but it does not exist.
- Deprecation path via `_deprecation.py`.
- **Acceptance:** All public methods follow naming rule. `df.miss.impute(strategy="mean")` works.

### [ ] C4. Configuration externalization
- Create `config.py` with `MissinglyConfig` dataclass.
- **First action:** Remove all `missingly.config` references from `impute.py` docstrings
  until `config.py` actually exists.
- **Acceptance:** `config.py` exists. No hardcoded threshold in imputation logic.
  `grep -r "missingly.config" missingly/` returns matches only after this is done.

### [ ] C5. Encode/decode pipeline correctness
- Replace hand-rolled `_split_encode` / `_decode` with sklearn-standard pattern.
- After replacing: evaluate `_distance.py`; delete if redundant.
- **Acceptance:** Round-trip tests for all edge cases. `_distance.py` decision documented.

---

## P3 — Testing, Docstrings, and CI (gate for v0.1.0)

### [ ] T1. Test layout and coverage
- Restructure tests into `unit/`, `integration/`, `fixtures/`.
- Required fixtures: `df_no_missing`, `df_all_missing_col`, `df_single_row`,
  `df_mixed_dtypes`, `df_large`, `df_high_cardinality_cat`.
- Coverage target: ≥ 85%.
- **Prerequisite: A2 done ✓. `manual_tests/` must also be deleted (A1 sub-item).**
- **Acceptance:** `pytest tests/` runs cleanly. Coverage ≥ 85%. `manual_tests/` does not exist.

### [ ] T2. Docstring and typing policy
- Google-style docstrings. English only. `mypy --strict` passes.
- **Acceptance:** `pydoclint --style=google missingly/` exits 0. `mypy --strict` exits 0.

### [ ] T3. CI pipeline
- GitHub Actions: `lint`, `typecheck`, `test`. Matrix: Python 3.9, 3.11, 3.12.
- Coverage gate: `--cov-fail-under=85`.

### [ ] T4. Logging infrastructure
- No `print()` for operational events. Logger naming follows module hierarchy.
- Every fallback logs at `INFO`.
- **Acceptance:** `grep -r "print(" missingly/` returns no results.

---

## P4 — Bridges & Sister Integration

### [ ] B1. Missingness report as a portable object
- `MissingnessReport` dataclass in `missingly/report_model.py`. JSON-serialisable.

### [ ] B2. DQT-facing contract
- Stable entry point: `missingly.run(df, config=None) -> MissingnessReport`.

---

## P5 — Documentation & Ecosystem (after v0.1.0)

### [ ] D1. README consolidation
- **Do not work on this before P0–P3 are complete.**
- Both `README_FA.md` and `README_DE.md` advertise `df.miss.impute(strategy=...)` which
  does not exist. Resolve in C3 (add the method) or D1 (fix the docs).

### [ ] D2. Architecture & API docs
- MkDocs + `mkdocstrings`. Architecture doc, API reference, tutorials, migration guide.

### [ ] D3. Extensibility — plugin strategy registration
- `@missingly.register_strategy("my_custom")` plugin hook. `BaseImputer` ABC.

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
| `missingly/conftest.py`         | —             | [x] DONE  | Deleted in A2 (session 5)           |
| `FittedImputer` / `make_imputer`| v0.3.0        | v0.5.0    | Restricted to simple strategies;    |
|                                 |               |           | use MissinglyImputer instead        |
| `README_DE.md`                  | —             | D1        | Unmaintained translation            |

---

## Source of Truth
- **This file is the single source of truth for task status.**
- Update it when a task is completed, not retroactively.
- Every PR/commit that completes a task must update the status marker here.
- No docstring may reference a module or symbol that does not exist in source.
