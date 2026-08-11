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

## [2026-08-11] Verified critical-recovery program

### Evidence status

An independent source review and local reproductions confirmed that a passing test suite
is not sufficient evidence of release readiness.  At the time of this entry, ``pytest
tests -q`` reports 1,045 passing tests, but total line coverage is 81% (``impute.py``
57%, ``naniar.py`` 11%), doctest collection fails, and several untested paths can
silently corrupt observed data or return statistically misleading output.

The local ``main`` ref was found to point to an unrelated historical project.  Its old
tip is preserved as ``backup/local-main-before-origin-sync``; local ``main`` now tracks
``origin/main``.  This is repository hygiene, not a claim that the release blockers are
resolved.

### Non-negotiable execution rules

- Every defect starts with a minimal failing regression test.  A passing smoke test is
  never evidence that a statistical method is correct.
- Each bounded change uses a dedicated branch, a Conventional Commit with the accurate
  type (``test:``, ``fix:``, ``feat:``, ``docs:``, or ``ci:``), a focused PR, and a
  linked issue.  Do not label ordinary code or documentation commits as ``ci:``.
- A PR may merge only after its required checks pass, its acceptance criteria are
  verified against source, and this document is updated when it changes roadmap status.
- No unexplained failure, warning, skip, xfail, or unsupported public claim is accepted
  as release-ready.  Temporary exclusions require an issue, rationale, and expiry.
- Every completed phase reports its measured parity or remaining gap relative to the
  named competitor capability; feature-count claims are not leadership claims.

### Recovery sequence

| Phase | Scope | Exit evidence |
|---|---|---|
| R0 | Repository/release truth: branch safety, issue/PR traceability, factual public claims | Canonical refs verified; unsupported claims marked experimental or removed; roadmap status matches source |
| R1 | TDD correctness firewall | Reproducers for every confirmed P0/P1 defect are committed before implementation |
| R2 | Serving data integrity | No observed value is overwritten; schema/order/dtype and unseen-category policies are deterministic |
| R3 | Diagnostics and pooling guardrails | Invalid/inconclusive diagnostics cannot report success; invalid variance inputs fail loudly |
| R4 | FCS/PMM statistical engine | Iterative conditional models, uncertainty draws, constraints, and locked cross-tool conformance evidence |
| R5 | sklearn and public API contract | Estimator checks, Pipeline/GridSearchCV, installed-package examples, and explicit container exclusions |
| R6 | Documentation and quality governance | Config/API/docs agree with source; doctest, lint, typing, docstring, and coverage gates are enforced |
| R7 | Large-data execution | Measured pandas/Arrow/Polars capability adapters with bounded-memory and numerical-parity tests |
| R8 | Competitor parity and release candidate | Published benchmark manifests, statistical conformance, security/offline audit, and versioned release decision |

### Active issue map

- #43 — P0 data integrity and serving-transform regressions.
- #44 — P0 statistical correctness and diagnostic guardrails.
- #45 — P1 public contracts, documentation, and enforceable quality gates.
- #36 — locked R oracle provenance and reproducibility.
- #40 — release preparation; it remains blocked by the applicable recovery phases.

The PR for R-oracle provenance is intentionally independent of the recovery work.  Do
not mix benchmark provenance changes with correctness fixes in the same pull request.

---

## Session Log

### [2026-08-03] Unrelated-branch reconciliation and statistical credibility

- Verified that `origin/main` and `agent/missingly-analytics-upgrade` have unrelated Git
  histories; the package branch remains canonical and the web/statistics product is not
  force-merged.
- Selectively ported the compatible strengths: evidence-gated competitor analysis,
  statistical conformance policy, and deterministic provenance.
- Added a keep/drop audit in `BRANCH_INTEGRATION.md`; FastAPI, auth, payments, general
  statistics, and product UI remain outside the missing-data-only scope.
- Removed placeholder-only skips and the external Gower test skip. Optional DQT tests
  remain conditional locally and are mandatory in their dedicated CI job.
- Corrected Rubin pooling terminology and behavior: large-sample Rubin degrees of
  freedom, distinct `lambda` and `fmi`, and finite/non-negative input guardrails.
- T1 remains open until measured line coverage reaches at least 85%; passing-test count
  alone is not a coverage claim.

---

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
1. `test_hotelling` / `test_pattern_monotone` removed from `__all__` and public namespace.
2. `chunk_apply`, `memory_usage_mb`, `optimize_dtypes` removed from `__all__`.
3. Duplicate `simulate_*` entries removed from `__all__`.
4. All other deprecated symbols remain callable but excluded from `__all__`.
5. `missingly/conftest.py` deleted.
6. `missingly/py.typed` added (PEP 561).

---

### [2026-07-06] Session 6 — A1 Implementation

**Commits:**
- [`1b18a33`](https://github.com/alisadeghiaghili/missingly/commit/1b18a33c4ab4d4abe0a2411bc965f873776a891c) — `accessor.py` imports `diagnostics` directly; remove `stats`/`summary` imports
- [`7bbcf67`](https://github.com/alisadeghiaghili/missingly/commit/7bbcf67adbe0e0833b192e9f80e9b33ec4450c63) — delete `missingly/stats.py` shim
- [`465be24`](https://github.com/alisadeghiaghili/missingly/commit/465be243a0604d946c681730f70bdaf4ce52ff1c) — delete `missingly/summary.py` shim

**What was done:**
1. **`accessor.py` refactored**: `from . import summary, stats` replaced with
   `from . import diagnostics`. All 9 call sites updated:
   - `summary.n_miss` → `diagnostics.n_miss`
   - `summary.n_complete` → `diagnostics.n_complete`
   - `summary.pct_miss` → `diagnostics.pct_miss`
   - `summary.pct_complete` → `diagnostics.pct_complete`
   - `summary.miss_var_summary` → `diagnostics.miss_var_summary`
   - `summary.miss_case_summary` → `diagnostics.miss_case_summary`
   - `summary.bind_shadow` → `diagnostics.bind_shadow`
   - `stats.mcar_test` → `diagnostics.mcar_test`
   - `mar_mnar_test` was already using `diagnostics` directly (X3 fix)
2. **`stats.py` deleted** — was a pure `__getattr__` shim forwarding to `diagnostics`.
3. **`summary.py` deleted** — was a pure `__getattr__` shim forwarding to `diagnostics`.

**Remaining A1 work (deferred to A1b or folded into later tasks):**
- `stats_extra.py` and `performance.py` — deprecated shims; scheduled for deletion at v0.3.0
- `manipulation.py` (root) — move to `missingly/utils/manipulation.py` (done in C3)
- `manual_tests/` — convert or delete (done in T1)

**Acceptance verification:**
- `accessor.py` imports: `from . import diagnostics, manipulation, visualise` ✓
- `stats.py` does not exist ✓
- `summary.py` does not exist ✓
- `import missingly; df.miss.n_miss()` path: `diagnostics.n_miss` ✓
- `import missingly; df.miss.mcar_test()` path: `diagnostics.mcar_test` ✓

**Updated execution order going forward:**
```
[x] C1 → [x] C2 → [x] X1 → [x] X2 → [x] X3 → [x] X4 → [x] A2 → [x] A1 →
[x] C4 (config.py verified present and complete — see session 7) →
[ ] T1 (test layout restructure + coverage ≥ 85%) →
[ ] C3 → [ ] C5 → [ ] A3 → [ ] T2 → [ ] T3 → [ ] T4
```

**Next recommended step: C4**
`impute.py` docstrings reference `missingly.config` which does not exist.
Either create `config.py` with `MissinglyConfig` (the real fix, ≈ 40 lines)
or remove the references (the fast fix). C4 unblocks T2 (mypy --strict).

---

### [2026-07-11] Session 7 — C4 Verification

**Finding:** `config.py` was already present and complete (SHA: `700712b`).
The previous session log incorrectly left C4 as `[ ] TODO`. This session corrects
the record after direct source verification.

**Verified contents of `missingly/config.py`:**
- `MissinglyConfig` dataclass with three fields:
  - `large_df_threshold: int = 50_000`
  - `knn_cat_neighbors_threshold: int = 5`
  - `strict_mode: bool = False`
- Package-level singleton `config: MissinglyConfig = MissinglyConfig()`
- `__all__ = ["MissinglyConfig", "config"]`
- No imports from other `missingly` sub-modules (circular-import safe)
- Module-level docstring documents runtime mutation pattern

**Verified that `impute.py` correctly imports and uses config:**
- `from missingly.config import config as _config` ✓
- `_warn_if_large` uses `_config.large_df_threshold` ✓
- `_warn_if_knn_heavy_categorical` uses `_config.knn_cat_neighbors_threshold` ✓
- `_impute_column_by_column` uses `_config.strict_mode` as fallback ✓

**`grep -r "missingly.config" missingly/` now resolves to real code.** ✓

**Status corrected:** C4 → `[x] DONE`

**Updated execution order:**
```
[x] C1 → [x] C2 → [x] X1–X4 → [x] A2 → [x] A1 → [x] C4 →
[ ] T1 → [ ] C3 → [ ] C5 → [ ] A3 → [ ] T2 → [ ] T3 → [ ] T4
```

**Next recommended step: T1**
Test layout restructure into `unit/`, `integration/`, `fixtures/`.
Coverage target ≥ 85%. Delete `manual_tests/`. Prerequisite: A1 done ✓, C4 done ✓.

---

### [2026-08-08] Critical statistical and competitive audit

**Verdict:** Missingly has a strong integrated Python surface, but it must not claim
global superiority or statistical parity with `mice`, SAS, SPSS, or Stata yet. The
immediate priority is correctness and evidence for existing functionality, not adding
more methods.

#### P0 — statistical validity and leakage blockers

1. **MAR versus MNAR is not identifiable from observed data alone.**
   `mar_mnar_test` must not claim to distinguish MAR from MNAR. Replace it with an
   explicitly limited observed-data missingness-association diagnostic, add references,
   and deprecate the misleading name. `diagnose_missing` must not infer MCAR, MAR, or
   MNAR from a p-value threshold or a nullity-correlation threshold. It may report
   evidence against MCAR and require an explicit, user-supplied assumption for the
   analysis model.
2. **Little's MCAR test is release-blocking until independently validated.**
   Validate the EM estimate, degrees of freedom, convergence behavior, singular and
   high-dimensional cases against frozen R fixtures. A non-rejection is not proof of
   MCAR. Issue #22 is P0, not documentation-only work.
3. **The MICE history path must generate genuine stochastic chains.**
   Regression tests must prove different chains differ when posterior draws are
   requested, and must reject the prior failure mode where identical histories produced
   an apparently acceptable R-hat. Convergence output is a diagnostic, never a proof of
   validity.
4. **PMM and categorical "MICE method" claims require a single FCS engine.**
   The standalone PMM/logreg/polyreg/polr functions must not be described as equivalent
   to R `mice` until they use iterative conditional models, uncertainty draws, correct
   donor matching, and cross-tool fixtures. A loop whose missing mask is exhausted on
   its first pass is not a chained-equations implementation.
5. **`MissinglyImputer` must have explicit train/transform semantics.**
   No transform path may refit on evaluation data or use evaluation rows as donors in
   inductive mode. Add anti-leakage tests for every strategy. If transductive imputation
   is supported, it must be opt-in, named, documented, and excluded from ordinary
   sklearn pipeline/CV use.

#### Evidence and test policy additions

- Every advertised inferential method needs an independent numerical oracle, not only
  an invariant or smoke test. Cross-tool fixtures record dataset licence/SHA-256,
  exact tool version, platform, seed, missing-data policy, estimand, uncertainty,
  tolerances, and known algorithmic differences.
- Simulation tests must measure bias, empirical confidence-interval coverage, Type-I
  error, donor/distribution preservation, convergence failure behavior, and leakage.
  Add monotone/non-monotone, high-dimensional, collinear, rare-category, unseen-level,
  structural-missingness, weighted, clustered, and longitudinal fixtures.
- Add property-based, metamorphic, mutation, minimum-dependency, installed-wheel, and
  cross-platform tests. Passing test count and line coverage are not statistical
  evidence.
- The release coverage floor is **85%**, with a non-decreasing ratchet. CI also runs
  formatting/linting, type checking, docstring-contract checks, dependency auditing,
  package build validation, and docs builds on supported Windows, macOS, and Linux
  combinations.

#### Competitive scope additions

The competitor analysis must be a capability registry, not a short list of famous
packages. It must cover these families before any broad leadership claim:

- likelihood/FIML/EM and joint Bayesian modelling;
- mixed FCS, PMM, predictor matrices, blocks, visit sequences, `where`/`ignore`,
  passive terms, constraints, and substantive-model compatibility;
- multilevel, cross-classified, nested, survey-weighted, and fractional hot-deck MI;
- Barnard--Rubin and Reiter pooling, joint tests, model adapters, contrasts,
  predictions, and marginal effects;
- MNAR sensitivity: delta adjustment, tipping points, NARFCS, pattern-mixture,
  selection, and reference-based longitudinal models;
- iterative forests/OOB diagnostics, matrix completion (PCA/MCA/MFA/SVD), copula,
  AutoML/deep methods, and plugin-backed specialist engines;
- time-series, spatial, survival, censored/limit-of-detection, count, compositional,
  graph/network, clinical-trial, causal, and complex-survey missing-data workflows;
- interoperability with pandas, sklearn, Polars/Arrow, sparse data, R/SAS/SPSS/Stata
  interchange, and long/wide multiple-imputation storage.

The core package remains missing-data focused. Specialist methods belong behind a
versioned plugin protocol rather than growing a second "god package". A capability can
be called *better* only when it has parity evidence plus a reproducible safety,
usability, performance, or memory advantage on a published benchmark.

#### Architecture and release policy additions

- Introduce typed `MissingnessSchema`, `ImputationPlan`, `FCSKernel`, `MIData`, and
  `ImputationResult` contracts. Results retain the original mask, method choices,
  seed sequence, traces, warnings, constraints, provenance, and validation outcome.
- Enforce sklearn estimator compliance deliberately: feature names, `set_output`,
  metadata routing, supported containers, and documented exclusions.
- Provenance is reproducibility metadata, **not** anonymisation. Add environment,
  dependency, platform, RNG, code revision, and output identity; offer keyed/HMAC
  fingerprints for sensitive low-entropy inputs.
- Runtime downloads must be opt-in or integrity-checked and offline-safe. Do not make
  a plotting call silently download an unpinned external asset.
- Release status must be factual. `v1.0.0` publication, online documentation, issue
  state, CI state, and deprecation milestones must be reconciled before declaring a
  release complete.

#### Large-data execution policy

Large-data support is a first-class R5 track, not an unverified claim that existing
pandas code "supports 10M rows". The architecture is capability-based:

- pandas remains the reference backend for public semantics and numerical parity;
- Apache Arrow/DataFrame Interchange is the container boundary where zero-copy or
  bounded-copy conversion is possible;
- optional Polars `LazyFrame` is the preferred single-machine streaming backend for
  projection, null counting, grouped summaries, pattern aggregation, and other
  operations that can be expressed without materialising the full input;
- a chunked pandas/Arrow fallback covers file and iterator sources when Polars is not
  installed; Polars must remain an optional extra, never a core dependency;
- Dask or another distributed engine is admitted only after reproducible benchmarks
  show that single-machine streaming is insufficient. No backend-specific conditionals
  may spread through statistical kernels; adapters expose explicit capabilities.

Every operation must declare one of four execution classes: exact-streaming,
bounded-memory batch, explicitly approximate, or materialising. Approximation and
sampling are opt-in, seeded, labelled in results, and include error/coverage metadata
where applicable. Algorithms such as exact KNN, dense nullity correlation, MICE, and
forest imputation must not silently claim streaming behavior; they either use a
documented scalable variant or fail before allocation with an actionable memory-budget
error.

R5 acceptance requires:

1. Exact summary results are invariant across pandas, Arrow/chunked, and Polars paths
   on shared fixtures, including mixed dtypes, sentinels, all-missing columns, and
   partition boundaries.
2. A public execution policy exposes backend choice, memory budget, chunk size,
   deterministic seed, approximation status, and copy/materialisation behavior.
3. Pull requests run a 1M-row smoke benchmark; scheduled/release jobs run documented
   10M-row narrow and wide Parquet benchmarks and record wall time, peak RSS, input
   bytes, extra-memory ratio, output identity, backend/version, and hardware.
4. Streaming summaries on 10M rows do not construct a dense shadow matrix and keep
   peak additional memory bounded by the configured budget. Benchmark failure is a
   release failure, not a warning hidden in logs.
5. Cancellation, partial-read failure, invalid schema, unsupported backend capability,
   and out-of-budget behavior have deterministic tests and actionable exceptions.
6. Performance claims name the dataset and competitor version and require numerical
   parity plus a statistically reported speed or memory advantage.

#### Revised execution order

```
R0 truth/release repair → R1 statistical-correctness firewall →
R2 benchmark and evidence platform → R3 unified MI architecture →
R4 algorithmic parity tracks → R5 ecosystem/scale → R6 measured leadership claims
```

**R0 acceptance:** CI is green, public documentation is live and executable, release/
checklist state is reconciled with source, and unsupported statistical claims are
removed or clearly marked experimental. **Current status: `[~] IN PROGRESS`**. The
2026-08-11 review found stale public APIs, a broken configuration export, and quality
gates below this policy; track remediation in #45.

**R1 acceptance:** independent fixtures validate every shipped statistical result;
no API claims to identify MAR versus MNAR; no sklearn strategy leaks transform data;
and MICE chains/diagnostics have end-to-end stochastic regression tests.

**R1 status:** earlier work in merged PRs #27, #28, and #31 resolved the specific
mechanism, MCAR-oracle, and anti-leakage failures they covered. **It is not a blanket
correctness sign-off.** The 2026-08-11 reproductions reopen untested serving,
hot-deck, diagnostic, pooling, and standalone-MI paths under #43 and #44. R2 evidence
infrastructure begins with issue #36; FCS/PMM parity remains R4 work and is not implied
by prior R1 work.

**R3 status:** issue #37 introduces the typed MI contracts before any legacy imputer
is migrated. The compatibility boundary is deliberate: contracts must be stable and
tested before they become the public return type of an FCS engine.

**Documentation deployment policy:** GitHub Pages is the canonical public site. Its
workflow builds Sphinx with warnings treated as errors, uploads an immutable Pages
artifact, and deploys only from ``main`` using GitHub's protected ``github-pages``
environment. The repository Pages source must be configured as GitHub Actions.

**R5 acceptance:** exact streaming diagnostics, bounded-memory execution, optional
Polars/Arrow interoperability, and reproducible 1M/10M-row benchmarks satisfy the
large-data policy above without weakening pandas semantics or statistical validity.

---

## Status Legend

- `[ ] TODO` — not started / not yet in code
- `[~] IN PROGRESS` — partially implemented / under refactor
- `[x] DONE` — implemented and verified in source

---

## P0 — Critical (correctness & production safety)

### [~] C1. Explicit error handling
- Broad ``except Exception`` remains in imputation and visualisation paths, including
  fallbacks that can hide estimator failures.  Existing ``# noqa: BLE001`` markers do
  not satisfy this task.
- **Acceptance:** `ruff` rule `BLE001` is enabled in the declared development tooling,
  runs in CI, and passes with no broad-exception suppression outside a documented,
  narrowly scoped boundary.
- **Tracking:** #44 for imputation failures and #45 for the enforced gate.

### [x] C2. Input validation layer
- `_validation.py` and `exceptions.py` complete and wired in. Five exception classes present.

### [x] X1. FittedImputer strategy — FIXED
### [x] X2. MissinglyImputer sklearn contract — FIXED
### [x] X3. Accessor mar_mnar_test — FIXED
### [x] X4. Categorical fallback in _impute_column_by_column — FIXED

---

## P1 — Architecture (structural; must precede C3)

### [x] A1. Facets-based module layout

**Completed in session 6 (2026-07-06).**

- `accessor.py` now imports directly from `diagnostics` (no shim dependency).
- `stats.py` deleted.
- `summary.py` deleted.

**Remaining deferred items (tracked separately):**
- `stats_extra.py` and `performance.py` — deprecated shims; delete at v0.3.0
- `manipulation.py` (root) — move to `utils/` during C3
- `manual_tests/` — delete during T1

Current module layout (verified):

```
missingly/
├── __init__.py          # [x] A2 complete
├── exceptions.py        # [x] DONE
├── _validation.py       # [x] DONE
├── config.py            # [x] DONE (C4)
├── diagnostics.py       # [x] DONE — canonical stats + summary
├── impute.py            # [x] DONE
├── compare.py           # [x] DONE
├── mi.py                # keep
├── mi_workflow.py       # keep
├── accessor.py          # [x] DONE — imports diagnostics directly
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

**Files deleted:**
- `stats.py` ✓
- `summary.py` ✓
- `missingly/conftest.py` ✓ (A2)

**Files still to delete (v0.3.0 window):**
- `stats_extra.py`
- `performance.py`

### [x] A2. Public API surface
- `__all__` clean. `test_*` removed. `py.typed` added. `conftest.py` deleted.

### [~] A3. sklearn interoperability contract
- Run `check_estimator` and document each skipped check.
- Integration test: `MissinglyImputer` inside `Pipeline` with `GridSearchCV`.
- Implement standard `feature_names_in_`, `n_features_in_`, and
  `get_feature_names_out(input_features=None)` semantics; preserve fit-time column
  order and define unknown-category behavior.
- **Current state:** no leakage was verified for supported inductive strategies, but
  single-row categorical transforms crash, tree predictions depend on transform-time
  column order, and `set_params` can bypass strategy validation. Track in #43/#44.

---

## P2 (was P0 C3–C5) — API & Config cleanup

### [ ] C3. API surface cleanup & naming consistency

**Prerequisite: A1 done ✓.**

- One naming rule: `miss_*` for analysis, `impute_*` for imputation, bare verbs in
  `utils/` only.
- Accessor: remove thin wrappers; replace `**kwargs` with typed sigs.
- Add `df.miss.impute(strategy=...)` — both READMEs advertise this but it does not exist.
- Move `manipulation.py` to `missingly/utils/manipulation.py` here.
- Deprecation path via `_deprecation.py`.
- **Acceptance:** All public methods follow naming rule. `df.miss.impute(strategy="mean")` works.

### [~] C4. Configuration externalization

`config.py` contains the intended dataclass and singleton, and internal imputation code
uses it.  The public contract is incomplete: `import missingly; missingly.config`
currently resolves to the module rather than the documented singleton, and
`MissinglyConfig` is not exported from the package root.

- `config.py` exists with `MissinglyConfig` dataclass.
- Fields: `large_df_threshold=50_000`, `knn_cat_neighbors_threshold=5`, `strict_mode=False`.
- Package singleton `config` and `MissinglyConfig` must be exported from `__init__.py`
  and included in package `__all__`.
- `impute.py` imports and uses `_config` at all three call sites.
- No hardcoded threshold remains in imputation logic.
- **Acceptance:** documented mutations of `missingly.config` change the same singleton
  read by imputation code; an installed-package regression test proves it.
- **Tracking:** #45.

### [~] C5. Encode/decode pipeline correctness
- Replace ad-hoc, independently fitted feature encodings with a fitted, reusable
  sklearn-standard representation.
- Coerce categorical serving columns before encoding; preserve observed unseen values
  rather than turning them into missing data; exclude missing values from decoder
  categories.
- After replacing: evaluate `_distance.py`; delete if redundant.
- **Acceptance:** round-trip, single-row, nullable-string, unseen-level, column-order,
  duplicate-index, and mixed-dtype tests pass. `_distance.py` decision documented.
- **Tracking:** #43 and #44.

---

## P3 — Testing, Docstrings, and CI (gate for v0.1.0)

### [ ] T1. Test layout and coverage
- Restructure tests into `unit/`, `integration/`, `fixtures/`.
- Required fixtures: `df_no_missing`, `df_all_missing_col`, `df_single_row`,
  `df_mixed_dtypes`, `df_large`, `df_high_cardinality_cat`.
- Current measured coverage is 81% (`impute.py` 57%, `naniar.py` 11%). Coverage must
  first reach ≥ 85% without exclusions that hide public paths, then ratchet toward the
  project-wide 90% target.
- **Prerequisite: A1 done ✓. `manual_tests/` must also be deleted here.**
- **Acceptance:** `pytest tests/` runs cleanly. Coverage ≥ 85%. `manual_tests/` does not exist.

### [ ] T2. Docstring and typing policy
- Google-style docstrings. English only. `mypy --strict` passes.
- **Acceptance:** `pydoclint --style=google missingly/` exits 0, `mypy --strict` exits
  0, and doctest collection plus all public examples pass from an installed package.

### [ ] T3. CI pipeline
- GitHub Actions: `lint`, `typecheck`, `docstrings`, `doctest`, `test`, package build,
  dependency audit, and docs build. Matrix: supported Python versions and Windows,
  macOS, and Linux coverage as defined by release policy.
- Coverage gate: `--cov-fail-under=85`, then a non-decreasing ratchet to 90.
- No unexplained warning, skip, xfail, or optional dependency degradation can produce a
  green release check.

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
| `summary.py`                    | A1            | [x] DONE  | Deleted in A1 (session 6)           |
| `stats.py`                      | A1            | [x] DONE  | Deleted in A1 (session 6)           |
| `manipulation.py` (root)        | C3            | v0.4.0    | Out of scope; moved to utils/       |
| `missingly/conftest.py`         | A2            | [x] DONE  | Deleted in A2 (session 5)           |
| `FittedImputer` / `make_imputer`| v0.3.0        | v0.5.0    | Restricted to simple strategies;    |
|                                 |               |           | use MissinglyImputer instead        |
| `README_DE.md`                  | —             | D1        | Unmaintained translation            |

---

## Source of Truth
- GitHub issues and pull requests are the operational source of truth; this file is
  the versioned architecture, quality, and capability roadmap.
- Update it in the same pull request as an implementation or a material audit. Do not
  mark a task complete based only on API resemblance, a smoke test, or a passing local
  test suite.
- Every PR/commit that completes a task must update the status marker here.
- No docstring may reference a module or symbol that does not exist in source.
