# missingly 2.0.0 code audit (2026-11-06)

Method: direct source reads. README claims ignored unless code backs them. Pytest could not be executed (no env with pandas on this machine); test counts are `def test_` counts (parametrisation not expanded).

## Global facts
- Version 2.0.0 (missingly/_version.py:3), license BUSL-1.1 (pyproject.toml), requires Python >=3.9; classifiers list 3.9-3.12, CI matrix runs 3.9-3.14 (.github/workflows/ci.yml).
- Hard deps: numpy, pandas, scipy, scikit-learn, matplotlib, seaborn, statsmodels, jinja2. Optional: plotly (interactive), arabic-reshaper + python-bidi (rtl). Polars is referenced in an error message `missingly[polars]` but there is NO `polars` extra in pyproject.toml.
- Public API: `__all__` in missingly/__init__.py:284-332 holds about 107 names (about 20 plots, 11 summary fns, 6 diagnostics, 8 imputers + 3 hotdeck, FittedImputer/make_imputer, MissinglyImputer, 2 pool fns, 3 mi_workflow fns, 2 compare, create_report, 4 ts fns, 11 naniar-parity fns, 4 simulate, 4 manipulation, 2 provenance, 5 MI/benchmark contract types). Legacy shims (hotelling_test, pattern_monotone_test, missing_correlation_matrix, performance.*, clean_names, remove_empty, coalesce_columns, miss_as_feature) exist but are NOT in `__all__` and emit warnings. `df.miss` accessor: about 45 methods (accessor.py).
- Source size about 14.6k lines (impute.py 2436, static.py 1735, diagnostics.py 1330, transformer.py 965).
- Tests: 1035 `def test_` across 56 files (tests/ and tests/unit/). Largest: test_visualise 80, rtl_base 51, report 51, unit/test_diagnostics 41, unit/test_impute 38.
- Coverage: documented authoritative Python 3.12 CI baseline 92.50% (3846/4158) (CONVENTIONS.md:35, instructions.md:16); CHANGELOG still says 89.64%. CI gate `--cov-fail-under=80` (ci.yml:41). Not re-measured by me.
- Typing: `py.typed` present (PEP 561). mypy in dev extras but CI runs mypy ONLY on missingly/benchmark.py (ci.yml:57, `--follow-imports=skip`). No mypy config in pyproject. So typing = annotations without enforced checking. Ruff runs in CI on missingly+tests.
- CI: ci.yml (lint + pytest matrix 3.9-3.14), r-oracle.yml (R 4.4.2 + naniar 1.1.0 oracle), docs-pages.yml, release.yml.
- Release history (CHANGELOG): 0.1.0 (2025-01-10 per changelog; but first git commit is 2026-04-21, so date is unreliable/back-dated), 0.2.0 2026-04-15, 0.3.0 2026-07-22, 1.0.0 2026-07-29, 2.0.0 (version bump commit ccf9853 for license change; no dated 2.0.0 CHANGELOG entry, a large [Unreleased] block sits above 1.0.0). Only git tag: v1.0.0. PyPI release history is unverified from the repo.
- Git: first commit 2026-04-21; latest commit 2026-09-05 (at time of snapshot); 457 commits since 2025-11-01 (all of history); 444 since 2026-05-01. Essentially one maintainer (Ali Sadeghi Aghili) in a 5-month burst. No release after 1.0.0 except 2.0.0 license bump.
- Design stance: project is rigorously "honest" - many strict contracts, explicit errors, many "does not identify MAR/MNAR" disclaimers; transformer deliberately raises NotImplementedError for unsafe transductive strategies.

## COMPETITOR_ANALYSIS.md (2026-08-08) gaps: closed vs open
Closed or partly closed since:
- Predictor matrix, visit sequence, `where` mask, per-column bounds in `impute_mice` (impute.py:1115-1300) - CLOSED (numeric/ordinal-encoded only).
- PMM: standalone `impute_pmm` (impute.py:1437) with donors; also `strategy="pmm"` in transformer. NOT inside the MICE engine - gap OPEN.
- Barnard-Rubin df: OPEN. mi.py:179 and :338-343 use classic Rubin df `(m-1)(1+1/r)^2`; `fmi` uses `(r+2/(df+3))/(r+1)` (small-sample style but with old df). No `dfcom`.
- Passive terms: OPEN (only mentioned as a "future" note at mi_contracts.py:233).
- MNAR delta/tipping point/pattern-mixture/reference-based: OPEN (no code; grep for tipping/delta/j2r returns nothing in package).
- Kalman/STL/seasonal time-series: OPEN. `impute_ts` supports only ffill/bfill/linear/time/spline (timeseries.py:345).
- Polars: only `polars_missing_summary` (polars_adapter.py:44), eager/lazy null summary; no polars imputation/plots/accessor. Polars extra not declared.
- sklearn `check_estimator`: OPEN, no test uses it. Transformer is sklearn-style with get_params/clone tests, DataFrame-only (rejects arrays/sparse).
- D1/D2/D3 multivariate pooling: OPEN. Pooling = scalar + linear-regression coefficient arrays only.
- Cross-tool oracle: CLOSED for Little's MCAR (R naniar oracle fixture, tests/fixtures/little_mcar, unit/test_little_mcar_oracle.py, r-oracle.yml). R mice fixture validator exists (tests/fixtures/mice_numeric_fcs, tests/test_r_mice_*_validator.py) but is "evidence-only artifact validator" tests, not a proven numerical parity claim.
- Little's MCAR rewritten with real EM (diagnostics.py:491).
- Mixed KNN with internal Gower (_distance.py:27) - CLOSED.
- BenchmarkManifest, provenance fingerprints, typed MI contracts (MIData, ImputationPlan, ImputationResult) - added.

## F1 Missingness visualization - score 8
Impl (missingly/visualisation/static.py, interactive.py, _base.py; timeseries.py): matrix:137, bar:228, miss_case:330, miss_var_pct:389, miss_which:447, upset:491, miss_patterns:650, miss_cooccurrence:708, heatmap:762, dendrogram:924, vis_miss:984, miss_cluster:1063, miss_row_profile:1118, shadow_scatter:1195, scatter_miss:1275, vis_miss_fct:1289, vis_miss_by_group:1328, vis_impute_dist:1378, miss_impute_compare:1482, vis_miss_cumsum_var/case:1532/1575, vis_miss_span:1615, vis_parallel_coords:1669; naniar-style gg_miss_var/case/fct/upset (naniar.py:299-505); TS: vis_ts_miss, vis_gap_lengths, vis_miss_over_time (timeseries.py:197-314); plice plot_mice_convergence (diagnostics.py:1242).
Interactive: Plotly backend for 9 plots only (interactive.py: vis_miss, heatmap, matrix, miss_var_pct, cooccurrence, bar, miss_case, upset, patterns) - optional extra.
Limits: matplotlib/seaborn static; no margin plot / missing-box-plot (VIM marginplot) - `shadow_scatter` + `scatter_miss` approximate; no aggregation plot; RTL fonts handled; offline-only contract. No pixel-level regression tests (smoke/contract tests).
Tests: test_visualise (80), smoke (29), interactive (23), rtl_base (51), rtl_fallback (33), timeseries_visuals (14), static contracts etc. roughly 270.
Justification: breadth matches/exceeds missingno, near VIM+naniar breadth, lacks margin/aggregate plots and ggplot grammar; interactive only partial.

## F2 Descriptive summaries and pattern analysis - score 7
Impl: n_miss/n_complete/pct_miss/pct_complete (diagnostics.py:105-165), miss_var_summary:184, miss_case_summary:224, summarise:264 (group), miss_scan_count:305, miss_summary:357, bind_shadow:68; naniar.py shadow_matrix:29, shadow_long:74, shadow_wide:123, special_missing:163, n_miss_var/case, pct_miss_var; miss_patterns/cooccurrence plots; manipulation.replace_with_na(_all):19/63, add_any_miss_var:92, bind_shadow_matrix:157; timeseries miss_ts_summary:126, gap_table:164 (run lengths); `missing_correlation_matrix` (stats_extra.py:95, shadow correlation, legacy).
Limits: no miss_var_run / miss_case_table / miss_prop_summary exact naniar equivalents; no dedicated md.pattern-style table function (pattern table is within plots/report); fewer group-by variants.
Tests: test_summary 17, test_naniar 7 (thin), test_stats 22, test_manipulation 32, test_timeseries_summary 12.
Score rationale: covers most naniar essentials incl. shadow long/wide, sentinel handling, gap table; not full naniar API.

## F3 Mechanism diagnostics - score 5
Impl: `mcar_test` (diagnostics.py:491): Little's MCAR via observed-data EM mean/cov, returns chi2/df/p, convergence contract, R naniar oracle (tests/unit/test_little_mcar_oracle.py, fixtures/little_mcar). `missingness_association_test` (diagnostics.py:678): per-variable logistic observedness association screen (explicitly not MAR proof). `diagnose_missing`:856 (MCAR + prevalence + nullity correlation). `mar_mnar_test`:808 is a deprecated warning wrapper. Legacy: hotelling_test, pattern_monotone_test (stats_extra.py:28,53).
Limits: only one formal MCAR test (numeric-only, normality assumption); no Jamshidian-Jalal / Hawkins / nonparametric tests, no t-test-per-variable (SPSS MVA style) except via association screen; no mechanism inference by design.
Tests: unit/test_diagnostics 41, unit/test_little_mcar_oracle 8, test_diagnostic_association_contracts 7, test_stats 22.
Rationale: one oracle-validated MCAR test + association screen; honest but narrow vs MissMech/SPSS MVA.

## F4 Single-imputation breadth - score 7
Impl (impute.py): impute_mean:887, median:942, mode:997, knn:1042 (euclid + Gower), pmm:1437, logreg:1579, polyreg:1699, polr:1817, rf:1997, gb:2068; hotdeck random/sequential/weighted (hotdeck.py:154/236/322); ts ffill/bfill/linear/time/spline; mice (as single draw). FittedImputer/make_imputer:2189/2409 for fit/transform.
Limits: no EM/MVN imputation, no regression imputation as named method (via mice estimator), no SVD/softImpute, no LOCF-by-group, no random-forest iterative (impute_rf uses sklearn IterativeImputer with RF default max_iter=1 - i.e. missForest-like but not OOB-converged, impute.py:2057-2066).
Tests: unit/test_impute 38, test_impute 20, knn 11+11, hotdeck 31, standalone 9, model_strategies 5.

## F5 Mixed-type / categorical - score 5
Impl: Gower KNN (_distance.py:27), hotdeck weighted via Gower, logreg/polyreg/polr standalone categorical imputers (statsmodels/sklearn), RF/GB with classifiers (impute.py:2057-2065).
Limits: in `impute_mice`, categoricals are ORDINAL-ENCODED then regressed with BayesianRidge and decoded (impute.py:1313-1345; changelog admits "unvalidated categorical FCS approximation"); result contract flags it (`_mice_conditional_model_contract`). logreg/polyreg/polr are transductive, raise NotImplementedError in transformer (transformer.py:500-521). Honest but the headline MI engine cannot do true categorical FCS.
Tests: knn_mixed 11, conditional_fallback 14, standalone 9.

## F6 MI engine - score 5
Impl: `impute_mice` (impute.py:1115): chains with distinct seeds (seed+i), BayesianRidge(sample_posterior=True) default via sklearn IterativeImputer (fast path 1351-1376) or custom manual loop `_run_mice_with_history` (impute.py ~690) when history/visit_sequence/predictor_matrix/where/bounds used; options: visit_sequence, predictor_matrix (square bool), where mask, bounds, custom estimator (no posterior draws then), return_history, typed `ImputationResult` (mi_contracts.py: MissingnessSchema, ImputationPlan, FCSKernel, MIData, ImputationResult). Convergence: mice_convergence (diagnostics.py:1118, Gelman-Rubin R-hat on per-iter means).
Limits: no PMM method inside MICE (impute_pmm separate, single-chain); no per-variable method dict (one estimator for all); no passive imputation; no blocks; no multilevel; categorical via ordinal-encoding (not logistic/polytomous); custom estimator -> deterministic predictions (not proper MI); fast path ignores predictor_matrix semantics only via branch switch. Not validated against R mice (fixture validator only).
Tests: test_impute_mice 32, mice_convergence 25, edge contracts 4, mi_contracts 14, pmm fitted 11.
Rationale: usable PROPER-ish numeric MI with some mice-like controls; far from mice's 10.

## F7 MI pooling and inference - score 4
Impl: pool_scalar_estimates (mi.py:33): Q-bar, U-bar, B, T, classic Rubin df, r, lambda, fmi (mi.py:158-191), degenerate cases handled; pool_linear_regression_results (mi.py:199): coef vector + full total covariance (mi.py:325-350) with element-wise df; mi_with/mi_fit/pool_linear_models (mi_workflow.py:40/95/165): pools a single slope from sklearn LinearRegression with OLS variance formula.
Limits: no Barnard-Rubin adjusted df (no dfcom), no D1/D2/D3, no GLM/logit/Cox/mixed pooling, no CIs/p-value table output, no statsmodels results pooling. Closed-form Rubin conformance tests only (tests/test_statistical_conformance.py 2, test_mi_pooling 23).
Rationale: correct Rubin's rules + covariance, but much narrower than mice/MIANALYZE.

## F8 MNAR sensitivity - score 1
No delta adjustment, tipping point, pattern-mixture, selection model, or J2R/CIR anywhere. Only `simulate_mnar` (simulation, F11) and report text recommending sensitivity analysis (report.py). Score 1 for advice only.

## F9 Time-series imputation - score 3
Impl: impute_ts (timeseries.py:345): ffill, bfill, linear, time, spline via pandas interpolate; limit; sentinel replacement; non-numeric ffill. Plus miss_ts_summary:126, gap_table:164, vis_ts_miss:197, vis_gap_lengths:254, vis_miss_over_time:314.
Limits: no Kalman/state-space, ARIMA, STL/seasonal, moving-average, seasonal split; no multi-gap diagnostics beyond gap table; `_require_sorted` raises on unsorted. Tests: test_timeseries 35, impute_ts 13, summary 12, visuals 14.

## F10 Advanced ML / matrix completion - score 2
Impl: impute_rf / impute_gb (impute.py:1997/2068) wrapping sklearn IterativeImputer with RF/GB; BayesianRidge MICE. No softImpute, SVD, missForest OOB, GAIN, VAE, diffusion, deep. Tests: model_strategies 5.

## F11 Missingness simulation - score 5
Impl (simulate.py): simulate_mcar:8 (per-column random fraction), simulate_mar:38 (threshold on predictor_col, upper/lower tail), simulate_mnar:79 (prob depends on target's own value vs median: frac*2 vs frac*0.25), simulate_mixed:112.
Limits: tail-threshold deterministic-ish designs, not logistic-model ampute; no pattern matrices, no mechanism calibration to hit exact proportion, no multivariate patterns weights; MNAR is a simple self-dependent probability. Tests: test_simulate 34.

## F12 Evaluation and benchmarking - score 6
Impl: compare_imputations (compare.py:57): mask-and-reconstruct on complete data, mask_frac, per-method metrics; cv_compare_imputations (compare.py:250): K-fold downstream-model score with fresh imputation per fold (no leakage); BenchmarkManifest (benchmark.py:75): schema-v2 privacy-preserving cross-tool evidence manifest with SHA-256s; r-oracle workflow.
Limits: no standardised benchmark datasets/leaderboard, no MCAR/MAR/MNAR sweep harness; plots via miss_impute_compare/vis_impute_dist. Tests: compare 35, cv_compare 11, benchmark 4.

## F13 Imputation diagnostics - score 5
Impl: mice_convergence R-hat (diagnostics.py:1118), plot_mice_convergence:1242, vis_impute_dist:1378 (observed vs imputed density), miss_impute_compare:1482. Limits: R-hat on imputed means only, no overimputation, no stripplot/xyplot-with-imputed, no plausibility/range checks except bounds. Tests: mice_convergence 25, visualise.

## F14 Reporting - score 6
Impl: create_report (report.py:403) with Jinja2 template (templates/report.html), self-contained HTML (base64 PNG), languages en/fa/de, RTL, MCAR interpretation; Limits: HTML only (no PDF/docx), no publication tables/LaTeX; PNG plots not interactive. Tests: test_report 51.

## F15 ML pipeline integration - score 6
Impl: MissinglyImputer (transformer.py): fit/transform, strategies mean/median/mode/knn/mice/rf/gb/pmm/hotdeck, get_params/set_params/clone-safe, train-only fit, strict schema check; FittedImputer. Limits: pandas DataFrame only (arrays/sparse rejected), no check_estimator, no set_output/metadata routing/missing-indicator, mixed KNN/logreg/polr/hotdeck seq/weighted raise NotImplementedError (transformer.py:123-145, 500-521). MICE in transformer is single imputation. Tests: test_transformer 33, fitted_imputer 18.

## F16 Data-frame integration - score 4
pandas-native with `.miss` accessor (accessor.py:92, ~45 methods). Polars: single `polars_missing_summary` (polars_adapter.py:44) rejecting pandas; no Arrow, no dask, no polars imputation. Tests: test_polars 3 (importorskip-style). Interchange protocol not used.

## F17 Scale and performance - score 1
`_warn_if_large` (impute.py:145) only warns; performance.py is three deprecated stubs (memory_usage_mb, optimize_dtypes, chunk_apply) raising warnings; no chunking/out-of-core/GPU/parallel; no published benchmarks (P3 roadmap in COMPETITOR_ANALYSIS.md).

## F18 Multilevel/survey/longitudinal - score 1
No multilevel imputation, no survey weights in MI/pooling (hotdeck_weighted is inverse-distance donor weights, not survey weights), no wide-long utilities (only time-series gap tools). Roadmap item only.

## F19 Internationalization - score 7
Report labels in en/fa/de (report.py:39-160), Persian/Arabic RTL shaping via arabic-reshaper + python-bidi optional extra with font fallback (visualisation/_base.py), README_FA.md/README_DE.md, tests rtl_base 51 + rtl_fallback 33. Unique among competitors; limited to 3 languages in reports and warnings/errors are English. Strongest differentiator.

## F20 Provenance and reproducibility - score 7
Impl: dataframe_fingerprint (provenance.py:139): SHA-256 over values/index/dtypes/categories without raw record persistence; analysis_provenance:221 config manifest; seed_sequence in ImputationPlan; random_state on all stochastic functions; deterministic bounded tests; BenchmarkManifest. Limits: no per-cell audit trail (which cells were imputed, with which method) beyond schema masks in ImputationResult; no replay. Tests: test_provenance 9, test_benchmark 4.

## N1 Maturity and API stability - score 3
Five months of git history (2026-04-21 to 2026-09), one tag (v1.0.0), 2.0.0 is a license bump, CHANGELOG [Unreleased] huge and not dated for 2.0.0; API "frozen" claim at 1.0.0 but many contract-breaking behavioural changes since (e.g. transformer refuses unsafe modes, Little's test rewritten). Single maintainer. Changelog's 0.1.0 date (2025-01-10) contradicts repo history.

## N5 Engineering quality - score 8
1035 test functions, documented 92.50% coverage (CI gate only 80), ruff in CI, py.typed, Python 3.9-3.14 CI matrix, R oracle CI workflow, docs deployment, explicit exception hierarchy (exceptions.py), validation helpers (_validation.py), contract/TDD docs (CONVENTIONS.md, TESTING.md), docstring examples. Minus: mypy only on one module; impute.py is a 2.4k-line module; some modules are thin shims.

## N7 Ease of use - score 8
`df.miss.*` accessor, one-line functions, naniar-like names, consistent `missing_values` sentinel param, clear exceptions, docs (Sphinx) + READMEs in 3 languages. Minus: several APIs return dicts/tuples varying by flags (impute_mice return type union); typed contracts partly opt-in.

## N8 License openness - score 4
BUSL-1.1 until 2030-01-01 (pyproject.toml, LICENSE; switched in 2026 per git log 78e6aa0). Source-available, not OSI open source; production-use restrictions per Additional Use Grant (see LICENSE - not scored in detail). XeroGraph_LICENSE file also present in repo root (unclear).

## N9 Ecosystem interoperability - score 5
Works with pandas (accessor), sklearn (transformer, Pipeline tested), statsmodels (polr/logreg), matplotlib/seaborn/plotly. Polars summary only; no Arrow/dask/spark; no R bridge (oracle tests only); no mlflow or other integrations; HTML report self-contained. Original README mentions data_quality_toolkit integration (removed from static rendering per changelog).

## Summary table
| dimension | score |
|---|---|
| F1 | 8 |
| F2 | 7 |
| F3 | 5 |
| F4 | 7 |
| F5 | 5 |
| F6 | 5 |
| F7 | 4 |
| F8 | 1 |
| F9 | 3 |
| F10 | 2 |
| F11 | 5 |
| F12 | 6 |
| F13 | 5 |
| F14 | 6 |
| F15 | 6 |
| F16 | 4 |
| F17 | 1 |
| F18 | 1 |
| F19 | 7 |
| F20 | 7 |
| N1 | 3 |
| N5 | 8 |
| N7 | 8 |
| N8 | 4 |
| N9 | 5 |
