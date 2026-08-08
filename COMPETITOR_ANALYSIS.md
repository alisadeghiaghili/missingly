# Missing-data competitor analysis and parity roadmap

Last reviewed: 2026-08-08

## Executive conclusion

Missingly is already an unusually broad Python-native missing-data toolkit: one package
combines pandas summaries, a `.miss` accessor, sentinel handling, static and Plotly
visualization, sklearn-compatible imputation, mixed-type Gower KNN, hot-deck methods,
simulation, diagnostics, multiple-imputation helpers, comparison, RTL HTML reports, and
deterministic provenance. No evidence yet supports the global claim that it is better
than the combined R, Python, SAS, SPSS, and Stata ecosystems.

The largest credibility and capability gaps are independent numerical benchmarks,
mixed-type fully conditional specification (FCS), predictor matrices and passive terms,
PMM inside the MI engine, Barnard–Rubin small-sample inference, broad pooled model
support, formal MNAR sensitivity workflows, and performance evidence at scale. “Best” is
the product target; superiority must be demonstrated per capability.

## Status and evidence rules

| Status | Meaning |
|---|---|
| Verified locally | Implementation, public docs, and deterministic tests exist in this repository |
| Partial | A useful subset exists, but important reference behavior is absent |
| Gap | No production implementation exists |
| Parity verified | Independent cross-tool fixture agrees within a justified tolerance |
| Better verified | Parity plus measured usability, safety, or performance advantage |

No row may be labeled `Parity verified` from API resemblance alone. `Better verified`
requires a reproducible benchmark and an explicit metric.

## Competitor matrix

| Ecosystem/tool | Reference strengths | Missingly position | Most important gap |
|---|---|---|---|
| R `naniar` | Tidy shadow matrices, summaries, special missing values, broad missingness graphics | Verified locally for analogous summaries, shadow features, sentinels, UpSet/static/interactive plots; no independent UX benchmark | Cross-tool visual/summary fixtures and tidy-long workflow |
| R `VIM` | Visualization plus kNN, hot-deck, and mixed-data workflows | Partial: comparable families exist, including internal Gower and hot-deck variants | Published algorithm/parity fixtures and survey-weighted workflows |
| R `mice` | Mixed continuous/binary/ordered/unordered FCS, PMM, per-variable methods, predictor matrices, blocks, visit order, passive and multilevel imputation, diagnostics and pooling | Partial: multiple numeric imputations, standalone categorical imputers, convergence helpers, scalar/linear Rubin pooling | Unified mixed FCS, predictor matrix, PMM, passive/multilevel terms, Barnard–Rubin pooling |
| R `Amelia` | Joint-model/bootstrap multiple imputation with time-series/cross-section priors | Gap for equivalent joint MI | Joint-model engine and comparative diagnostics |
| R `missForest` / `missRanger` | Iterative tree imputation for mixed data, OOB diagnostics, scalable variants | Partial: RF/GB and mixed KNN exist, but not equivalent iterative forest MI | OOB error, iterative mixed forest engine, benchmarked scale |
| Python `missingno` | Focused matrix, bar, heatmap, and dendrogram exploration | Verified locally for these plot families plus a broader report; parity not independently tested | Frozen visual semantics and performance fixtures |
| scikit-learn imputers | Stable estimator/pipeline contracts, Simple/KNN/Iterative imputation, missing indicators | Partial: sklearn-compatible transformer and more domain features, but estimator compliance is not yet exhaustively checked | `check_estimator`, feature-name, sparse, metadata-routing, and all dtype contracts |
| statsmodels MICE | PMM-based chained equations and pooled analysis over statsmodels models | Partial: wider exploration/report surface, narrower MI analysis and PMM support | PMM and general model/result pooling |
| `miceforest`, `autoimpute`, `fancyimpute` | Alternative multiple/iterative engines, tree methods, matrix completion, tuning | Partial across method families | Independent quality/speed benchmark and advanced engines |
| SAS `PROC MI` / `MIANALYZE` | Monotone/FCS/MCMC imputation, constraints, diagnostics, broad pooled inference | Partial | MI method breadth, covariance-aware pooled tests, enterprise metadata |
| IBM SPSS Missing Values / MI | GUI-guided patterns, Little's test, automatic/monotone/FCS, PMM, constraints, pooled supported procedures | Partial: stronger open Python composability; weaker guided workflow and pooled breadth | Guided UI, auto method selection, pooled model coverage |
| Stata `mi` | Multiple storage styles, chained/monotone/MVN imputation, `mi estimate`, post-estimation, survey integration | Partial | MI data object/serialization, post-estimation, survey and model breadth |
| JASP / jamovi | Accessible GUI, reproducible analysis files, polished tables | Gap for a desktop GUI | Guided no-code workflow and publication-ready tables |

## Defensible current differentiators

These are architectural differentiators, not global superiority claims:

- A single Python API spans exploration, transformation, imputation, simulation,
  comparison, visualization, inference helpers, and self-contained multilingual reports.
- pandas accessor and sklearn transformer surfaces coexist without requiring a GUI or
  proprietary file format.
- Persian/Arabic RTL rendering and explicit sentinel handling are first-class concerns.
- Native Gower distance removes a fragile optional runtime dependency.
- Provenance fingerprints bind values, index, dtypes, category metadata, operation, and
  settings without persisting raw records.
- Domain-specific exceptions and TDD guardrails prefer explicit failure over plausible
  but invalid output.

## Phased parity roadmap

### P0 — credibility and reproducibility

1. Freeze licensed benchmark datasets with SHA-256 and generator scripts.
2. Capture official outputs from at least two independent tools for every advertised
   inferential method.
3. Add tolerances for estimates, variance, degrees of freedom, missing-row policy,
   warnings, and convergence.
4. Enforce coverage, docstring, typing, build, and optional-integration gates in CI.
5. Publish benchmark hardware, versions, memory, and runtime rather than unqualified
   “fastest” claims.

Exit criterion: every statistical claim has an independent numerical oracle. Expected
competitive improvement: trustworthiness becomes auditable; breadth does not change.

### P1 — mixed-type FCS leadership

1. One MI engine with PMM, Gaussian, logistic, ordinal, multinomial, CART, and RF methods.
2. Predictor matrices, blocks, visit order, `where` masks, bounds, and per-column rules.
3. Passive transformations and two-level imputation.
4. Convergence traces, overimputation, distribution overlays, and chain diagnostics.

Exit criterion: parity against R `mice`, statsmodels MICE, SPSS FCS, and Stata chained
imputation on public fixtures. Expected improvement: closes the largest algorithmic gap.

### P2 — MI inference and sensitivity

1. Barnard–Rubin degrees of freedom and joint Wald/D1/D2/D3-style tests.
2. Pool GLM, ordinal/multinomial, mixed, GEE, survival, contrasts, marginal effects, and
   predictions through a typed result protocol.
3. MNAR delta adjustment, tipping-point analysis, and pattern-mixture sensitivity.
4. Long/wide MI import-export with consistency validation and replay manifests.

Exit criterion: parity against `mice::pool`, SAS MIANALYZE, SPSS pooled procedures, and
Stata `mi estimate`. Expected improvement: moves from imputation utility to valid analysis
workflow.

### P3 — scale, usability, and ecosystem

1. Build a capability-based execution layer: pandas reference semantics, Arrow/DataFrame
   Interchange boundaries, optional Polars LazyFrame streaming, and a chunked fallback.
2. Classify every operation as exact-streaming, bounded-memory batch, approximate, or
   materialising; enforce memory budgets before allocation and expose copy semantics.
3. Benchmark 1M rows on pull requests and 10M-row narrow/wide Parquet workloads on
   scheduled/release jobs. Record wall time, peak RSS, extra-memory ratio, output hash,
   backend/version, and hardware alongside numerical-parity checks.
4. Keep exact KNN, dense correlations, MICE, and forest methods honest: implement a
   documented scalable variant or reject over-budget execution before materialisation.
5. Complete sklearn estimator compliance and container interoperability without making
   Polars, Arrow, or any distributed scheduler a mandatory core dependency.
6. Add publication-grade tables, notebook widgets, an optional guided UI outside core,
   and a stable plugin protocol for specialist engines.

Exit criterion: measured advantage on at least one published safety, usability, memory,
or speed metric while retaining numerical parity. Large-data readiness additionally
requires deterministic failure/cancellation tests and bounded-memory 10M-row evidence.
Only then may that specific advantage be described as “better.”

## Cross-tool benchmark manifest

Every fixture must record dataset license and SHA-256; exact tool/package version and
platform; method and estimand; category/reference coding; seed; missing-value policy;
expected estimates, uncertainty and diagnostics; absolute/relative tolerances; known
algorithmic differences; and the Missingly API/version used to reproduce it.

## Official source registry

- R `mice` FCS and configuration: https://amices.org/mice/reference/mice.html
- R `mice` pooling: https://amices.org/mice/reference/pool.html
- R `naniar` overview: https://naniar.njtierney.com/
- R `naniar` visual gallery: https://naniar.njtierney.com/articles/naniar-visualisation.html
- R `VIM`: https://cran.r-project.org/package=VIM
- R `Amelia`: https://cran.r-project.org/package=Amelia
- R `missForest`: https://cran.r-project.org/package=missForest
- scikit-learn imputation API: https://scikit-learn.org/stable/api/sklearn.impute.html
- statsmodels MICE: https://www.statsmodels.org/stable/imputation.html
- SAS missing-data procedures: https://support.sas.com/rnd/app/stat/procedures/multipleimputation.html
- IBM SPSS Multiple Imputation: https://www.ibm.com/docs/en/spss-statistics/31.0.0?topic=imputation-impute-missing-data-values-multiple
- Stata multiple imputation: https://www.stata.com/features/multiple-imputation/
- jamovi missing-values module library: https://www.jamovi.org/library.html
- JASP: https://jasp-stats.org/

## Review policy

Review this file for every release that changes an analytical claim. A status may advance
only in the same PR as its implementation, public documentation, benchmark evidence, and
tests. Marketing text must link to the specific evidence and must not infer global
superiority from one dataset or method.
