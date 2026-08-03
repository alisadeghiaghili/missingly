# Missingly competitor analysis and parity roadmap

Last reviewed: 2026-08-03

## Executive conclusion

Missingly is not currently broader or statistically more mature than the combined R,
Python, SAS, SPSS, and Stata ecosystems. Its strongest position is an integrated,
authenticated, in-memory workflow with explicit guardrails, reproducibility fingerprints,
foreign-file ingestion, a Persian web UI, and one JSON API. Its largest gaps are
multiple-imputation breadth, complex-survey inference, generalized mixed models,
post-estimation, model diagnostics, time-varying survival, causal inference, Bayesian
analysis, SEM, psychometrics, time series, meta-analysis, power analysis, visualization,
and validated cross-tool numerical parity.

The product may claim a capability only after its estimand, assumptions, numerical
results, uncertainty, failure modes, serialization, API, documentation, and release build
pass the TDD gates in TESTING.md. “Better than R/Python/SAS/SPSS/Stata” is a target, not a
current product claim.

## Comparison method

Status meanings:

| Status | Meaning |
|---|---|
| Supported | Implemented, documented, exposed through the authenticated API, and tested |
| Partial | A useful subset exists, but major competitor behavior is absent |
| Missing | No production implementation |
| Unverified | Implemented locally but not yet benchmarked against an independent tool |

Competitive superiority requires all of the following:

1. Numerical agreement against at least two independent implementations or a published
   closed-form result.
2. Explicit and stable event, reference, contrast, tie, weight, and missing-value rules.
3. Better failure behavior: singular, separated, non-converged, unsupported, or
   non-identifiable fits must not silently produce an answer.
4. Reproducible settings, dependency versions, seeds, input fingerprints, and audit data.
5. Accessible API and UI with exportable tables, diagnostics, predictions, and reports.
6. Competitive performance and memory benchmarks on small, medium, and maximum-size data.

## Competitor set

### R

| Package or family | Reference capability | Missingly position |
|---|---|---|
| mice | Mixed-type chained equations, PMM, CART, RF, multilevel imputation, passive imputation, diagnostics, sensitivity analysis, pooling, prediction | Partial: posterior iterative numeric MI and Rubin-pooled OLS only |
| Amelia, mi, missForest | Joint-model and tree-based imputation alternatives | Missing except generic iterative and KNN single imputation |
| VIM, naniar | Missingness visualization and exploration | Partial: tables and associations, no mature visualization |
| survival | Cox, diagnostics, stratification, counting-process and multi-state workflows | Partial: Cox and exploratory residual-time diagnostics |
| lifelines equivalent in Python | Time-varying Cox, prediction and survival utilities | Missing time-varying Cox |
| lme4 and nlme | LMM, GLMM, nonlinear mixed models, crossed and nested effects, covariance structures | Partial: Gaussian random intercept and one random slope |
| glmmTMB | Broad GLMM families, zero inflation, dispersion models, offsets and random effects | Missing GLMM; non-mixed ZIP/ZINB exists |
| geepack | Mean, scale and correlation models, clustered categorical responses, flexible correlation | Partial: Gaussian/Binomial/Poisson, Independence/Exchangeable/AR1 |
| survey | Multistage designs, strata, PSU, FPC, replicate weights, calibration, survey GLM/Cox | Missing |
| ordinal, nnet, VGAM | Ordinal, multinomial and flexible categorical-response models | Partial: basic proportional-odds and multinomial logit |
| brms, rstanarm | Bayesian multilevel and distributional modeling | Missing |
| lavaan | SEM, CFA, latent variables and invariance | Missing |
| metafor | Meta-analysis, meta-regression and diagnostics | Missing |
| forecast, fable | Time-series models, decomposition and forecasting | Missing |

### Python

| Package | Reference capability | Missingly position |
|---|---|---|
| pandas, Polars | Rich data manipulation, joins, grouping, lazy/streaming execution | Missing as an end-user transformation API |
| SciPy | Broad hypothesis tests, distributions, optimization and resampling | Partial: five classical procedures |
| statsmodels | OLS/GLM/GEE/MixedLM, MICE, time series, robust covariance, diagnostics and post-estimation | Partial |
| scikit-learn | Pipelines, imputers, preprocessing, model selection, metrics and ML estimators | Partial imputation only |
| lifelines | Survival models, time-varying covariates, prediction and diagnostics | Partial |
| PyMC, Bambi | Bayesian modeling, posterior diagnostics and prediction | Missing |
| miceforest, autoimpute | Multiple-imputation workflows and alternative engines | Missing |
| missingno | Missingness visualization | Missing |
| Pingouin | Convenient statistical tests, effect sizes and power | Mostly missing |
| scikit-survival | Survival ML, competing risks and evaluation metrics | Missing |

### Commercial and GUI tools

| Tool | Reference capability | Missingly position |
|---|---|---|
| SAS/STAT | GENMOD/GEE, MIXED/GLIMMIX, MI/MIANALYZE, survey procedures, survival, extensive diagnostics | Partial |
| IBM SPSS Statistics | Missing Value Analysis, MI, MIXED/GENLIN/GEE, Complex Samples, GUI and output management | Partial |
| Stata | MI, multilevel and panel models, survival, complex surveys, causal inference, SEM, power and post-estimation | Partial |
| JMP and Minitab | Guided modeling, DOE, quality/statistical graphics and interactive diagnostics | Missing |
| JASP and jamovi | Accessible GUI, Bayesian and frequentist analyses, publication-ready output | Partial UI, missing breadth |
| GraphPad Prism | Biostatistical workflows and polished scientific graphing | Missing graphing breadth |
| MATLAB Statistics and Machine Learning Toolbox | Statistical modeling, ML, simulation and engineering integration | Missing breadth |

## Capability matrix

| Domain | Current Missingly support | Status | Required parity milestone |
|---|---|---|---|
| Import | CSV, JSON, XLSX, SAV, DTA, XPT, SAS7BDAT | Supported | Preserve labels, formats, dates and user-missing metadata |
| Data preparation | Validation and CSV export | Partial | Typed transformations, recodes, joins, reshape, labels, pipelines |
| Missingness description | Patterns, rates, numeric associations | Partial | Visual maps, upset plots, Little/Jamshidian-Jalal tests with warnings |
| Simple imputation | Mean, median, mode, constant | Supported | Add indicators and grouped rules |
| Single multivariate imputation | KNN and iterative numeric | Partial | Mixed data, bounds, PMM, CART, RF and diagnostics |
| Multiple imputation | Posterior iterative numeric | Partial | Mixed types, per-variable methods, predictor matrix, passive terms |
| MI inference | Rubin-pooled numeric OLS | Partial | GLM, Cox, mixed, GEE, predictions, joint tests, Barnard-Rubin DF |
| Classical tests | Pearson, Welch, chi-square | Partial | Nonparametric, paired/repeated, exact, resampling, multiplicity |
| Linear models | OLS, WLS, categorical terms and interactions | Partial | Robust/cluster covariance, diagnostics, contrasts, marginal means |
| Binary models | Logistic with inference | Partial | Probit/cloglog, marginal effects, calibration, ROC, penalization |
| Count models | Poisson, NB2, ZIP, ZINB2, hurdle Poisson | Partial | Hurdle NB, truncated, mixed, GEE, diagnostics and prediction |
| Ordinal/multinomial | Basic ordinal and multinomial logit | Partial | Categorical predictors everywhere, assumption tests, partial PO |
| Mixed models | Gaussian random intercept and one slope | Partial | Nested/crossed effects, multiple slopes, GLMM, covariance structures |
| GEE | Gaussian/Binomial/Poisson; Independence/Exchangeable/AR1; robust/bias-reduced SE | Partial | Unstructured, nested, fixed correlation, ordinal/multinomial, weights |
| Survival | KM, log-rank and Cox | Partial | Formal PH tests, plots, time-varying Cox, competing risks, AFT, frailty |
| Complex surveys | Analytic-weighted OLS only | Missing | PSU, strata, FPC, replicate weights, calibration, design effects |
| Causal inference | None | Missing | Estimand-first IPW/AIPW/matching/DID/IV with diagnostics |
| Bayesian modeling | None | Missing | Posterior inference, diagnostics, predictive checks and priors |
| SEM and latent variables | None | Missing | CFA, SEM, mediation, invariance and latent classes |
| Psychometrics | None | Missing | Reliability, EFA/CFA and IRT |
| Time series | None | Missing | ARIMA/ETS/state space, diagnostics, backtesting and intervals |
| Meta-analysis | None | Missing | Fixed/random effects, heterogeneity, forest/funnel, meta-regression |
| Power and sample size | None | Missing | Tests, regression, survival, cluster and precision-based designs |
| Diagnostics | Model-specific minimums | Partial | Residuals, influence, calibration, GOF, assumptions, visual diagnostics |
| Prediction | Limited model output | Missing | Predictions, intervals, marginal effects, validation and export |
| Reporting | Descriptive self-contained HTML | Partial | Model tables, plots, assumptions, provenance and publication formats |
| Reproducibility | Input/settings SHA-256 and engine label | Supported | Dependency lock, schema version, environment manifest, replay bundle |
| Extensibility | Internal Python functions | Missing | Stable plugin/estimator protocol and third-party method registration |
| Scale | Request caps and in-memory execution | Partial | Streaming, Arrow/Polars, async jobs, cancellation and benchmarks |
| Accessibility | Persian web UI and JSON API | Partial | Guided analysis, effect explanations, accessible plots and localization |

## What Missingly can plausibly do better

The realistic path to superiority is not to reimplement every algorithm immediately. It is
to combine trustworthy methods with a stronger integrated contract:

- One secure API and guided UI across file formats and model families.
- No silent coercion, hidden reference category, convergence failure, or unsupported claim.
- First-class provenance and replay metadata for every result.
- Machine-readable results and human-readable reports from the same validated object.
- Persian-first accessibility while keeping English statistical terminology explicit.
- A cross-tool conformance suite that catches numerical drift before release.
- Clear separation between analytic weights, frequency weights and survey-design weights.
- Explicit estimands: conditional versus marginal, subject-specific versus
  population-averaged, causal versus associational.

These are current design advantages or achievable differentiators. They are not evidence
that the statistical breadth already exceeds competitors.

## TDD roadmap

### P0 — credibility and benchmark infrastructure

1. Add frozen benchmark datasets with licenses and checksums.
2. Store expected R, statsmodels/lifelines, Stata, SAS and SPSS outputs with tool versions.
3. Define absolute and relative tolerances per statistic.
4. Test coefficient estimates, covariance, confidence intervals, likelihoods, coding,
   missing-row counts, warnings and convergence.
5. Add schema snapshots and backward-compatibility tests for every API response.
6. Enable Node/UI tests in release CI; local skips cannot count as passed coverage.
7. Replace deprecated FastAPI startup hooks and naive UTC calls.

Exit criterion: every currently advertised method has an independent numerical oracle.

### P1 — missing-data leadership

1. Mixed-type FCS with PMM, logistic, ordinal and multinomial conditional models.
2. Predictor matrices, blocks, visit order, bounds, passive imputation and per-column rules.
3. Convergence traces, distribution overlays, overimputation and sensitivity diagnostics.
4. MNAR delta adjustment and documented sensitivity analysis.
5. Rubin/Barnard-Rubin pooling for GLM, Cox, mixed, GEE, contrasts and predictions.
6. Import/export and consistency validation for multiply imputed datasets.

Exit criterion: benchmark parity with mice and Stata MI on public fixtures, plus simpler
reproducible API and clearer failure messages.

### P2 — correlated and hierarchical outcomes

1. Multiple, nested and crossed random effects.
2. Gaussian residual covariance structures and ML/REML selection.
3. Binomial, Poisson, NB2 and ordinal GLMM.
4. GEE unstructured, nested and fixed correlation; weights and categorical responses.
5. Small-cluster corrections with explicit method metadata.
6. Estimated marginal means, contrasts, predictions and residual diagnostics.

Exit criterion: parity fixtures against lme4, glmmTMB, geepack, statsmodels and Stata.

### P3 — survival analysis

1. Formal scaled-Schoenfeld per-term and global PH tests.
2. Diagnostic plots, influence, martingale/deviance residuals and baseline survival.
3. Counting-process start/stop and time-varying covariates.
4. Competing risks, recurrent events, frailty and parametric AFT families.
5. Prediction, calibration, time-dependent metrics and uncertainty.

Exit criterion: parity against R survival and Python lifelines on documented fixtures.

### P4 — complex surveys

1. Typed survey-design object with weight, PSU, strata, nest and FPC metadata.
2. Taylor-linearized descriptive estimates and GLMs.
3. Replicate-weight methods: bootstrap, jackknife, BRR and SDR where applicable.
4. Calibration, raking, poststratification, lonely-PSU policy and design effects.
5. Survey-weighted survival and integration with multiple imputation.

Exit criterion: parity against R survey, Stata svy and SAS survey procedures.

### P5 — breadth expected from full statistical platforms

1. Causal inference with overlap, balance and sensitivity diagnostics.
2. Bayesian models and posterior predictive checking.
3. SEM/CFA, psychometrics and IRT.
4. Time series and forecasting with honest backtesting.
5. Meta-analysis and meta-regression.
6. Power, precision and sample-size calculations.
7. DOE, quality control and publication-grade statistical graphics.

## Cross-tool benchmark manifest

Every benchmark case must contain:

| Field | Requirement |
|---|---|
| Dataset | Frozen local file, license, provenance URL and SHA-256 |
| Model specification | Formula/design, coding, links, offsets, weights and missing policy |
| Tool metadata | Product/package name, exact version and platform |
| Expected output | Estimates, covariance/SE, CI, tests, fit statistics and row counts |
| Tolerance | Justified absolute and relative thresholds |
| Known differences | Tie method, likelihood constants, DF correction, contrast or optimizer |
| Missingly contract | Endpoint, request fixture and stable response-schema version |

## Source registry

Primary or official documentation used for this analysis:

- mice: https://amices.org/mice/reference/mice.html
- lme4: https://lme4.github.io/lme4/reference/lme4-package.html
- glmmTMB: https://glmmtmb.github.io/glmmTMB/index.html
- geepack: https://CRAN.R-project.org/package=geepack
- R survey: https://r-survey.r-forge.r-project.org/survey/
- scikit-learn IterativeImputer: https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html
- statsmodels API: https://www.statsmodels.org/stable/api.html
- statsmodels MICE: https://www.statsmodels.org/stable/imputation.html
- lifelines: https://lifelines.readthedocs.io/en/latest/
- lifelines time-varying Cox: https://lifelines.readthedocs.io/en/latest/Time%20varying%20survival%20regression.html
- SAS GENMOD: https://documentation.sas.com/api/collections/pgmsascdc/v_058/docsets/statug/content/genmod.pdf
- IBM SPSS Missing Value Analysis: https://www.ibm.com/docs/en/spss-statistics/31.0.0?topic=values-missing-value-analysis
- IBM SPSS MIXED: https://www.ibm.com/docs/en/spss-statistics/32.0.0?topic=mixed-overview-command
- Stata feature index: https://www.stata.com/features/
- Stata multiple imputation: https://www.stata.com/features/multiple-imputation/
- Stata complex surveys: https://www.stata.com/features/overview/survey-features/
- Stata multilevel models: https://www.stata.com/features/overview/multilevel-mixed-effects-models/

## Review policy

Review this file at least once per release. A competitor capability can be marked
Supported only when implementation, docs and tests exist on the release branch. A claim
of “better” additionally requires benchmark evidence for correctness, usability,
performance or safety. Marketing copy must link to that evidence and must not infer global
superiority from one successful model or dataset.
