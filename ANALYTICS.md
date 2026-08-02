# Analytics foundation

Missingly Analytics is an authenticated, in-memory analysis service: raw input records are not saved to the application database. Audit events retain only operation type and row count.

Each request is capped at 10,000 rows, 100 columns, 250,000 supplied cells, and the configured `MAX_ANALYSIS_BODY_BYTES` (5 MB by default).

## Ingestion and preprocessing

- The authenticated import endpoint accepts `.csv`, `.json` (array of records), `.xlsx`, `.sav` (SPSS), `.dta` (Stata), `.xpt`, and `.sas7bdat` (SAS). The upload has a separate `MAX_ANALYSIS_UPLOAD_BYTES` cap and is parsed in memory.
- Simple imputation supports `mean`, `median`, `mode`, or an explicit `constant`; each response reports the exact filled columns, values, and cell counts. It does not silently choose an algorithm.
- Advanced numeric single imputation supports `knn` and deterministic `iterative` regression. Both require at least two numeric, non-all-missing selected columns and return the seed/settings in the fingerprint. They are not Rubin-pooled multiple imputation.
- The authenticated CSV export endpoint returns the validated current records without retaining the dataset server-side.

## Multiple imputation with inference

`/analysis/multiple-imputation/ols` runs posterior-draw iterative imputation for numeric columns and fits OLS to every completed dataset. It pools coefficient estimates, within/between/total variance, standard errors, degrees of freedom, confidence intervals, p-values, and fraction of missing information using Rubin's rules. The method is limited to numeric OLS and does not establish MAR/MCAR assumptions for the user; those remain a study-design judgment.

## Survival and reporting

- Kaplan–Meier accepts non-negative numeric time and a two-category event column. It reports at-risk counts, events, Greenwood log-log confidence intervals, and a log-rank test only for exactly two groups.
- `/analysis/cox` fits a Cox proportional-hazards model for strictly positive follow-up time, a two-category event column, and numeric predictors. It returns log hazard ratios, hazard ratios with confidence intervals, a partial-likelihood ratio test, Efron or Breslow tie handling, optional strata, or optional cluster-robust standard errors (one at a time). It does not test the proportional-hazards assumption or support time-varying coefficients.
- The HTML report is self-contained and escapes all user-provided titles and values. It includes dataset profile, missing-data notes and patterns, correlations, interpretation boundaries, and the reproducibility fingerprint; it is a descriptive report, not a claim of causal or mechanism-specific inference.
- Random-forest imputation and mechanism-specific missing-data modeling are deliberately not presented as supported until they have validation benchmarks.

## Clustered and weighted continuous outcomes

- `/analysis/mixed-linear` fits a Gaussian linear mixed-effects model with one random intercept grouping column and fixed numeric predictors, using REML. It reports fixed-effect z inference, variance components, and the intraclass correlation. Random slopes, crossed effects, generalized mixed models, and repeated-measures covariance structures are not yet supported.
- `/analysis/weighted-ols` fits weighted least squares with strictly positive numeric analytic weights. It reports coefficient t inference and weighted R-squared. It is not a complex-survey estimator: strata, primary sampling units, finite-population corrections, calibration, and design-based standard errors are intentionally out of scope.

## Count outcomes

- `/analysis/count-regression` fits Poisson or NB2 negative-binomial regression for non-negative integer outcomes and numeric predictors. A strictly positive exposure column becomes a log offset, so the model estimates rates when an exposure is supplied.
- It returns log rate ratios, rate ratios with confidence intervals, AIC, Pearson chi-square, and an overdispersion ratio. Poisson responses flag substantial overdispersion so the user can compare NB2 instead of silently treating counts as continuous.
- Zero-inflated, hurdle, multinomial, repeated-count, and complex-survey count models are not yet supported; the output explicitly does not diagnose zero inflation or dependence.

## Ordinal outcomes

- `/analysis/ordinal-logistic` fits a proportional-odds ordinal logistic model for an outcome with three to twenty ordered categories and numeric predictors. The caller must supply every category in its intended order, preventing accidental alphabetical or numeric ordering.
- It reports common cumulative odds ratios, cutpoints, likelihood, AIC/BIC, and confidence intervals. The proportional-odds assumption is not automatically tested, and multinomial outcomes without a defensible order are intentionally excluded.

## Unordered multi-class outcomes

- `/analysis/multinomial-logistic` fits baseline-category multinomial logistic regression for three to twenty unordered outcome categories and numeric predictors. The caller must choose an observed reference category explicitly.
- It reports a coefficient set and relative-risk ratio for every non-reference category, plus likelihood-ratio inference, McFadden pseudo-R², AIC, and BIC. It rejects non-convergence and non-finite estimates rather than returning a misleading fit.

## Supported now

- Dataset profile: schema, missingness, numeric summaries, categorical frequencies, duplicate rows, and Pearson correlations.
- Missingness report: missing-value patterns and point-biserial associations between a missingness indicator and observed numeric variables. It explicitly avoids treating low missingness as proof of MCAR.
- Statistical tests: Pearson correlation, two-sided Welch two-sample t-test, Pearson chi-square independence test, and OLS regression with coefficient confidence intervals.
- Models: binary logistic regression (maximum likelihood) with event/reference coding, log-odds, odds ratios, confidence intervals, AIC/BIC, McFadden pseudo-R², and convergence/separation rejection.
- Models: Gaussian random-intercept linear mixed-effects models and analytic-weighted OLS for continuous outcomes.
- Models: Cox proportional-hazards regression with Efron/Breslow ties, optional strata, and optional cluster-robust standard errors.
- Models: Poisson and NB2 negative-binomial count regression with an optional log-exposure offset.
- Models: ordinal logistic (proportional-odds) regression with an explicit outcome-category order.
- Models: baseline-category multinomial logistic regression with an explicit reference category.
- Reproducibility: each result has the engine version and SHA-256 fingerprint of its input plus analysis settings.

## Statistical guardrails

- Numeric methods reject columns with mixed observed numeric/non-numeric values rather than silently coercing them.
- Pearson requires at least three non-constant paired values; Welch requires two observations per group; OLS rejects perfect collinearity and insufficient degrees of freedom.
- Chi-square reports a warning when an expected cell count is below five.
- This phase does not claim support for Little's MCAR test, full complex-survey estimation, zero-inflated or hurdle models, random-slope or generalized mixed models, proportional-hazards diagnostics, time-varying Cox effects, or causal inference. Those require separate validated implementations.

The contracts in `tests/test_analytics.py` are the initial numerical benchmark suite. Any new method must add known-result tests and document assumptions before it is exposed in the API.
