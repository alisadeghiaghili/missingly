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
- The HTML report is self-contained and escapes all user-provided titles and values. It includes dataset profile, missing-data notes, correlations, and the reproducibility fingerprint; it is a descriptive report, not a claim of causal or mechanism-specific inference.
- Multiple imputation, KNN, random forest imputation, and mechanism-specific modeling are deliberately not presented as supported until they have validation benchmarks.

## Supported now

- Dataset profile: schema, missingness, numeric summaries, categorical frequencies, duplicate rows, and Pearson correlations.
- Missingness report: missing-value patterns and point-biserial associations between a missingness indicator and observed numeric variables. It explicitly avoids treating low missingness as proof of MCAR.
- Statistical tests: Pearson correlation, two-sided Welch two-sample t-test, Pearson chi-square independence test, and OLS regression with coefficient confidence intervals.
- Models: binary logistic regression (maximum likelihood) with event/reference coding, log-odds, odds ratios, confidence intervals, AIC/BIC, McFadden pseudo-R², and convergence/separation rejection.
- Reproducibility: each result has the engine version and SHA-256 fingerprint of its input plus analysis settings.

## Statistical guardrails

- Numeric methods reject columns with mixed observed numeric/non-numeric values rather than silently coercing them.
- Pearson requires at least three non-constant paired values; Welch requires two observations per group; OLS rejects perfect collinearity and insufficient degrees of freedom.
- Chi-square reports a warning when an expected cell count is below five.
- This phase does not claim support for Little's MCAR test, multiple imputation, survey weights, mixed models, survival analysis, or causal inference. Those require separate validated implementations.

The contracts in `tests/test_analytics.py` are the initial numerical benchmark suite. Any new method must add known-result tests and document assumptions before it is exposed in the API.
