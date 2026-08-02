# Analytics foundation

Missingly Analytics is an authenticated, in-memory analysis service: raw input records are not saved to the application database. Audit events retain only operation type and row count.

Each request is capped at 10,000 rows, 100 columns, 250,000 supplied cells, and the configured `MAX_ANALYSIS_BODY_BYTES` (5 MB by default).

## Ingestion and preprocessing

- The authenticated import endpoint accepts `.csv`, `.json` (array of records), `.xlsx`, `.sav` (SPSS), `.dta` (Stata), `.xpt`, and `.sas7bdat` (SAS). The upload has a separate `MAX_ANALYSIS_UPLOAD_BYTES` cap and is parsed in memory.
- Simple imputation supports `mean`, `median`, `mode`, or an explicit `constant`; each response reports the exact filled columns, values, and cell counts. It does not silently choose an algorithm.
- Advanced numeric single imputation supports `knn` and deterministic `iterative` regression. Both require at least two numeric, non-all-missing selected columns and return the seed/settings in the fingerprint. They are not Rubin-pooled multiple imputation.
- The authenticated CSV export endpoint returns the validated current records without retaining the dataset server-side.
- Multiple imputation, KNN, random forest imputation, and mechanism-specific modeling are deliberately not presented as supported until they have validation benchmarks.

## Supported now

- Dataset profile: schema, missingness, numeric summaries, categorical frequencies, duplicate rows, and Pearson correlations.
- Missingness report: missing-value patterns and point-biserial associations between a missingness indicator and observed numeric variables. It explicitly avoids treating low missingness as proof of MCAR.
- Statistical tests: Pearson correlation, two-sided Welch two-sample t-test, Pearson chi-square independence test, and OLS regression with coefficient confidence intervals.
- Reproducibility: each result has the engine version and SHA-256 fingerprint of its input plus analysis settings.

## Statistical guardrails

- Numeric methods reject columns with mixed observed numeric/non-numeric values rather than silently coercing them.
- Pearson requires at least three non-constant paired values; Welch requires two observations per group; OLS rejects perfect collinearity and insufficient degrees of freedom.
- Chi-square reports a warning when an expected cell count is below five.
- This phase does not claim support for Little's MCAR test, multiple imputation, survey weights, mixed models, survival analysis, or causal inference. Those require separate validated implementations.

The contracts in `tests/test_analytics.py` are the initial numerical benchmark suite. Any new method must add known-result tests and document assumptions before it is exposed in the API.
