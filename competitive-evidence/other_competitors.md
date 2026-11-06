# Other competitors: missing-data scorecard (assessed 2026-11-06)

Caveat: web search gave little release-note detail for closed products. Versions marked "unv." were not confirmed from a primary source. Scores are conservative; many reflect product knowledge plus docs, not fresh page reads. N5 for closed products is a judgement from QA/validation documentation (uncertain).

## 1. Metadata
| Tool | Category | Version / date | License / price | Maintenance |
|---|---|---|---|---|
| SAS/STAT (PROC MI, MIANALYZE, SURVEYIMPUTE) | Commercial stats | SAS/STAT 15.x / Viya; 2026 build unv. | Commercial, per-seat/site (high) | Active, vendor |
| IBM SPSS Statistics Missing Values + MI | Commercial stats | v31 docs; v32.0.0 docs also indexed; date unv. | Commercial subscription; MV is an add-on module | Active, vendor |
| Stata mi suite | Commercial stats | Stata 19 (StataNow); MI manual 2025; no MI changes in "What's new" | Commercial perpetual/annual | Active, vendor |
| JMP Pro | Commercial stats/GUI | JMP 19.1 help dated 2026-01-13 (19.2 draft help exists) | Commercial (Pro tier) | Active, vendor |
| MATLAB + Stats & ML Toolbox | Commercial numeric | R2026b assumed; unv. | Commercial | Active, 2 releases/yr |
| Mplus | Commercial SEM | 8.x (8.11 unv.) | Commercial | Slow, vendor |
| Blimp | Free closed-source Bayesian MI | v3 (guide 2023); later unv. | Free, closed source | Moderate, IES-funded |
| JASP | GUI open | 0.19/0.20 unv. | AGPL, free | Active |
| jamovi | GUI open | 2.7 unv. | AGPL, free | Active |
| Impute.jl (Julia) | Julia package | v0.7.1, 2026-10-04 | MIT | Active |
| Mice.jl (Julia) | Julia package | v0.4.1, 2026-05-29 | MIT (unv.) | Active, single maintainer |
| danfo.js (JS/TS) | JS dataframe | unv. | MIT | Low (unv.) |
| polars fill_null (Rust/Python) | Dataframe baseline | version unv. | MIT | Very active |
| Apache Spark MLlib Imputer | Data platform | PySpark 4.3.0, 2026-10-30 | Apache-2.0 | Very active |
| H2O-3 | ML platform | 3.46.0.12, 2026-08-12 | Apache-2.0 (paid enterprise support) | Active |
| KNIME Analytics Platform | GUI workflow | 5.11 latest per NodePit (unv.) | Open core GPLv3 + paid Business Hub | Active |
| Orange Data Mining | GUI workflow | 3.40.0, 2025-12-20 | GPL-3.0 | Active |
| Altair RapidMiner (AI Studio) | GUI/ML commercial | 2026 version unv. | Commercial (limited free edition) | Active, vendor |
| DataRobot | AutoML platform | SaaS, continuous | Commercial, enterprise (high) | Active, vendor |
| Great Expectations | Data quality | 1.24.1 (PyPI; date unv.) | Apache-2.0 core; paid Cloud | Very active |
| Soda | Data quality | soda-core 4.30.0 (PyPI; date unv.) | Apache-2.0 core; paid Cloud | Active |

## 2. Scores
Columns: F1..F20 then N1..N10.

| Tool | F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | F11 | F12 | F13 | F14 | F15 | F16 | F17 | F18 | F19 | F20 | N1 | N2 | N3 | N4 | N5 | N6 | N7 | N8 | N9 | N10 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SAS PROC MI/MIANALYZE | 3 | 6 | 3 | 4 | 6 | 9 | 9 | 9 | 5 | 2 | 2 | 3 | 6 | 6 | 2 | 3 | 6 | 6 | 5 | 6 | 10 | 9 | 9 | 9 | 7 | 7 | 5 | 1 | 5 | 10 |
| IBM SPSS MVA + MI | 4 | 7 | 8 | 5 | 3 | 6 | 7 | 1 | 3 | 1 | 0 | 1 | 4 | 5 | 1 | 2 | 3 | 4 | 8 | 4 | 10 | 9 | 7 | 6 | 5 | 7 | 9 | 1 | 6 | 9 |
| Stata mi | 3 | 7 | 3 | 3 | 6 | 9 | 9 | 4 | 4 | 1 | 1 | 2 | 5 | 4 | 1 | 3 | 5 | 8 | 3 | 6 | 10 | 9 | 10 | 9 | 6 | 8 | 7 | 1 | 5 | 9 |
| JMP Pro | 5 | 6 | 3 | 5 | 3 | 2 | 1 | 0 | 3 | 6 | 0 | 2 | 3 | 5 | 1 | 3 | 4 | 1 | 7 | 4 | 9 | 7 | 8 | 5 | 5 | 8 | 9 | 1 | 5 | 8 |
| MATLAB | 2 | 3 | 0 | 6 | 2 | 0 | 0 | 0 | 6 | 3 | 0 | 1 | 1 | 3 | 2 | 3 | 6 | 1 | 5 | 4 | 10 | 9 | 9 | 5 | 6 | 9 | 8 | 1 | 6 | 9 |
| Mplus | 0 | 4 | 3 | 3 | 4 | 6 | 7 | 3 | 2 | 0 | 4 | 1 | 4 | 1 | 0 | 1 | 4 | 8 | 1 | 5 | 9 | 7 | 7 | 8 | 5 | 6 | 5 | 1 | 4 | 7 |
| Blimp | 1 | 2 | 1 | 2 | 7 | 8 | 7 | 3 | 2 | 1 | 0 | 1 | 6 | 2 | 0 | 2 | 4 | 9 | 1 | 5 | 6 | 5 | 6 | 8 | 4 | 6 | 6 | 5 | 4 | 6 |
| JASP | 1 | 1 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 0 | 2 | 1 | 0 | 8 | 3 | 7 | 7 | 6 | 3 | 6 | 8 | 10 | 8 | 5 | 5 |
| jamovi | 1 | 1 | 2 | 2 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 0 | 2 | 1 | 0 | 7 | 3 | 7 | 7 | 6 | 3 | 5 | 8 | 10 | 8 | 5 | 5 |
| Impute.jl | 0 | 1 | 0 | 5 | 1 | 0 | 0 | 0 | 4 | 3 | 0 | 1 | 0 | 0 | 2 | 6 | 5 | 1 | 0 | 3 | 5 | 3 | 4 | 1 | 6 | 7 | 7 | 10 | 6 | 4 |
| Mice.jl | 0 | 1 | 0 | 2 | 3 | 5 | 5 | 0 | 0 | 0 | 0 | 0 | 2 | 1 | 1 | 6 | 6 | 1 | 0 | 3 | 3 | 2 | 5 | 4 | 5 | 6 | 6 | 9 | 6 | 3 |
| danfo.js | 0 | 2 | 0 | 2 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 3 | 0 | 0 | 2 | 5 | 4 | 5 | 1 | 4 | 3 | 8 | 10 | 5 | 3 |
| polars fill_null | 0 | 3 | 0 | 4 | 2 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 2 | 9 | 9 | 0 | 1 | 5 | 9 | 8 | 8 | 4 | 9 | 10 | 8 | 10 | 8 | 7 |
| Spark MLlib Imputer | 0 | 3 | 0 | 3 | 1 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 8 | 6 | 9 | 0 | 1 | 3 | 8 | 9 | 8 | 3 | 8 | 10 | 6 | 10 | 8 | 7 |
| H2O-3 | 0 | 3 | 0 | 3 | 2 | 0 | 0 | 0 | 2 | 5 | 0 | 1 | 0 | 1 | 4 | 4 | 8 | 0 | 1 | 4 | 8 | 6 | 7 | 3 | 6 | 7 | 6 | 10 | 6 | 7 |
| KNIME Missing Value | 1 | 3 | 0 | 5 | 3 | 0 | 0 | 0 | 4 | 0 | 0 | 1 | 0 | 5 | 5 | 3 | 5 | 0 | 2 | 4 | 8 | 7 | 7 | 2 | 6 | 9 | 9 | 8 | 7 | 7 |
| Orange Impute | 1 | 2 | 0 | 4 | 3 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 1 | 4 | 2 | 2 | 0 | 2 | 3 | 8 | 6 | 7 | 2 | 6 | 8 | 9 | 8 | 6 | 5 |
| RapidMiner | 1 | 3 | 0 | 5 | 3 | 0 | 0 | 0 | 3 | 0 | 3 | 2 | 0 | 3 | 5 | 3 | 4 | 0 | 2 | 4 | 8 | 6 | 7 | 2 | 5 | 6 | 8 | 2 | 6 | 8 |
| DataRobot | 1 | 3 | 0 | 3 | 5 | 0 | 0 | 0 | 2 | 1 | 0 | 2 | 2 | 4 | 8 | 4 | 8 | 1 | 2 | 5 | 6 | 6 | 6 | 3 | 5 | 8 | 8 | 1 | 6 | 9 |
| Great Expectations | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 7 | 3 | 6 | 7 | 0 | 1 | 5 | 8 | 8 | 8 | 2 | 8 | 10 | 6 | 10 | 8 | 7 |
| Soda | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 2 | 6 | 7 | 0 | 1 | 4 | 7 | 6 | 7 | 2 | 7 | 8 | 7 | 10 | 7 | 7 |

## 3. Per-tool justification (grouped; "unv." = unverified)
- **SAS**: F6 FCS, MCMC, monotone, PMM in PROC MI; F7 MIANALYZE pooling, multivariate tests. F8 MNAR statement (MODEL/ADJUST; control-based, tipping point) https://support.sas.com/documentation/cdl/en/statug/68162/HTML/default/statug_mi_syntax08.htm. F18 PROC SURVEYIMPUTE (hot-deck, weighted). F3 EM only, no dedicated Little MCAR in MI (unv.). F1 sparse ODS graphics. F9 via PROC EXPAND/ETS. F13 trace/ACF plots. F20 SEED=. N4 JSS paper https://www.jstatsoft.org/v45/i06. N5 uncertain (closed). N8 commercial, high cost. N1/N2/N10 decades of pharma use.
- **SPSS**: MVA gives Little's MCAR, EM, pattern tables https://www.ibm.com/docs/en/SSLVMB_31.0.0/pdf/IBM_SPSS_Missing_Values.pdf. MULTIPLE IMPUTATION FCS/monotone with PMM; pooled output for supported procedures only. No MNAR sensitivity. UI localized in many languages. N8 add-on module cost.
- **Stata**: mi impute chained/mvn/monotone/pmm etc.; mi estimate with Barnard-Rubin df, mi test; panel/multilevel support https://www.stata.com/features/overview/multiple-imputation/. F8 manual delta only (4). F3 no built-in MCAR test. Stata 19 "What's new" lists no MI changes https://www.stata.com/new-in-stata/. N3 manuals and books.
- **JMP Pro**: Explore Missing Values: MVN, SVD, RPCA, automated low-rank imputation https://www.jmp.com/support/help/en/19.1/jmp/explore-missing-values.shtml. MI and Rubin pooling not documented there (scored low, unv.). Strong interactive GUI; Pro-tier price.
- **MATLAB**: fillmissing (constant, previous/next, linear, spline, pchip, makima, moving window, knn), fillmissing2, rmmissing https://www.mathworks.com/help/matlab/ref/fillmissing.html. No MI, no MCAR test. GPU/tall arrays help F17.
- **Mplus**: FIML default, H1/H0 imputation, Bayes, TYPE=IMPUTATION pooling (statmodel.com discussion pages). TYPE=COMPLEX not supported with MI in a 2019 forum post (may be outdated). Monte Carlo can generate missing data (F11 4).
- **Blimp**: Bayesian model-based and FCS multilevel MI (2-3 levels), mixed types via latent variables; Enders, Du & Keller 2020; guide v3 2023 https://ies.ed.gov/use-work/awards/model-based-multiple-imputation-multilevel-data-methodological-extensions-and-software-enhancements. Little viz/descriptives. Closed source, free.
- **JASP / jamovi**: no imputation module found; jamovi forum says none https://forum.jamovi.org/viewtopic.php?p=5787; only exclude/computed-variable mean substitution; third-party Little MCAR module for jamovi (June 2025, community, unv.). N7 10; N8 AGPL=8.
- **Impute.jl**: LOCF/NOCB, interpolation, mean/median, SVD, kNN, drop/replace, Tables.jl; v0.7.1 2026-10-04. No MI, no diagnostics.
- **Mice.jl**: FCS with pmm, norm, mean, sample only https://github.com/tom-metherell/Mice.jl; Rubin pooling over GLM; v0.4.1 2026-05-29; benchmark faster than R mice on large jobs (Julia Discourse). Young, single maintainer.
- **danfo.js**: fillNa/dropNa/isNa only (knowledge; unv. current status).
- **polars**: fill_null strategies, interpolate, null_count, drop_nulls; no statistical imputation; lazy/streaming engine (F17 9). Version unv.
- **Spark**: Imputer mean/median/mode, approximate median; categorical unsupported per docs https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.Imputer; Pipeline-integrated; distributed. PySpark 4.3.0 2026-10-30 (PyPI).
- **H2O**: h2o.impute mean/median/mode with group-by https://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-munging/imputing-data.html; GLM/DL MeanImputation or Skip plus plug_values; trees handle NA natively; GLRM low-rank completion (knowledge, F10 5). 3.46.0.12 2026-08-12.
- **KNIME**: Missing Value node with per-type defaults and per-column overrides; mean/median/most frequent/fixed/interpolation/moving average/remove; PMML record https://hub.knime.com/n/uVmaGQkzUOFCTqUe. Version 5.11 unv.
- **Orange**: Impute widget: average/most frequent, random, model-based, indicator https://orange3.readthedocs.io/projects/orange-visual-programming/en/latest/widgets/data/impute.html; 3.40.0 Dec 2025.
- **RapidMiner**: Replace/Declare/Impute Missing Values (kNN, model-based); Generate Missing Values operator for amputation (knowledge, unv.). Evidence is community forum only.
- **DataRobot**: automatic per-model imputation plus indicators, shown in Data Quality Handling Report https://docs.datarobot.com/en/modeling/analyze-models/describe/dq-report.html; no user-facing MI; leakage-safe pipelines. Enterprise pricing.
- **Great Expectations**: not-null expectations with mostly, Data Docs HTML https://docs.greatexpectations.io/docs/reference/learn/data_quality_use_cases/missingness; detection only. v1.24.1.
- **Soda**: missing_count/missing_percent checks in SodaCL; detection only. soda-core 4.30.0 (date unv.).
