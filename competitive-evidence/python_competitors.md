# Python missing-data tools, scored 2026-11-06 (rubric.md applied)

Data sources: PyPI JSON API (version/date/license), GitHub REST API (stars, pushed_at, archived), pypistats.org `recent` API (last_month downloads), GitHub README pages. "unverified" = not confirmed. Downloads are PyPI-only (include mirrors/bots/transitive installs).

## 1. Metadata

| Tool | Version (PyPI) | Release date | License | GitHub stars | Downloads/month | Maintenance (last push) |
|---|---|---|---|---|---|---|
| missingno | 0.5.2 | 2023-02-26 | MIT | 4,211 | 248,754 | Stale: last push 2024-05-14, no release in 3.7 yr |
| scikit-learn (sklearn.impute) | 1.9.1 | 2026-09-11 | BSD-3 | 68,020 | 184,326,320 | Very active (push 2026-11-06) |
| statsmodels (imputation.mice) | 0.15.0 | 2026-08-27 (GitHub release) | BSD-3 | 11,743 | 32,097,137 | Active (push 2026-11-05); MICE module itself little changed |
| miceforest | 6.0.5 | 2025-10-27 | MIT (PyPI; GitHub shows none) | 415 | 31,552 | Moderate: last push 2025-10-27 (~12 mo) |
| autoimpute | 0.14.1 | 2023-12-30 | MIT | 252 | 118 | Abandoned (push 2023-12-30) |
| hyperimpute | 0.1.17 | 2023-02-28 | MIT | 202 | 916 | Abandoned (push 2023-04-04) |
| fancyimpute | 0.7.0 | 2021-10-21 | Apache-2.0 | 1,279 | 43,794 | Abandoned (push 2023-10-25) |
| missingpy | 0.2.0 | 2018-12-10 | GPL-3.0 | 246 | 157,814 (suspiciously high, likely transitive/mirror; unverified cause) | Abandoned (push 2024-02-29) |
| missforest (PyPI "MissForest") | 4.2.3 | 2025-01-03 | MIT | 80 | 1,139 | Light activity (push 2026-08-14) |
| scikit-na | 0.4.1 | 2026-08-29 | MIT | 5 | 79 | Active (push 2026-08-29) |
| MissMecha (missmecha-py) | 0.1.2 | 2025-07-29 | MIT | 111 | 18 | Light (push 2026-06-23) |
| NApy (PyPI "napypi") | 2.1 | 2026-10-02 | GPL-3.0 | 9 | 799 | Active (push 2026-10-02) |
| PyPOTS | PyPI page shows 1.5 (date unverified); GitHub release v2.0 | v2.0: 2026-10-30 | BSD-3 | 2,093 | 61,227 | Very active (push 2026-10-31) |
| ImputeGAP | 1.1.2 | 2026-01-21 | MIT | 70 | 82 (suspiciously low; unverified) | Very active (push 2026-11-05) |
| pyampute | 0.0.3 | 2022-08-26 | BSD (GitHub: NOASSERTION) | 75 | 3,882 | Abandoned (push 2022-12-17) |
| ydata-profiling / fg-data-profiling | 4.20.0 (fg-data-profiling, 2026-09-11); ydata-profiling 4.18.4 (2026-04-22, "renamed" notice) | 2026-09-11 | MIT | 13,752 | 661,379 (ydata-profiling name only) | Active (push 2026-11-05) |
| impyute | 0.0.8 | 2019-04-29 | GPL-3.0 on PyPI (GitHub says MIT; conflict) | 360 | 862 | Abandoned (push 2021-11-06) |
| datawig | 0.2.0 | 2020-07-16 | Apache-2.0 | 492 | 206 | ARCHIVED (repo archived; last push 2026-01-14) |
| MIDASpy (newcomer/extra) | 1.4.2 | 2026-03-08 | Apache-2.0 | 160 | 209 | Release in 2026, last commit 2024-08-31 |
| mipy-impute (2026 newcomer) | 2.2.0 | 2026-11-03 | MIT | unverified | 0 (pypistats; brand new) | Very new, active |
| pandas / polars built-ins (baseline) | unverified | unverified | BSD-3 / MIT | unverified | unverified | Presumed very active; not fetched |

Notes: no `statsmodels`-style MICE standalone download count exists (it ships with statsmodels). pyampute PyPI license "BSD". NApy on PyPI "napy" is an unrelated 2018 package; the real tool is `napypi`.

## 2. Scores (0-10)

Columns: F1 viz, F2 summaries, F3 mechanism tests, F4 single imp, F5 mixed-type, F6 MI engine, F7 pooling, F8 MNAR sens., F9 time series, F10 ML/deep, F11 amputation, F12 evaluation, F13 diagnostics, F14 reporting, F15 ML pipeline, F16 dataframe, F17 scale, F18 multilevel, F19 i18n, F20 provenance; N1 maturity, N2 adoption, N3 docs, N4 validation, N5 eng quality, N6 maintenance, N7 ease, N8 license, N9 interop, N10 support.

| Tool | F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | F11 | F12 | F13 | F14 | F15 | F16 | F17 | F18 | F19 | F20 | N1 | N2 | N3 | N4 | N5 | N6 | N7 | N8 | N9 | N10 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| missingno | 8 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 5 | 2 | 0 | 0 | 0 | 6 | 9 | 5 | 2 | 4 | 2 | 10 | 10 | 5 | 2 |
| sklearn.impute | 0 | 1 | 0 | 6 | 3 | 3 | 0 | 0 | 1 | 4 | 0 | 2 | 1 | 0 | 10 | 6 | 5 | 0 | 0 | 4 | 10 | 10 | 9 | 8 | 9 | 10 | 8 | 10 | 10 | 6 |
| statsmodels MICE | 2 | 2 | 0 | 1 | 2 | 5 | 5 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 1 | 6 | 2 | 2 | 0 | 3 | 9 | 6 | 6 | 6 | 8 | 8 | 5 | 10 | 6 | 5 |
| miceforest | 0 | 1 | 0 | 3 | 6 | 7 | 0 | 0 | 0 | 6 | 0 | 3 | 6 | 0 | 5 | 5 | 6 | 0 | 0 | 4 | 7 | 6 | 6 | 3 | 6 | 4 | 8 | 10 | 6 | 4 |
| autoimpute | 2 | 2 | 1 | 6 | 5 | 6 | 6 | 1 | 3 | 1 | 0 | 2 | 4 | 0 | 5 | 4 | 1 | 0 | 0 | 3 | 4 | 3 | 6 | 2 | 4 | 0 | 6 | 10 | 3 | 1 |
| hyperimpute | 0 | 0 | 0 | 6 | 5 | 2 | 0 | 0 | 0 | 8 | 3 | 4 | 0 | 0 | 7 | 4 | 4 | 0 | 0 | 3 | 4 | 3 | 4 | 6 | 6 | 0 | 6 | 10 | 4 | 1 |
| fancyimpute | 0 | 0 | 0 | 4 | 1 | 1 | 0 | 0 | 0 | 6 | 0 | 1 | 0 | 0 | 4 | 2 | 2 | 0 | 0 | 2 | 6 | 6 | 4 | 4 | 3 | 0 | 7 | 10 | 3 | 1 |
| missingpy | 0 | 0 | 0 | 3 | 2 | 0 | 0 | 0 | 0 | 5 | 0 | 0 | 0 | 0 | 6 | 2 | 1 | 0 | 0 | 1 | 2 | 3 | 2 | 1 | 2 | 0 | 7 | 8 | 3 | 0 |
| missforest (PyPI) | 0 | 0 | 0 | 3 | 5 | 0 | 0 | 0 | 0 | 5 | 0 | 0 | 0 | 0 | 4 | 4 | 3 | 0 | 0 | 2 | 3 | 2 | 3 | 1 | 3 | 5 | 8 | 10 | 3 | 2 |
| scikit-na | 5 | 6 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 0 | 5 | 2 | 0 | 0 | 1 | 3 | 1 | 5 | 1 | 4 | 6 | 7 | 10 | 4 | 2 |
| MissMecha | 4 | 3 | 5 | 2 | 2 | 0 | 0 | 0 | 0 | 0 | 8 | 4 | 0 | 0 | 4 | 5 | 2 | 0 | 0 | 4 | 1 | 1 | 5 | 3 | 4 | 5 | 7 | 10 | 3 | 2 |
| NApy | 0 | 1 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 3 | 8 | 0 | 0 | 1 | 2 | 1 | 5 | 6 | 5 | 7 | 6 | 8 | 4 | 2 |
| PyPOTS | 1 | 0 | 0 | 4 | 1 | 2 | 0 | 0 | 9 | 9 | 4 | 6 | 1 | 1 | 5 | 3 | 7 | 1 | 0 | 5 | 7 | 7 | 8 | 7 | 6 | 10 | 5 | 10 | 7 | 5 |
| ImputeGAP | 4 | 2 | 0 | 5 | 0 | 0 | 0 | 0 | 8 | 7 | 5 | 8 | 2 | 2 | 2 | 3 | 3 | 0 | 0 | 4 | 4 | 2 | 6 | 5 | 6 | 10 | 6 | 10 | 4 | 3 |
| pyampute | 2 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 7 | 0 | 0 | 0 | 5 | 4 | 1 | 0 | 0 | 3 | 3 | 2 | 6 | 5 | 4 | 0 | 7 | 10 | 4 | 1 |
| ydata-profiling | 5 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 7 | 0 | 7 | 5 | 0 | 1 | 2 | 8 | 9 | 7 | 3 | 7 | 9 | 9 | 10 | 7 | 6 |
| impyute | 0 | 0 | 0 | 5 | 1 | 1 | 0 | 0 | 2 | 2 | 0 | 0 | 0 | 0 | 1 | 2 | 1 | 0 | 0 | 1 | 3 | 3 | 4 | 1 | 3 | 0 | 6 | 8 | 3 | 0 |
| datawig | 0 | 0 | 0 | 4 | 6 | 0 | 0 | 0 | 0 | 6 | 0 | 2 | 0 | 0 | 3 | 5 | 3 | 0 | 0 | 2 | 4 | 4 | 4 | 5 | 4 | 0 | 6 | 10 | 2 | 0 |
| MIDASpy | 0 | 0 | 0 | 0 | 5 | 6 | 5 | 0 | 0 | 7 | 0 | 2 | 1 | 0 | 2 | 4 | 5 | 0 | 0 | 3 | 5 | 3 | 4 | 7 | 4 | 4 | 4 | 10 | 3 | 2 |
| mipy-impute | 3 | 2 | 2 | 6 | 3 | 4 | 5 | 2 | 2 | 4 | 5 | 8 | 3 | 7 | 1 | 4 | 2 | 5 | 0 | 3 | 2 | 0 | 3 | 3 | 3 | 10 | 8 | 10 | 2 | 1 |
| pandas/polars built-ins | 0 | 2 | 0 | 3 | 1 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 10 | 6 | 0 | 0 | 2 | 10 | 10 | 9 | 6 | 9 | 10 | 10 | 10 | 10 | 6 |

## 3. Justifications and sources

General: N6 scored from release/push dates vs 2026-11 (release within 12 mo and push within 6 mo = 8-10; 12-24 mo = 3-5; older = 0-2). N8 per rubric (MIT/BSD/Apache 10, GPL 8). All stats from pypi.org/pypi/<pkg>/json, api.github.com/repos/<owner>/<repo>, pypistats.org/api/packages/<pkg>/recent.

**missingno** (https://github.com/ResidentMario/missingno): scope is matrix, bar, heatmap, dendrogram plots of nullity only; no imputation or tests.
- F1 8: four plot types, matplotlib only, no interactive/upset/shadow plots (known README).
- F2 3: nullity correlation only, no pattern tables or group summaries.
- N6 2: last release 2023-02, last push 2024-05 (https://api.github.com/repos/ResidentMario/missingno).
- N2 9: 248k monthly downloads, 4.2k stars.

**sklearn.impute** (https://scikit-learn.org/stable/modules/impute.html): SimpleImputer, KNNImputer, IterativeImputer (experimental), MissingIndicator.
- F15 10: reference implementation of fit/transform and Pipeline use.
- F6 3: IterativeImputer sample_posterior gives draws but no multi-dataset workflow.
- F5 3: categorical only via most_frequent/constant; KNN/Iterative numeric only.
- F16 6: set_output supports pandas/polars (v1.2+).
- N1/N2/N6 10: 1.9.1 on 2026-09-11, 68k stars, 184M downloads/month (pypistats).
- N4 8: peer-reviewed library paper; imputers themselves less benchmarked.

**statsmodels MICE** (https://www.statsmodels.org/stable/imputation.html): MICEData with PMM, formulas, Rubin pooling via MICE.fit.
- F6 5: chained equations, PMM, per-variable formulas, no passive imputation/blocks.
- F7 5: pooled results for any statsmodels model, no D1/D2/D3 or Barnard-Rubin (unverified df handling; conservative).
- F13 2: plot_missing_pattern, plot_fit_obs, plot_bootstrap-style helpers only.
- N6 8: statsmodels 0.15.0 2026-08-27, MICE module rarely changed.

**miceforest** (https://github.com/AnotherSamWilson/miceforest, https://pypi.org/project/miceforest/): LightGBM-based FCS with PMM.
- F6 7: multiple datasets, PMM variants, per-variable LightGBM params, variable schema.
- F13 6: convergence/feature-importance/distribution plots per README.
- F17 6: LightGBM, GPU option, in-place imputation.
- F7 0: no pooling or analysis of imputed sets.
- N6 4: last release/push 2025-10-27 (~12 months).
- N8 10: MIT on PyPI, but GitHub license field empty (flag).

**autoimpute** (https://github.com/kearnz/autoimpute): single/multiple imputation with pooled linear/logistic regression.
- F7 6: Rubin pooling for linear/logistic models (README, unverified depth).
- F6 6: MultipleImputer with PMM, Bayesian, etc.
- N6 0: abandoned since 2023-12; 118 downloads/month.

**hyperimpute** (https://github.com/vanderschaarlab/hyperimpute): ICML 2022 AutoML imputation framework (plugins: GAIN, MIWAE, MICE, missForest, sinkhorn, softimpute).
- F10 8: broadest deep plugin set among tabular tools.
- F11 3: simulate_nan helper (unverified depth).
- N4 6: ICML 2022 paper with benchmarks.
- N6 0: push 2023-04; 916 downloads/month.

**fancyimpute** (https://github.com/iskandr/fancyimpute): SoftImpute, IterativeSVD, NuclearNorm, MatrixFactorization, KNN.
- F10 6: matrix completion references (Mazumder et al.).
- N6 0: last release 2021-10, push 2023-10.
- N2 6: 1.3k stars, 43.8k downloads/month.

**missingpy** (https://github.com/epsilon-machine/missingpy): KNN and MissForest in sklearn style.
- N6 0: 2018 release, push 2024-02.
- N2 3: 157k downloads/month look transitive/bot-driven (unverified); 246 stars.

**missforest (PyPI)** (https://github.com/yuenshingyan/MissForest): Random forest iterative imputer with categorical support.
- F5 5: handles categorical columns (README, unverified depth).
- N6 5: release 2025-01, push 2026-08.

**scikit-na** (https://github.com/maximtrp/scikit-na): NA summaries, Altair plots, logistic missingness model, MCAR/hypothesis tests, export.
- F2 6: summary/describe/correlate by missingness.
- F3 4: test_hypothesis includes MCAR tests; model() logistic missingness (details unverified).
- F1 5: Altair heatmap, stairs, histogram, interactive widget.
- N2 1: 5 stars, 79 downloads/month.
- N6 6: 0.4.1 released 2026-08-29.

**MissMecha** (https://github.com/echoid/MissMecha, https://echoid.github.io/MissMecha/): simulate MCAR/MAR/MNAR (many subtypes), MCARTest, plots, SimpleSmartImputer, evaluation.
- F11 8: three mechanism families with multiple subtypes plus custom, per-column generator.
- F3 5: MCARTest exists; which tests (Little's?) unverified.
- N2 1: 18 downloads/month, 111 stars.
- N4 3: demo-track paper only.

**NApy / napypi** (https://github.com/DyHealthNet/NApy, https://arxiv.org/abs/2505.00448): C++/Numba pairwise-deletion statistics on mixed data; no imputation.
- F17 8: OpenMP C++ backend; benchmarked orders of magnitude faster (paper).
- N4 6: GigaScience 2025 paper plus benchmark repo.
- N8 8: GPL-3.0.
- F4/F6 0: explicitly ignores missing pairwise, no imputation.

**PyPOTS** (https://github.com/WenjieDu/PyPOTS, https://docs.pypots.com): deep partially-observed time series library (imputation, forecasting, classification, clustering).
- F9 9 / F10 9: dozens of neural imputers (SAITS, BRITS, CSDI, etc.), PyTorch.
- F12 6: benchmarking via BenchPOTS, metrics (separate packages).
- N6 10: v2.0 released 2026-10-30; PyPI version/date shown as 1.5 (unverified conflict).
- N4 7: multiple peer-reviewed model papers by the author.

**ImputeGAP** (https://github.com/eXascaleInfolab/ImputeGAP, https://arxiv.org/abs/2503.15250): 40+ time-series imputers, 8 missingness patterns, benchmark, SHAP explainer.
- F9 8: broad TS algorithm coverage, matrix-completion families.
- F12 8: unified benchmark pipeline with custom metrics.
- F11 5: 8 patterns (MCAR + TS-specific).
- N2 2: 82 downloads/month looks anomalously low (unverified), 70 stars.

**pyampute** (https://github.com/RianneSchouten/pyampute): multivariate amputation (MCAR/MAR/MNAR with pattern weights), port of mice::ampute.
- F11 7: faithful ampute port, not extended since 2022.
- N6 0: 0.0.3, 2022-08.
- N4 5: published method paper (Schouten et al.), unverified citation details.

**ydata-profiling / fg-data-profiling** (https://github.com/ydataai/ydata-profiling): general EDA HTML report; missing section uses missingno-style bar/matrix/heatmap/dendrogram.
- F1 5: four missing plots inside general report.
- F14 7: strong HTML report, but not missing-specific.
- N1 8 note: package renamed to fg-data-profiling (PyPI banner); API continuity risk.

**impyute** (https://github.com/eltonlaw/impyute): mean, LOCF, interpolation, KNN, EM, MICE for numpy arrays.
- N6 0: 2019 release, 2021 push.
- N8 8: PyPI says GPL-3.0, GitHub says MIT (conflict).

**datawig** (https://github.com/awslabs/datawig): MXNet deep imputer for tabular/categorical/text columns.
- F5 6: categorical and text columns handled natively (CIKM 2019 design).
- N6 0: repo archived (api.github.com/repos/awslabs/datawig archived=true), last release 2020-07.

**MIDASpy** (https://github.com/MIDASverse/MIDASpy): denoising autoencoder MI, combine() applies Rubin rules.
- F6 6, F7 5: multiple imputations plus regression pooling.
- N4 7: published (Lall and Robinson) political-analysis paper.
- N6 4: release 2026-03-08, last commit 2024-08.

**mipy-impute** (https://pypi.org/project/mipy-impute/): 2026 newcomer, benchmark-then-impute for high-dimensional longitudinal data, HTML report, pool_estimates (Rubin).
- F12 8 / F14 7: automated benchmark under three missingness assumptions plus HTML report (README, unverified details).
- F18 5: longitudinal ID/visit handling.
- N2 0 / N1 2: released 2026, 0 downloads/month in pypistats, stars unverified; score conservatively.

**pandas/polars baseline**: fillna/ffill/interpolate, fill_null/interpolate.
- Versions, stars, downloads not fetched (unverified); scores reflect standard known capabilities, F16 10 as native frame.

Other findings: no 2026 Python package found for MNAR sensitivity analysis (F8 max score here is 2), no Python Little's test/MI engine matching R mice found via search. Search did not exhaustively check autoencoder/diffusion tabular newcomers.
