# Missing-data tool scoring rubric (0–10)

Date of assessment: 2026-11-06. Score what the tool ships TODAY (latest stable release), not roadmaps.

General anchors (apply to every dimension):
- 0  = absent / not applicable
- 1–2 = trivial (one basic function, e.g. "fillna(mean)")
- 3–4 = basic but usable subset
- 5–6 = solid, covers common cases, notable gaps
- 7–8 = strong, covers most professional needs
- 9   = near reference level
- 10  = THE reference implementation others are judged against (at most 1–2 tools per dimension)

## Feature dimensions (F)
F1  Missingness visualization — breadth (matrix, bar, heatmap, dendrogram, upset, shadow/margin plots, per-variable, interactive) and quality. Anchors: naniar ≈ 10, missingno ≈ 8, VIM ≈ 9.
F2  Descriptive summaries & pattern analysis — counts/props by var/case/group, pattern tables, shadow matrices, run lengths, co-occurrence. Anchor: naniar ≈ 10, mice::md.pattern ≈ 6.
F3  Mechanism diagnostics — Little's MCAR test, other MCAR tests (Jamshidian–Jalal, etc.), MAR evidence (missingness models), mechanism inference. Anchor: SPSS MVA / naniar mcar_test / MissMecha / MissMech(R).
F4  Single-imputation method breadth — simple, regression, KNN, hot-deck, LOCF, tree, EM etc.
F5  Mixed-type / categorical handling — correct treatment of nominal/ordinal/binary (not ordinal-encoding hacks), Gower etc.
F6  Multiple-imputation engine — FCS/MICE and/or joint model, PMM, per-variable methods, predictor matrix, blocks, passive imputation, visit order. Anchor: R mice = 10.
F7  MI pooling & inference — Rubin's rules, Barnard–Rubin df, multivariate tests (D1/D2/D3), breadth of pooled models. Anchor: R mice/mitml, SAS MIANALYZE, Stata mi estimate ≈ 9–10.
F8  MNAR sensitivity analysis — delta adjustment, tipping point, pattern-mixture, selection models, reference-based (J2R/CIR). Anchor: R rbmi / SAS PROC MI MNAR statement ≈ 9–10.
F9  Time-series imputation — interpolation, seasonal decomposition, Kalman/state-space, ARIMA, TS-specific viz. Anchor: R imputeTS = 10, PyPOTS (deep) ≈ 9.
F10 Advanced ML / matrix completion / deep imputation — softImpute, iterative SVD, missForest, GAIN, MIWAE, VAE, diffusion etc.
F11 Missingness simulation (amputation) — generate MCAR/MAR/MNAR. Anchor: mice::ampute, pyampute, MissMecha.
F12 Imputation evaluation & benchmarking — compare methods, masking-based accuracy, downstream-model CV.
F13 Imputation diagnostics — convergence traces, density/strip overlays, overimputation, plausibility checks.
F14 Reporting — automated HTML/PDF report, publication tables, export.
F15 ML pipeline integration — sklearn-style fit/transform, pipeline use, no leakage.
F16 Data-frame / data-stack integration — pandas/tidyverse/data.table/polars/Arrow native fit.
F17 Scale & performance — large data, out-of-core, GPU, distributed, measured speed.
F18 Multilevel / survey / longitudinal support — clustered/2-level imputation, survey weights, panel/wide-long.
F19 Internationalization — multilingual output/docs, RTL support.
F20 Provenance & reproducibility — seeds, deterministic output, audit trail of what was imputed, fingerprints.

## Non-feature dimensions (N)
N1  Maturity & API stability — age, release history, semantic stability.
N2  Adoption & community — downloads, stars, citations, StackOverflow presence.
N3  Documentation & learning resources — reference docs, vignettes, books, courses.
N4  Statistical validation — peer-reviewed method papers, reference-tested against other tools, published benchmarks.
N5  Engineering quality — tests/coverage, CI, typing, code health.
N6  Maintenance activity — releases/commits in the last 12 months (as of 2026-11).
N7  Ease of use / learning curve — 10 = productive in minutes.
N8  License openness & cost — 10 = permissive OSS (MIT/BSD/Apache), 8 = GPL, 4–5 = source-available (BUSL etc.), 1–3 = commercial license, cost scales down.
N9  Ecosystem interoperability — works with surrounding tools of its ecosystem and beyond.
N10 Support — commercial support, responsive maintainers, issue tracker health.

Rules:
- Every score needs a one-line justification grounded in a source (URL or file:line).
- If you cannot verify something, say "unverified" and score conservatively; do NOT inflate.
- Be calibrated across tools: a 10 means "reference"; most scores for most tools should be 0–7.
