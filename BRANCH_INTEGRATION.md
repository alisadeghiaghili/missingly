# Branch reconciliation record

Reviewed: 2026-08-03

## Decision

`origin/main` is the canonical Missingly Python package. The branch
`agent/missingly-analytics-upgrade` is a separate FastAPI/web-statistics product with no
shared Git ancestor. A Git merge would combine unrelated histories and silently broaden
Missingly beyond its missing-data charter. The safe integration strategy is therefore a
selective port of compatible capabilities onto a branch created from `origin/main`.

## Capability selection

| Capability | Canonical package | Analytics/web branch | Integration decision |
|---|---|---|---|
| pandas-native missing-data API | Mature package API and `.miss` accessor | Not the primary interface | Keep canonical implementation |
| sklearn compatibility | Transformer and pipeline support | Not the primary interface | Keep canonical implementation |
| Missingness diagnostics and plots | Static, interactive, RTL, report, MCAR | Narrow descriptive surface | Keep canonical implementation and repair contracts |
| Imputation | Simple, KNN/Gower, hot-deck, iterative, MICE helpers | Numeric workflow embedded in a web product | Keep canonical algorithms |
| Statistical TDD policy | Large suite but no standalone conformance policy | Strong closed-form/cross-tool gate document | Port and specialize in `TESTING.md` |
| Competitor analysis | Roadmap fragments in conventions | Evidence-based but much broader than missing data | Rewrite for missing-data competitors only |
| Reproducibility identity | Seeds in several APIs | Explicit input/settings SHA-256 | Port as package-native `provenance` utilities |
| FastAPI, auth, payments, file upload | Out of scope | Product infrastructure | Do not merge |
| GEE, Cox, mixed models, general tests | Explicitly out of scope | Broad analytics surface | Do not merge |
| Persian web pages and SEO content | Not a package concern | Product UI | Do not merge |

## Preserved history

The unrelated branch remains available on the remote and is not rewritten or deleted.
All integrated changes are ordinary commits based on `origin/main`, so reviewers can
audit each selected capability independently. No claim of statistical superiority is
inherited from either branch; such a claim requires the benchmarks defined in
`COMPETITOR_ANALYSIS.md` and `TESTING.md`.

