# R `mice` numeric FCS evidence fixture

`numeric_fcs.csv` is a small, deterministic, synthetic numeric fixture. It is
intentionally suitable for public version control and is never used as a claim
of real-world performance.

`generate_mice_reference.R` regenerates aggregate-only evidence using R 4.4.2,
the pinned CRAN `mice` 3.19.0 package, `m = 3`, `maxit = 5`,
the `norm` method for every column, and seed `20260811`. The workflow records
the input and script SHA-256 values, validates the artifact schema, and uploads
the evidence for review.

`benchmark_manifest.json` freezes the reviewed aggregate outputs and their
absolute tolerances. CI regenerates the R artifact and rejects metric or
provenance drift. This is a scoped R-to-R regeneration contract, not a
comparison to Missingly and not a claim of general parity with R `mice`.
