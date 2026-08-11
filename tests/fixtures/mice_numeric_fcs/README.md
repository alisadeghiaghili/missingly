# R `mice` numeric FCS evidence fixture

`numeric_fcs.csv` is a small, deterministic, synthetic numeric fixture. It is
intentionally suitable for public version control and is never used as a claim
of real-world performance.

`generate_mice_reference.R` regenerates aggregate-only evidence using R 4.4.2,
the current CRAN `mice` package resolved by the workflow, `m = 3`, `maxit = 5`,
the `norm` method for every column, and seed `20260811`. The workflow records
the input and script SHA-256 values, validates the artifact schema, and uploads
the evidence for review.

This is a reproducibility harness, not a numerical conformance fixture. It
does **not** compare the output to Missingly or establish parity with R `mice`.
Only a future reviewed manifest with an explicit estimand, frozen reference
metrics, tolerances, and documented intentional differences may make a scoped
cross-tool comparison.
