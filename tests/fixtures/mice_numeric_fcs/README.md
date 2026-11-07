# R `mice` numeric FCS evidence fixture

`numeric_fcs.csv` is a small, deterministic, synthetic numeric fixture. It is
intentionally suitable for public version control and is never used as a claim
of real-world performance.

`generate_mice_reference.R` regenerates aggregate-only evidence using R 4.4.2,
the pinned CRAN `mice` 3.19.0 package, `m = 3`, `maxit = 5`,
the `norm` method for every column, and seed `20260811`. The workflow records
the input and script SHA-256 values, validates the artifact schema, and uploads
the evidence for review.

`benchmark_manifest.json` freezes the reviewed R aggregate outputs and their
absolute tolerances. CI regenerates the R artifact and rejects metric or
provenance drift.

The workflow also runs `generate_missingly_reference.py` on the same CSV and
`validate_cross_tool_comparison.py` compares the Missingly output with R. That
comparison is the only place Missingly output is compared, and it is limited to
the four pooled column means (`pooled_mean_*`, three imputations, the shared
seed) inside the absolute bounds in `cross_tool_contract.json`. It is a scoped
check on this synthetic fixture, not a claim of general parity with R `mice`
or of imputation accuracy.

The comparison validator computes the CSV SHA-256 itself and requires the
Missingly artifact to come from the running `missingly` version. The
`missingly_version` field in `benchmark_manifest.json` is stale (`1.0.0`) and
is not used for this check. The manifest's `known_differences` entry, which
says no Missingly output is compared, predates the cross-tool comparison. It is
left unchanged because the manifest is frozen and awaits owner review.
