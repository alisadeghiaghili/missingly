# Frozen Little's MCAR oracle fixture

`benchmark_manifest.json` is the machine-readable source of truth for the
fixture identity, expected scalar metrics, tolerances, and provenance limits.
It intentionally stores no raw records. The current oracle was imported from
an upstream regression fixture that did **not** publish the R runtime, platform,
or exact `naniar` package version; it is therefore an independent numerical
regression guard, **not** cross-tool parity evidence. Regenerate it from a
locked R environment before making a parity claim.

`generate_naniar_reference.R` and the **Reproduce R statistical oracle**
workflow provide that locked regeneration path: Ubuntu 24.04, R 4.4.2, and
`naniar` v1.1.0. The workflow uploads a scalar-only evidence artifact recording
the R/package/platform and SHA-256 identities, then compares it against the
reviewed numerical oracle. A follow-up PR must replace the legacy provenance
gap in `benchmark_manifest.json` with the reviewed artifact values before the
fixture is labelled cross-tool parity evidence.

`airquality.csv` is the public R `datasets::airquality` data set, mirrored by
[Rdatasets](https://vincentarelbundock.github.io/Rdatasets/). Its SHA-256 is
`65d2c4afd976c169af9bb0bd97e9e78e1e8a185f1b52e2e3153e30f90c7fb5f8`.

The expected values in the manifest are frozen from `naniar::mcar_test(airquality)`
and its published regression test:

- statistic: `35.1061288689702`
- degrees of freedom: `14`
- p-value: `0.00141778113856683`
- missing patterns: `4`

The R implementation uses `norm::em.norm`/`getparam.norm` and cites Little
(1988). Sources: https://github.com/njtierney/naniar/blob/main/tests/testthat/test-mcar-test.R
and https://github.com/njtierney/naniar/blob/main/R/mcar-test.R.

This fixture is vendored so test execution neither downloads data nor requires R.
