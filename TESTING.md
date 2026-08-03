# Statistical testing strategy

Missingly uses test-driven development for missing-data diagnostics, imputation,
comparison, simulation, and inference. A capability is not considered supported until
its numerical contract, failure behavior, public documentation, and release artifact are
covered. Unsupported future work belongs in the roadmap, not in placeholder tests that
are permanently skipped.

## Test layers

1. **Closed-form conformance** — hand-calculable fixtures verify statistics without
   using Missingly as its own oracle. Exact algebra uses absolute tolerance at most
   `1e-10`.
2. **Deterministic simulation** — seeded MCAR, MAR, and MNAR fixtures verify recovery,
   convergence, invariants, and sensitivity within a documented tolerance.
3. **Data invariants** — imputation preserves observed values, index, column order, and
   supported dtypes; pooled variance is valid; probabilities and p-values remain in
   their mathematical domains.
4. **Guardrails** — all-missing variables, singular systems, invalid variance, unknown
   strategies, impossible constraints, and non-convergence fail explicitly or return a
   documented diagnostic state.
5. **API and ecosystem contracts** — the pandas accessor, sklearn transformer, static
   and interactive plots, HTML report, and optional DQT bridge agree with the core API.
6. **Cross-tool benchmarks** — parity claims require frozen data, a licensed source,
   checksum, exact competitor version, expected estimate and uncertainty, and justified
   absolute and relative tolerances.

## TDD cycle

For a bug or capability: add one focused failing regression test, implement the smallest
statistically valid behavior, run the focused tests, run the complete suite with coverage,
build both wheel and source distribution, validate their metadata, and only then commit.
Skipped integration tests must name the unavailable external dependency and must run in a
dedicated CI job where that dependency is installed.

## Numerical review rules

- Never test only whether a p-value crosses `0.05`; test the estimand, direction,
  magnitude, uncertainty, degrees-of-freedom rule, and edge behavior.
- Distinguish Rubin's large-sample degrees of freedom from the Barnard–Rubin
  small-sample adjustment. Distinguish `lambda` (variance fraction attributable to
  missingness) from `fmi` (small-sample fraction of missing information).
- Tests for stochastic algorithms must set a local seed and must not mutate global RNG
  state.
- Cross-tool tolerances must explain known differences in category coding, likelihood
  constants, optimizers, donor selection, or finite-sample corrections.
- A local skip is not a pass and is excluded from claims of coverage or parity.

## Required release gates

- Full pytest suite and a non-decreasing coverage ratchet.
- Public API, docstring, doctest, and Python compilation checks.
- Wheel and source-distribution builds followed by metadata validation.
- Dependency consistency and clean Git diff checks.
- Dedicated execution of optional integration suites used by advertised features.
- Independent numerical fixtures for every capability described as competitor parity.

