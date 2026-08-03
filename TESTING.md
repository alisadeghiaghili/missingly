# Statistical testing strategy

Missingly uses test-driven development for every analytical capability. A method is not
advertised as supported until its contract, numerical result, invalid-input behavior,
serialization, documentation, API endpoint, and release build are covered.

## Test layers

1. **Closed-form conformance** — small datasets with hand-calculable estimates establish
   coefficient, survival, and transformation correctness without using the implementation
   as its own oracle.
2. **Deterministic simulation** — seeded datasets verify effect direction, recovery
   tolerance, convergence handling, and reproducibility for models without simple closed
   forms.
3. **Statistical invariants** — tests enforce finite JSON output, explicit event/reference
   coding, preservation of observed values, valid confidence intervals, and deterministic
   fingerprints.
4. **Guardrails** — invalid support, singular designs, insufficient clusters, duplicate
   longitudinal times, unsupported categories, and unsafe inputs must fail explicitly.
5. **API contracts** — authenticated endpoints must match the analytical core and must not
   persist raw records.
6. **Cross-tool benchmarks** — before claiming parity with an R, Python, SAS, SPSS, or Stata
   procedure, add a frozen fixture, tool/version metadata, expected estimates and
   uncertainty, and justified absolute/relative tolerances.

## TDD cycle

Each capability begins with one focused failing test. Implement the smallest statistically
valid behavior, make the focused test pass, run the complete suite, build the release
archive, and only then commit. Bug fixes require a regression test that fails before the
fix. Unsupported future capabilities belong in the roadmap, not as skipped tests that
inflate apparent coverage.

## Numerical tolerance policy

- Exact algebraic identities use an absolute tolerance no larger than 1e-10.
- Deterministic simulations use a tolerance justified by sample size and estimator
  variability.
- Cross-tool comparisons store both absolute and relative tolerances and explain known
  differences such as tie handling, contrast coding, likelihood constants, or covariance
  corrections.
- Tests never assert only that a p-value is significant; they verify estimand, direction,
  magnitude, uncertainty, and coding metadata.

## Required CI gates

The full pytest suite, public-docstring contract, Python compilation, release build, and
Git diff check must pass. Environment-dependent UI tests must be enabled in release CI;
a local skip is not evidence that the UI contract passed.
