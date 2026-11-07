# Contributing to missingly

First off, thank you for considering contributing to `missingly`. It's people like you that make open source such a great community.

## Where do I go from here?

If you've noticed a bug or have a feature request, please [open an issue](https://github.com/alisadeghiaghili/missingly/issues). It's generally best to discuss features before working on them to ensure they align with the project's goals.

## Fork & create a branch

If you're ready to contribute code, fork the repository and create a new branch for your feature or bug fix.

```bash
git checkout -b my-new-feature
```

## Get the code running

1.  Create a virtual environment.
2.  Install the package in editable mode with all the development dependencies:

```bash
pip install -e .[docs,test]
```

## Running tests

Please make sure to run the tests before submitting a pull request.

```bash
pytest
```

## Lint ratchet

Ruff lint and format checks run as a ratchet rather than a hard zero. The
rule set in `pyproject.toml` (`[tool.ruff.lint]`) mirrors the families SonarQube
reports. Existing debt is recorded per file and rule in
`tools/ruff_baseline.json`.

```bash
python tools/ruff_ratchet.py            # check against the baseline
python tools/ruff_ratchet.py --update   # rewrite the baseline after fixing debt
```

The check fails when a violation count rises, a new file/rule pair appears, or
a formatted file becomes unformatted. Fixing debt passes and prints a hint to
run `--update`; commit the refreshed baseline in the same pull request. Do not
raise the baseline to hide a new violation. Tests for the gate live in
`tests/test_ruff_ratchet.py`.

## Project rules

### Supported runtime

- **Target Python**: 3.9+. Target Python 3.9+ language and library features
  only; CI exercises Python 3.9 through 3.14.
- Core dependencies are `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`,
  `scikit-learn`, `statsmodels` and `jinja2`; `upsetplot`, Plotly, RTL support
  and Polars are optional extras. Do not add a dependency without prior
  discussion in an issue.
- Core code never imports `data_quality_toolkit`. The deprecated forwarding
  shims are the only call-time exception and are tracked for removal.

### Quality gate

- CI fails below the `fail_under` value in `pyproject.toml`,
  currently 90% line and branch coverage. That file is the single source of
  the threshold; documents state the gate and do not copy a measured figure.
- Do not mask uncovered code with exclusions.
- Every defect starts with a failing regression test, committed before the fix.
  A passing smoke test is not evidence that a statistical method is correct.
- Always test these edge cases: an all-missing column or row, no missing data,
  mixed dtypes (int, float, str, category, datetime), high-cardinality
  categoricals, single-row and single-column frames, and duplicate column
  names (which must raise rather than corrupt data).
- Shared fixtures live in the root `conftest.py`; tests live in `tests/`.

### Documentation policy

- Use NumPy-style docstrings. Every module has a module docstring. Every public
  function documents a one-line summary and Parameters, Returns, Examples (and
  Notes for algorithm-specific behaviour); cite the source for a published
  algorithm. Type hints are required on public signatures.
- Parameters, Returns, Examples sections must be executable or verifiable
  against the code; a claim in documentation needs code and a test behind it.
- Record every user-visible change in `CHANGELOG.md` under `[Unreleased]`. When
  public behaviour changes, update `README.md` and the relevant `docs/` pages
  in the same pull request.
- Do not claim large-data support without a benchmark and a tested execution
  contract. Native Polars missingness summaries exist; full Arrow/Polars
  execution and million-row benchmarks are planned work.

### Public API in examples

Examples in documentation use only supported names:

```python
df.miss.impute(strategy="mean")
simulate_mcar(df, frac=0.2, random_state=42)
from missingly.report import create_report
from missingly.timeseries import gap_table, miss_ts_summary, vis_ts_miss, impute_ts
```

### Errors

Fail loudly with an actionable message. Prefer the most specific domain
exception at public boundaries: `MissinglyError` (base), `ImputationError`,
`InsufficientDataError`, `InvalidStrategyError`, `MissingColumnError` and
`ConfigurationError`. Never silently return a wrong result when input is bad.

### Workflow

1. Read the relevant modules before changing them.
2. Write the failing test first, then the implementation with full docstrings.
3. Run the whole suite; no regressions, warnings, skips or xfails are accepted
   without an issue, a rationale and an expiry.
4. Use Conventional Commits with the accurate type (`test:`, `fix:`, `feat:`,
   `docs:`, `chore:`, `ci:`), one bounded change per pull request, linked to an
   issue.

## Submitting a pull request

When you're ready to submit your pull request, please provide a clear description of the changes you've made. If your PR addresses an existing issue, please reference it in the description.

Thank you for your contribution!
