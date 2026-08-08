"""TDD contracts for the future unified multiple-imputation architecture."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import missingly
from missingly.mi_contracts import ImputationPlan, ImputationResult, MIData, MissingnessSchema


@pytest.fixture()
def original() -> pd.DataFrame:
    """Return a small, typed incomplete frame with an unambiguous index."""
    return pd.DataFrame({"age": [20.0, np.nan], "group": ["a", "b"]}, index=[10, 20])


@pytest.fixture()
def schema(original) -> MissingnessSchema:
    """Return the original-data schema for the MI contract tests."""
    return MissingnessSchema.from_dataframe(original)


@pytest.fixture()
def plan() -> ImputationPlan:
    """Return a reproducible two-chain plan."""
    return ImputationPlan({"age": "pmm", "group": "logreg"}, 2, 5, (101, 202))


def test_schema_captures_mask_without_mutating_input(original, schema):
    """Schema construction preserves caller data and validates structural identity."""
    assert schema.original_mask.to_dict("list") == {"age": [False, True], "group": [False, False]}
    assert original.iloc[1, 0] != original.iloc[1, 0]
    schema.validate_frame(original.copy())


def test_schema_and_plan_reject_ambiguous_or_unknown_contracts(original, schema):
    """Invalid alignment, methods, and seed sequences must fail before execution."""
    with pytest.raises(ValueError, match="duplicate"):
        MissingnessSchema.from_dataframe(pd.DataFrame([[1, 2]], columns=["x", "x"]))
    with pytest.raises(ValueError, match="one seed"):
        ImputationPlan({"age": "pmm"}, 2, 3, (1,))
    with pytest.raises(ValueError, match="duplicate seeds"):
        ImputationPlan({"age": "pmm"}, 2, 3, (1, 1))
    with pytest.raises(ValueError, match="unknown schema columns"):
        ImputationPlan({"missing": "pmm"}, 1, 3, (1,)).validate_schema(schema)


def test_mi_data_preserves_observed_cells_and_result_validates_histories(original, schema, plan):
    """Completed chains retain observed values and traces align with chain count."""
    chain_a = pd.DataFrame({"age": [20.0, 30.0], "group": ["a", "b"]}, index=[10, 20])
    chain_b = pd.DataFrame({"age": [20.0, 31.0], "group": ["a", "b"]}, index=[10, 20])
    data = MIData(original, schema, plan, (chain_a, chain_b))
    result = ImputationResult(data, histories=({"age": (25.0, 25.2)}, {"age": (25.5, 25.4)}))

    assert missingly.MIData is MIData
    assert data.n_imputations == 2
    assert result.validation == {"result_valid": True}

    changed = chain_a.copy()
    changed.loc[10, "age"] = 999.0
    with pytest.raises(ValueError, match="observed cells"):
        MIData(original, schema, plan, (changed, chain_b))
    with pytest.raises(ValueError, match="one mapping per imputation"):
        ImputationResult(data, histories=({"age": (1.0,)},))
