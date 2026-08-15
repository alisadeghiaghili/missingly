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


def test_plan_accepts_a_column_named_after_its_method(schema):
    """A valid column label must not be confused with a duplicate method name."""
    plan = ImputationPlan({"mean": "mean"}, 1, 3, (1,))

    plan.validate_schema(MissingnessSchema.from_dataframe(pd.DataFrame({"mean": [1.0]})))
    assert plan.methods == {"mean": "mean"}


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


def test_mi_data_rejects_chain_that_leaves_an_originally_missing_cell_unfilled(
    original, schema, plan
):
    """A completed chain must fill every cell that its schema marks as eligible."""
    incomplete_chain = original.copy()
    completed_chain = pd.DataFrame(
        {"age": [20.0, 31.0], "group": ["a", "b"]}, index=[10, 20]
    )

    with pytest.raises(ValueError, match="fill every originally missing cell"):
        MIData(original, schema, plan, (incomplete_chain, completed_chain))


@pytest.mark.parametrize(
    ("columns", "dtypes", "index", "mask", "error", "message"),
    [
        (("age",), (), pd.Index([10, 20]), pd.DataFrame({"age": [False, True]}, index=[10, 20]), ValueError, "equal length"),
        (("age",), ("float64",), [10, 20], pd.DataFrame({"age": [False, True]}, index=[10, 20]), TypeError, "pandas Index"),
        (("age",), ("float64",), pd.Index([10, 10]), pd.DataFrame({"age": [False, True]}, index=[10, 10]), ValueError, "duplicate labels"),
        (("age",), ("float64",), pd.Index([10, 20]), pd.DataFrame({"other": [False, True]}, index=[10, 20]), ValueError, "columns"),
        (("age",), ("float64",), pd.Index([10, 20]), pd.DataFrame({"age": [False, True]}, index=[11, 12]), ValueError, "index"),
        (("age",), ("float64",), pd.Index([10, 20]), pd.DataFrame({"age": [0, 1]}, index=[10, 20]), TypeError, "boolean"),
    ],
)
def test_schema_constructor_rejects_invalid_structural_contracts(
    columns, dtypes, index, mask, error, message
):
    """Direct construction rejects every ambiguous schema alignment boundary."""
    with pytest.raises(error, match=message):
        MissingnessSchema(columns, dtypes, index, mask)


@pytest.mark.parametrize(
    ("frame", "error", "message"),
    [
        ([1, 2], TypeError, "pandas DataFrame"),
        (pd.DataFrame(index=[0]), ValueError, "at least one column"),
        (pd.DataFrame([[1.0]], columns=[1]), TypeError, "columns must be strings"),
    ],
)
def test_schema_factory_rejects_inputs_without_an_unambiguous_tabular_identity(frame, error, message):
    """Schema creation rejects non-tabular, empty, and non-string-column inputs."""
    with pytest.raises(error, match=message):
        MissingnessSchema.from_dataframe(frame)


def test_schema_validate_frame_rejects_type_columns_index_and_dtype_drift(schema, original):
    """Serving data must retain the exact fit-time schema identity."""
    with pytest.raises(TypeError, match="candidate must be a pandas DataFrame"):
        schema.validate_frame([1, 2], name="candidate")
    with pytest.raises(ValueError, match="columns"):
        schema.validate_frame(original.rename(columns={"age": "years"}))
    with pytest.raises(ValueError, match="index"):
        schema.validate_frame(original.rename(index={10: 11}))
    with pytest.raises(ValueError, match="dtypes"):
        schema.validate_frame(original.astype({"age": "Int64"}))


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"methods": {}}, ValueError, "nonempty mapping"),
        ({"methods": {" ": "pmm"}}, TypeError, "nonempty strings"),
        ({"methods": {"age": ""}}, TypeError, "nonempty strings"),
        ({"n_imputations": True}, ValueError, "positive integer"),
        ({"max_iter": 0}, ValueError, "positive integer"),
        ({"seed_sequence": [101, 202]}, ValueError, "tuple"),
        ({"seed_sequence": (101, True)}, TypeError, "integers"),
        ({"constraints": []}, TypeError, "mapping"),
        ({"provenance": []}, TypeError, "mapping"),
    ],
)
def test_imputation_plan_rejects_nonreproducible_settings(kwargs, error, message):
    """Plans reject malformed methods, iteration counts, seeds, and metadata."""
    settings = {
        "methods": {"age": "pmm"},
        "n_imputations": 2,
        "max_iter": 3,
        "seed_sequence": (101, 202),
    }
    settings.update(kwargs)

    with pytest.raises(error, match=message):
        ImputationPlan(**settings)


def test_plan_rejects_wrong_schema_type_and_unknown_constraint(schema):
    """Plans reject a non-schema boundary and constraints without a target column."""
    plan = ImputationPlan({"age": "pmm"}, 1, 3, (101,), constraints={"missing": "x > 0"})

    with pytest.raises(TypeError, match="MissingnessSchema"):
        plan.validate_schema(pd.DataFrame())
    with pytest.raises(ValueError, match="unknown schema columns"):
        plan.validate_schema(schema)


def test_result_contract_rejects_invalid_trace_warning_and_validation_metadata(original, schema):
    """Result metadata must align with chains and remain serialisable diagnostic data."""
    plan = ImputationPlan({"age": "pmm"}, 1, 3, (101,))
    completed = pd.DataFrame({"age": [20.0, 30.0], "group": ["a", "b"]}, index=[10, 20])
    data = MIData(original, schema, plan, (completed,))

    with pytest.raises(ValueError, match="unknown schema columns"):
        ImputationResult(data, histories=({"missing": (1.0,)},))
    with pytest.raises(TypeError, match="finite numbers"):
        ImputationResult(data, histories=({"age": (float("nan"),)},))
    with pytest.raises(TypeError, match="tuple of strings"):
        ImputationResult(data, warnings=["warning"])
    with pytest.raises(TypeError, match="string names to booleans"):
        ImputationResult(data, validation={"valid": 1})
