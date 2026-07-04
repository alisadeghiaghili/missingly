"""Unit tests for missingly._validation."""

import numpy as np
import pandas as pd
import pytest

from missingly._validation import (
    validate_columns,
    validate_dataframe,
    validate_positive_int,
    validate_strategy,
)
from missingly.exceptions import InvalidStrategyError, MissingColumnError


# ---------------------------------------------------------------------------
# validate_dataframe
# ---------------------------------------------------------------------------

class TestValidateDataframe:
    def test_valid_dataframe_passes(self):
        validate_dataframe(pd.DataFrame({"a": [1, 2]}))

    def test_series_raises_type_error(self):
        with pytest.raises(TypeError, match="pandas DataFrame"):
            validate_dataframe(pd.Series([1, 2]))

    def test_list_raises_type_error(self):
        with pytest.raises(TypeError, match="pandas DataFrame"):
            validate_dataframe([1, 2, 3])

    def test_dict_raises_type_error(self):
        with pytest.raises(TypeError, match="pandas DataFrame"):
            validate_dataframe({"a": [1]})

    def test_none_raises_type_error(self):
        with pytest.raises(TypeError, match="pandas DataFrame"):
            validate_dataframe(None)

    def test_numpy_array_raises_type_error(self):
        with pytest.raises(TypeError):
            validate_dataframe(np.array([[1, 2]]))

    def test_empty_dataframe_raises_value_error_by_default(self):
        with pytest.raises(ValueError, match="empty"):
            validate_dataframe(pd.DataFrame())

    def test_empty_dataframe_allowed_when_flag_set(self):
        validate_dataframe(pd.DataFrame(), allow_empty=True)  # must not raise

    def test_min_rows_respected(self):
        with pytest.raises(ValueError, match="3 row"):
            validate_dataframe(pd.DataFrame({"a": [1, 2]}), min_rows=3)

    def test_min_rows_exactly_met_passes(self):
        validate_dataframe(pd.DataFrame({"a": [1, 2, 3]}), min_rows=3)

    def test_custom_param_name_in_error(self):
        with pytest.raises(TypeError, match="`my_frame`"):
            validate_dataframe("not a df", param="my_frame")


# ---------------------------------------------------------------------------
# validate_columns
# ---------------------------------------------------------------------------

class TestValidateColumns:
    def setup_method(self):
        self.df = pd.DataFrame({"age": [1.0, 2.0], "city": ["A", "B"]})

    def test_existing_column_passes(self):
        validate_columns(self.df, ["age"])

    def test_missing_column_raises(self):
        with pytest.raises(MissingColumnError, match="salary"):
            validate_columns(self.df, ["salary"])

    def test_multiple_missing_columns_listed_in_error(self):
        with pytest.raises(MissingColumnError) as exc_info:
            validate_columns(self.df, ["x", "y"])
        msg = str(exc_info.value)
        assert "x" in msg
        assert "y" in msg

    def test_require_numeric_passes_for_numeric_col(self):
        validate_columns(self.df, ["age"], require_numeric=True)

    def test_require_numeric_raises_for_string_col(self):
        with pytest.raises(TypeError, match="city"):
            validate_columns(self.df, ["city"], require_numeric=True)

    def test_empty_columns_list_passes(self):
        validate_columns(self.df, [])


# ---------------------------------------------------------------------------
# validate_strategy
# ---------------------------------------------------------------------------

class TestValidateStrategy:
    def test_valid_strategy_passes(self):
        validate_strategy("mean", ["mean", "median"])

    def test_invalid_strategy_raises(self):
        with pytest.raises(InvalidStrategyError, match="avg"):
            validate_strategy("avg", ["mean", "median"])

    def test_example_shown_in_error_message(self):
        with pytest.raises(InvalidStrategyError, match="mean"):
            validate_strategy("avg", ["mean"], example="mean")

    def test_case_sensitive(self):
        with pytest.raises(InvalidStrategyError):
            validate_strategy("Mean", ["mean"])


# ---------------------------------------------------------------------------
# validate_positive_int
# ---------------------------------------------------------------------------

class TestValidatePositiveInt:
    def test_positive_int_passes(self):
        validate_positive_int(5, param="n")

    def test_zero_raises_by_default(self):
        with pytest.raises(ValueError, match="positive"):
            validate_positive_int(0, param="n")

    def test_zero_allowed_when_flag_set(self):
        validate_positive_int(0, param="n", allow_zero=True)

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            validate_positive_int(-1, param="n")

    def test_float_raises_type_error(self):
        with pytest.raises(TypeError, match="integer"):
            validate_positive_int(1.5, param="n")

    def test_string_raises_type_error(self):
        with pytest.raises(TypeError):
            validate_positive_int("5", param="n")

    def test_numpy_integer_passes(self):
        validate_positive_int(np.int64(3), param="n")

    def test_param_name_in_error(self):
        with pytest.raises(ValueError, match="`my_param`"):
            validate_positive_int(-1, param="my_param")


# ---------------------------------------------------------------------------
# validate_dataframe — param name propagation
# ---------------------------------------------------------------------------

def test_param_name_propagates_in_type_error():
    with pytest.raises(TypeError, match="`input_df`"):
        validate_dataframe(42, param="input_df")
