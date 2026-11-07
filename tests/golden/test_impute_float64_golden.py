"""Characterisation (golden) tests for float64 imputation outputs.

``impute_float64_golden.json`` holds, per software stack, the output of the
imputers recorded **before** the dtype-normalisation layer existed (captured
with ``capture_golden.py`` from ``origin/fix/pyproject-regression``).  Float64
inputs must keep producing the same output whatever happens to the nullable /
extension dtype handling around them.

The golden values are keyed per stack (numpy major.minor, pandas major,
scikit-learn major.minor) because chained-equation, forest and boosting
output legitimately depends on those library versions, and a string column is
``object`` on pandas 2 but ``str`` on pandas 3 (string-like dtypes compare as
``text``).  Within one stack the comparison allows ``1e-12`` relative and
absolute difference, which only absorbs platform maths-library rounding.

The ``mice`` and ``mice_history`` entries were re-recorded for every stack
when the MICE chain seeds became independent (``SeedSequence.spawn``, WP-05,
F-246); every other entry is still the pre-change recording.

A stack without recorded values fails the test; it is never skipped.  Record
it with the pre-change code::

    python tests/golden/capture_golden.py out.json

and merge ``out.json`` into the golden file.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.golden.capture_golden import (
    CASES,
    build_frame,
    dtype_label,
    stack_key,
)

GOLDEN_PATH = Path(__file__).with_name("impute_float64_golden.json")
GOLDEN_TOLERANCE = 1e-12


def _decode(payload: dict) -> pd.DataFrame:
    data = {}
    for column, dtype in zip(payload["columns"], payload["dtypes"]):
        raw = payload["values"][column]
        if dtype.startswith("float"):
            data[column] = np.array([float.fromhex(v) for v in raw], dtype=dtype)
        else:
            data[column] = pd.Series(raw, dtype=object)
    return pd.DataFrame(data, columns=payload["columns"])


@pytest.fixture(scope="module")
def golden() -> dict:
    recorded = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    key = stack_key()
    if key not in recorded:
        pytest.fail(
            f"No golden values recorded for the stack {key!r} (recorded: "
            f"{sorted(recorded)}). Generate them with the pre-change code: "
            "python tests/golden/capture_golden.py out.json, then merge out.json "
            "into impute_float64_golden.json."
        )
    return recorded[key]


@pytest.mark.parametrize("name", sorted(CASES))
def test_float64_output_matches_recorded_golden(name: str, golden: dict) -> None:
    result = CASES[name](build_frame())
    pd.testing.assert_frame_equal(
        result.reset_index(drop=True),
        _decode(golden[name]),
        check_exact=False,
        rtol=GOLDEN_TOLERANCE,
        atol=GOLDEN_TOLERANCE,
        check_dtype=False,
    )
    assert [dtype_label(d) for d in result.dtypes] == golden[name]["dtypes"]


def test_golden_file_covers_every_case(golden: dict) -> None:
    assert sorted(golden) == sorted(CASES)
