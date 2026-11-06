"""Characterisation (golden) tests for float64 imputation outputs.

The numbers in ``impute_float64_golden.json`` were recorded from the
implementation before the internal dtype-normalisation layer was introduced.
Float64 inputs must keep producing bit-identical output, whatever happens to
the nullable / extension dtype handling around them.

Regenerate the file only for an intentional, reviewed numeric change::

    python -m tests.golden.test_impute_float64_golden
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict

import numpy as np
import pandas as pd
import pytest

from missingly.impute import (
    impute_gb,
    impute_knn,
    impute_mean,
    impute_median,
    impute_mice,
    impute_mode,
    impute_pmm,
    impute_rf,
)

GOLDEN_PATH = Path(__file__).with_name("impute_float64_golden.json")
_N_ROWS = 40
_SEED = 20261106


def build_frame() -> pd.DataFrame:
    """Return the deterministic float64 + object frame used by every golden case."""
    rng = np.random.default_rng(_SEED)
    base = rng.normal(size=(_N_ROWS, 3))
    frame = pd.DataFrame(
        {
            "a": base[:, 0] * 3.0 + 10.0,
            "b": base[:, 0] + base[:, 1],
            "c": base[:, 2] * 0.5,
        }
    )
    frame["g"] = np.where(frame["a"] > 10.0, "hi", "lo").astype(object)
    for column, step in (("a", 7), ("b", 5), ("c", 9), ("g", 6)):
        frame.loc[frame.index[step - 1 :: step], column] = np.nan
    return frame


def _knn(frame: pd.DataFrame) -> pd.DataFrame:
    return impute_knn(frame, n_neighbors=3)


def _knn_mixed(frame: pd.DataFrame) -> pd.DataFrame:
    return impute_knn(frame, n_neighbors=3, metric="mixed")


def _mice(frame: pd.DataFrame) -> pd.DataFrame:
    return impute_mice(frame, max_iter=3, random_state=1)


def _mice_history(frame: pd.DataFrame) -> pd.DataFrame:
    result, _ = impute_mice(frame, max_iter=3, random_state=1, return_history=True)
    return result


def _pmm(frame: pd.DataFrame) -> pd.DataFrame:
    return impute_pmm(frame, max_iter=2, random_state=2)


def _rf(frame: pd.DataFrame) -> pd.DataFrame:
    return impute_rf(frame, random_state=3, n_estimators=8)


def _gb(frame: pd.DataFrame) -> pd.DataFrame:
    return impute_gb(frame, random_state=4, n_estimators=8)


CASES: Dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
    "mean": impute_mean,
    "median": impute_median,
    "mode": impute_mode,
    "knn": _knn,
    "knn_mixed": _knn_mixed,
    "mice": _mice,
    "mice_history": _mice_history,
    "pmm": _pmm,
    "rf": _rf,
    "gb": _gb,
}


def _encode(frame: pd.DataFrame) -> dict:
    payload: dict = {"columns": list(frame.columns), "dtypes": [str(d) for d in frame.dtypes]}
    values = {}
    for column in frame.columns:
        series = frame[column]
        if pd.api.types.is_float_dtype(series):
            values[column] = [float(v).hex() for v in series.to_numpy(dtype=float)]
        else:
            values[column] = series.tolist()
    payload["values"] = values
    return payload


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
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", sorted(CASES))
def test_float64_output_is_bit_identical_to_recorded_golden(name: str, golden: dict) -> None:
    result = CASES[name](build_frame())
    expected = _decode(golden[name])
    pd.testing.assert_frame_equal(
        result.reset_index(drop=True),
        expected,
        check_exact=True,
        check_dtype=False,
    )
    assert [str(d) for d in result.dtypes] == golden[name]["dtypes"]


def test_golden_file_covers_every_case(golden: dict) -> None:
    assert sorted(golden) == sorted(CASES)


def _regenerate() -> None:
    payload = {name: _encode(func(build_frame())) for name, func in CASES.items()}
    GOLDEN_PATH.write_text(json.dumps(payload, indent=1), encoding="utf-8")


if __name__ == "__main__":
    _regenerate()
