"""Capture the float64 golden outputs of the current environment.

Run this script with the **pre-change** imputers (for example from a checkout
of ``origin/fix/pyproject-regression`` with this file copied in) to record the
golden values of one software stack::

    python tests/golden/capture_golden.py out.json

The output is ``{stack_key: {case: payload}}``; merge several files into
``impute_float64_golden.json``.  The script only depends on numpy, pandas and
``missingly.impute``, so it runs unchanged on the base code.
"""

from __future__ import annotations

import json
import sys
from typing import Callable, Dict

import numpy as np
import pandas as pd
import sklearn

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

_N_ROWS = 40
_SEED = 20261106


def stack_key() -> str:
    """Return the software-stack signature the golden values depend on."""
    numpy_mm = ".".join(np.__version__.split(".")[:2])
    sklearn_mm = ".".join(sklearn.__version__.split(".")[:2])
    pandas_major = pd.__version__.split(".")[0]
    return f"numpy{numpy_mm}-pandas{pandas_major}-sklearn{sklearn_mm}"


def dtype_label(dtype: object) -> str:
    """Name a dtype; every string-like dtype (object, str, string) is ``text``."""
    if pd.api.types.is_float_dtype(dtype):
        return str(dtype)
    return "text"


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


def encode(frame: pd.DataFrame) -> dict:
    """Serialise a result frame with exact (hex) floats."""
    payload: dict = {
        "columns": list(frame.columns),
        "dtypes": [dtype_label(d) for d in frame.dtypes],
    }
    values = {}
    for column in frame.columns:
        series = frame[column]
        if pd.api.types.is_float_dtype(series):
            values[column] = [float(v).hex() for v in series.to_numpy(dtype=float)]
        else:
            values[column] = series.tolist()
    payload["values"] = values
    return payload


def capture() -> Dict[str, Dict[str, dict]]:
    """Run every case in this environment and return the keyed payload."""
    return {
        stack_key(): {name: encode(func(build_frame())) for name, func in CASES.items()}
    }


if __name__ == "__main__":
    with open(sys.argv[1], "w", encoding="utf-8") as handle:
        json.dump(capture(), handle, indent=1)
