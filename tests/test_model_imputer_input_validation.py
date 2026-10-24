"""Model-based imputers must reject invalid input before fitting.

``inf`` / ``-inf`` and unsupported dtypes (complex, timedelta) anywhere in the
frame would make every conditional fit fail mid-way, so they raise a
``ValueError`` that names the columns, whatever ``strict_mode`` says.
"""

from __future__ import annotations

from typing import Callable, Dict

import numpy as np
import pandas as pd
import pytest

from missingly.impute import (
    impute_gb,
    impute_logreg,
    impute_polr,
    impute_polyreg,
    impute_rf,
)

Imputer = Callable[..., pd.DataFrame]

ALL_IMPUTERS: Dict[str, Imputer] = {
    "rf": impute_rf,
    "gb": impute_gb,
    "logreg": impute_logreg,
    "polyreg": impute_polyreg,
    "polr": impute_polr,
}


def _frame_for(name: str) -> pd.DataFrame:
    """Return a frame with a target column suited to the named imputer."""
    rng = np.random.default_rng(3)
    n = 90
    x = rng.normal(size=n)
    frame = pd.DataFrame({"x": x, "w": x + rng.normal(scale=0.3, size=n)})
    levels = {
        "logreg": ["u", "v"],
        "polyreg": ["red", "green", "blue"],
        "polr": ["low", "mid", "high"],
    }.get(name)
    if levels is None:
        frame["y"] = 2.0 * x + rng.normal(scale=0.1, size=n)
        frame.loc[1::6, "y"] = np.nan
    else:
        frame["label"] = rng.choice(levels, size=n)
        frame.loc[1::6, "label"] = np.nan
    return frame


def _target(name: str) -> str:
    return "label" if name in {"logreg", "polyreg", "polr"} else "y"



# ---------------------------------------------------------------------------
# Up-front validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(ALL_IMPUTERS))
@pytest.mark.parametrize("bad", [np.inf, -np.inf], ids=["+inf", "-inf"])
def test_infinite_predictor_raises_before_any_fit(name: str, bad: float) -> None:
    frame = _frame_for(name)
    frame.loc[5, "x"] = bad
    with pytest.raises(ValueError, match=r"infinite.*\['x'\]"):
        ALL_IMPUTERS[name](frame, random_state=0)


@pytest.mark.parametrize("name", ["rf", "gb"])
def test_infinite_target_value_raises_and_names_the_column(name: str) -> None:
    frame = _frame_for(name)
    frame.loc[7, "y"] = np.inf
    with pytest.raises(ValueError, match=r"infinite.*\['y'\]"):
        ALL_IMPUTERS[name](frame, random_state=0)


@pytest.mark.parametrize("name", sorted(ALL_IMPUTERS))
def test_infinite_values_are_rejected_even_with_strict_mode_off(name: str) -> None:
    frame = _frame_for(name)
    frame.loc[5, "w"] = np.inf
    with pytest.raises(ValueError, match="w"):
        ALL_IMPUTERS[name](frame, random_state=0, strict_mode=False)


@pytest.mark.parametrize("name", sorted(ALL_IMPUTERS))
@pytest.mark.parametrize("kind", ["timedelta", "complex"])
def test_unsupported_dtype_raises_and_names_the_column(name: str, kind: str) -> None:
    frame = _frame_for(name)
    n = len(frame)
    if kind == "timedelta":
        frame["bad"] = pd.to_timedelta(np.arange(n), unit="D")
    else:
        frame["bad"] = np.arange(n) * 1j
    with pytest.raises(ValueError, match=r"unsupported dtype.*\['bad'\]"):
        ALL_IMPUTERS[name](frame, random_state=0)
