"""Deterministic provenance utilities for missing-data analyses.

The functions in this module create content-derived identifiers without
persisting the caller's raw records.  They deliberately include pandas dtype,
index, and categorical metadata because those properties can change the
meaning or result of an imputation workflow even when printed values look the
same.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date, datetime, time
from decimal import Decimal
import hashlib
import json
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from missingly._version import __version__

__all__ = ["analysis_provenance", "dataframe_fingerprint"]

_SCHEMA_VERSION = 1


def _canonical_json(value: Any) -> Any:
    """Convert a supported value to an unambiguous JSON-compatible form."""
    if value is pd.NA or value is pd.NaT:
        return {"__missing__": True}
    if isinstance(value, np.generic):
        return _canonical_json(value.item())
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if np.isnan(value):
            return {"__float__": "nan"}
        if np.isposinf(value):
            return {"__float__": "+inf"}
        if np.isneginf(value):
            return {"__float__": "-inf"}
        return value
    if isinstance(value, Decimal):
        return {"__decimal__": str(value)}
    if isinstance(value, pd.Timestamp):
        return {"__timestamp__": value.isoformat()}
    if isinstance(value, pd.Timedelta):
        return {"__timedelta__": value.isoformat()}
    if isinstance(value, datetime):
        return {"__datetime__": value.isoformat()}
    if isinstance(value, date):
        return {"__date__": value.isoformat()}
    if isinstance(value, time):
        return {"__time__": value.isoformat()}
    if isinstance(value, bytes):
        return {"__bytes_hex__": value.hex()}
    if isinstance(value, pd.Interval):
        return {
            "__interval__": {
                "left": _canonical_json(value.left),
                "right": _canonical_json(value.right),
                "closed": value.closed,
            }
        }
    if isinstance(value, Mapping):
        if all(isinstance(key, str) for key in value):
            return {
                key: _canonical_json(value[key])
                for key in sorted(value)
            }
        pairs = [
            [_canonical_json(key), _canonical_json(item)]
            for key, item in value.items()
        ]
        pairs.sort(key=_stable_json)
        return {"__mapping__": pairs}
    if isinstance(value, tuple):
        return {"__tuple__": [_canonical_json(item) for item in value]}
    if isinstance(value, list):
        return [_canonical_json(item) for item in value]
    if isinstance(value, (set, frozenset)):
        items = [_canonical_json(item) for item in value]
        items.sort(key=_stable_json)
        return {"__set__": items}

    try:
        is_missing = pd.isna(value)
    except (TypeError, ValueError):
        is_missing = False
    if isinstance(is_missing, (bool, np.bool_)) and bool(is_missing):
        return {"__missing__": True}

    raise TypeError(
        "Unsupported provenance value of type "
        f"{type(value).__module__}.{type(value).__qualname__}. "
        "Convert it to a stable scalar or container before hashing."
    )


def _stable_json(value: Any) -> str:
    """Serialize a canonical value using stable JSON settings."""
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _dtype_metadata(series: pd.Series) -> Dict[str, Any]:
    """Return deterministic metadata for a pandas Series dtype."""
    dtype = series.dtype
    metadata: Dict[str, Any] = {"dtype": str(dtype)}
    if isinstance(dtype, pd.CategoricalDtype):
        metadata["categories"] = [
            _canonical_json(item) for item in dtype.categories.tolist()
        ]
        metadata["ordered"] = bool(dtype.ordered)
    return metadata


def _validate_inputs(
    df: pd.DataFrame,
    operation: Optional[str],
    parameters: Optional[Mapping[str, Any]],
) -> None:
    """Validate the public provenance contract before serialization."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if operation is not None and not isinstance(operation, str):
        raise TypeError("operation must be a string or None")
    if parameters is not None and not isinstance(parameters, Mapping):
        raise TypeError("parameters must be a mapping or None")


def dataframe_fingerprint(
    df: pd.DataFrame,
    *,
    operation: Optional[str] = None,
    parameters: Optional[Mapping[str, Any]] = None,
) -> str:
    """Return a deterministic SHA-256 identity for a DataFrame and context.

    The fingerprint covers column order and labels, dtype metadata, index
    structure and values, cell values, ``DataFrame.attrs``, and—when
    supplied—the analytical operation and its parameters.  The function does
    not modify ``df`` and does not write raw data anywhere.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data whose exact analytical identity should be recorded.
    operation : str, optional
        Stable operation name, for example ``"impute_mice"``.  Include it
        when the identifier should distinguish different workflows on the
        same input.
    parameters : mapping of str to Any, optional
        Operation settings such as seeds, method names, and iteration counts.
        Values must consist of deterministic scalar or container types.

    Returns
    -------
    str
        A lowercase, 64-character hexadecimal SHA-256 digest.

    Raises
    ------
    TypeError
        If ``df`` is not a DataFrame, the context types are invalid, or a
        value cannot be represented deterministically.

    Examples
    --------
    >>> import pandas as pd
    >>> from missingly.provenance import dataframe_fingerprint
    >>> frame = pd.DataFrame({"age": [20.0, None, 40.0]})
    >>> digest = dataframe_fingerprint(
    ...     frame,
    ...     operation="impute_simple",
    ...     parameters={"strategy": "mean"},
    ... )
    >>> len(digest)
    64
    >>> digest == dataframe_fingerprint(
    ...     frame,
    ...     operation="impute_simple",
    ...     parameters={"strategy": "mean"},
    ... )
    True
    """
    _validate_inputs(df, operation, parameters)

    payload = {
        "schema_version": _SCHEMA_VERSION,
        "columns": [_canonical_json(label) for label in df.columns.tolist()],
        "column_names": [_canonical_json(name) for name in df.columns.names],
        "dtypes": [
            _dtype_metadata(df.iloc[:, position])
            for position in range(df.shape[1])
        ],
        "index": {
            "class": type(df.index).__name__,
            "names": [_canonical_json(name) for name in df.index.names],
            "values": [_canonical_json(value) for value in df.index.tolist()],
        },
        "data": [
            [_canonical_json(value) for value in row]
            for row in df.itertuples(index=False, name=None)
        ],
        "attrs": _canonical_json(df.attrs),
        "operation": operation,
        "parameters": _canonical_json(parameters or {}),
    }
    serialized = _stable_json(payload).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def analysis_provenance(
    df: pd.DataFrame,
    operation: str,
    parameters: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a replay-oriented provenance manifest for an analysis.

    Parameters
    ----------
    df : pandas.DataFrame
        Exact input data supplied to the analytical operation.
    operation : str
        Stable public operation name, such as ``"impute_mice"``.
    parameters : mapping of str to Any, optional
        Deterministic operation settings, including any random seed.

    Returns
    -------
    dict
        JSON-compatible manifest containing schema and package versions, the
        normalized settings, an input-only digest, and a configured-analysis
        digest.

    Raises
    ------
    TypeError
        If the input or context violates the deterministic serialization
        contract.
    ValueError
        If ``operation`` is empty or contains only whitespace.

    Examples
    --------
    >>> import pandas as pd
    >>> from missingly.provenance import analysis_provenance
    >>> frame = pd.DataFrame({"age": [20.0, None, 40.0]})
    >>> manifest = analysis_provenance(
    ...     frame,
    ...     "impute_mice",
    ...     {"m": 5, "random_state": 42},
    ... )
    >>> manifest["package"]
    'missingly'
    >>> len(manifest["analysis_sha256"])
    64
    """
    _validate_inputs(df, operation, parameters)
    if not operation.strip():
        raise ValueError("operation must not be empty")

    normalized_parameters = _canonical_json(parameters or {})
    return {
        "schema_version": _SCHEMA_VERSION,
        "package": "missingly",
        "package_version": __version__,
        "operation": operation,
        "parameters": normalized_parameters,
        "input_sha256": dataframe_fingerprint(df),
        "analysis_sha256": dataframe_fingerprint(
            df,
            operation=operation,
            parameters=parameters,
        ),
    }
