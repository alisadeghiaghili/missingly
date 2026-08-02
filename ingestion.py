"""Safe, in-memory tabular import helpers for analytics endpoints."""

from __future__ import annotations

import io
import json
from pathlib import PurePath
from typing import Any

import pandas as pd
import pyreadstat

from analytics import AnalyticsError, _frame_to_records, profile_dataset


SUPPORTED_FORMATS = {".csv", ".json", ".xlsx", ".sav", ".dta", ".xpt", ".sas7bdat"}


class IngestionError(ValueError):
    """Raised when an upload cannot be safely interpreted as a supported table."""


def import_tabular_bytes(filename: str, content: bytes) -> dict[str, Any]:
    suffix = PurePath(filename or "").suffix.lower()
    if suffix not in SUPPORTED_FORMATS:
        formats = ", ".join(sorted(SUPPORTED_FORMATS))
        raise IngestionError(f"Unsupported file format. Use one of: {formats}")
    if not content:
        raise IngestionError("Uploaded file is empty")
    buffer = io.BytesIO(content)
    try:
        if suffix == ".csv":
            try:
                frame = pd.read_csv(buffer)
            except UnicodeDecodeError:
                buffer.seek(0)
                frame = pd.read_csv(buffer, encoding="utf-8-sig")
        elif suffix == ".json":
            decoded = json.loads(content.decode("utf-8-sig"))
            if not isinstance(decoded, list) or not all(isinstance(record, dict) for record in decoded):
                raise IngestionError("JSON uploads must be an array of record objects")
            frame = pd.DataFrame(decoded)
        elif suffix == ".xlsx":
            frame = pd.read_excel(buffer, engine="openpyxl")
        elif suffix == ".sav":
            frame, _ = pyreadstat.read_sav(buffer, apply_value_formats=False)
        elif suffix == ".dta":
            frame = pd.read_stata(buffer, convert_categoricals=False)
        elif suffix == ".xpt":
            frame, _ = pyreadstat.read_xport(buffer)
        else:
            frame, _ = pyreadstat.read_sas7bdat(buffer)
    except IngestionError:
        raise
    except Exception as exc:
        raise IngestionError(f"Could not parse the uploaded {suffix} file") from exc
    if frame.empty:
        raise IngestionError("The uploaded table has no rows")
    if frame.columns.duplicated().any():
        raise IngestionError("The uploaded table contains duplicate column names")
    records = _frame_to_records(frame)
    try:
        profile = profile_dataset(records)
    except AnalyticsError as exc:
        raise IngestionError(str(exc)) from exc
    return {
        "format": suffix.removeprefix("."),
        "rows": profile["dataset"]["rows"],
        "columns": [column["name"] for column in profile["columns"]],
        "records": records,
    }
