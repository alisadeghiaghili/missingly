"""Generate scalar Missingly evidence for the pinned numeric FCS fixture."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

from missingly import impute_mice
from missingly._version import __version__


def main(input_path: Path, output_path: Path) -> None:
    """Write pooled completed-data means without serializing input rows.

    Parameters
    ----------
    input_path : pathlib.Path
        Public synthetic CSV fixture used by the pinned R reference.
    output_path : pathlib.Path
        Destination for scalar-only JSON evidence.

    Returns
    -------
    None
        Writes an auditable JSON artifact containing no source or completed rows.

    Examples
    --------
    >>> isinstance(Path("numeric_fcs.csv"), Path)
    True
    """
    frame = pd.read_csv(input_path)
    completed = impute_mice(frame, n_imputations=3, max_iter=5, random_state=20260811)
    if not isinstance(completed, list):
        raise TypeError("expected impute_mice to return three completed DataFrames")
    pooled = pd.concat(completed, ignore_index=True).mean(numeric_only=True)
    artifact = {
        "subject_tool": "Missingly",
        "subject_tool_version": __version__,
        "method": "missingly.impute_mice(n_imputations=3, max_iter=5)",
        "seed": 20260811,
        "dataset_sha256": "9469f77a9c2ae5c63f22e16e5e8a974070c65af4c9d84adb87759aacafc9bd1a",
        "metrics": {f"pooled_mean_{name}": float(value) for name, value in pooled.items()},
    }
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("Usage: generate_missingly_reference.py <fixture.csv> <output.json>")
    main(Path(sys.argv[1]), Path(sys.argv[2]))
