"""Public API contracts for package-wide Missingly configuration."""

from __future__ import annotations

import missingly
import pandas as pd
import pytest
from missingly.config import MissinglyConfig


def test_missingly_config_is_publicly_exported() -> None:
    """Users can construct the documented configuration type from the package."""
    assert missingly.MissinglyConfig is MissinglyConfig


def test_public_config_uses_the_exported_configuration_type() -> None:
    """The singleton configuration remains an instance of its public type."""
    assert isinstance(missingly.config, missingly.MissinglyConfig)


def test_public_config_mutation_changes_imputation_runtime_policy(monkeypatch) -> None:
    """Mutating ``missingly.config`` changes the warning policy used at runtime."""
    frame = pd.DataFrame({"score": [1.0, None, 3.0]})
    monkeypatch.setattr(missingly.config, "large_df_threshold", 1)

    with pytest.warns(UserWarning, match="DataFrame has 3 rows"):
        result = frame.miss.impute("knn", n_neighbors=1)

    assert result.loc[1, "score"] == 2.0
