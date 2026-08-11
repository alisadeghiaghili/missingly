"""Public API contracts for package-wide Missingly configuration."""

from __future__ import annotations

import missingly
from missingly.config import MissinglyConfig


def test_missingly_config_is_publicly_exported() -> None:
    """Users can construct the documented configuration type from the package."""
    assert missingly.MissinglyConfig is MissinglyConfig


def test_public_config_uses_the_exported_configuration_type() -> None:
    """The singleton configuration remains an instance of its public type."""
    assert isinstance(missingly.config, missingly.MissinglyConfig)
