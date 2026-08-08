"""Documentation contracts for honest, evidence-based competitor claims."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def analysis_text() -> str:
    """Load the competitor analysis as normalized lowercase text."""
    path = Path(__file__).parents[1] / "COMPETITOR_ANALYSIS.md"
    return path.read_text(encoding="utf-8").lower()


@pytest.mark.parametrize(
    "required_competitor",
    [
        "mice",
        "naniar",
        "vim",
        "missforest",
        "missingno",
        "scikit-learn",
        "statsmodels",
        "sas",
        "spss",
        "stata",
        "jasp",
        "jamovi",
    ],
)
def test_analysis_covers_reference_ecosystems(analysis_text, required_competitor):
    """The roadmap must cover open-source, commercial, and GUI references."""
    assert required_competitor in analysis_text


@pytest.mark.parametrize(
    "required_gate",
    ["sha-256", "tolerance", "tool/package version", "parity verified", "better verified"],
)
def test_superiority_requires_reproducible_evidence(analysis_text, required_gate):
    """Claims must remain tied to auditable evidence rather than adjectives."""
    assert required_gate in analysis_text


def test_analysis_rejects_global_superiority_claim(analysis_text):
    """The current document must explicitly reject an unsupported global claim."""
    assert "no evidence yet supports the global claim" in analysis_text
    assert "missingly is better than all" not in analysis_text

