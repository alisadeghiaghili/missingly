"""Regression checks for public API documentation contracts."""

from __future__ import annotations

import inspect

import analytics
import ingestion
import reporting


PUBLIC_FUNCTIONS = (
    *(getattr(analytics, name) for name in analytics.__all__ if inspect.isfunction(getattr(analytics, name))),
    *(getattr(ingestion, name) for name in ingestion.__all__ if inspect.isfunction(getattr(ingestion, name))),
    *(getattr(reporting, name) for name in reporting.__all__ if inspect.isfunction(getattr(reporting, name))),
)

REQUIRED_SECTIONS = ("Args:", "Returns:", "Raises:", "Examples:")


def test_public_functions_keep_complete_docstring_contracts():
    """Require every supported callable to document inputs, outputs, failures, and use."""
    undocumented = {
        function.__name__: [
            section for section in REQUIRED_SECTIONS if section not in (inspect.getdoc(function) or "")
        ]
        for function in PUBLIC_FUNCTIONS
    }
    failures = {name: sections for name, sections in undocumented.items() if sections}
    assert not failures, f"Public API documentation is incomplete: {failures}"
