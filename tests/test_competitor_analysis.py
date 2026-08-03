"""Contracts that keep the competitor analysis evidence-based and actionable."""

from __future__ import annotations

import re
from pathlib import Path


ANALYSIS_PATH = Path(__file__).resolve().parents[1] / "COMPETITOR_ANALYSIS.md"


def test_competitor_analysis_covers_required_ecosystems_and_delivery_gates():
    content = ANALYSIS_PATH.read_text(encoding="utf-8")
    required_terms = {
        "mice",
        "lme4",
        "glmmTMB",
        "geepack",
        "statsmodels",
        "scikit-learn",
        "lifelines",
        "SAS/STAT",
        "IBM SPSS Statistics",
        "Stata",
        "Capability matrix",
        "TDD roadmap",
        "Cross-tool benchmark manifest",
        "Review policy",
    }
    missing = sorted(term for term in required_terms if term not in content)
    assert not missing, f"Competitor analysis is missing required coverage: {missing}"


def test_competitor_analysis_sources_are_https_and_not_search_pages():
    content = ANALYSIS_PATH.read_text(encoding="utf-8")
    source_section = content.split("## Source registry", maxsplit=1)[1].split("## Review policy", maxsplit=1)[0]
    urls = re.findall(r"https://[^\s)]+", source_section)
    assert len(urls) >= 15
    assert len(urls) == len(set(urls))
    assert all("google.com/search" not in url and "bing.com/search" not in url for url in urls)


def test_competitor_analysis_does_not_claim_unverified_global_superiority():
    content = ANALYSIS_PATH.read_text(encoding="utf-8").lower()
    prohibited_claims = (
        "missingly is better than all",
        "missingly is superior to all",
        "missingly has full parity",
    )
    assert not any(claim in content for claim in prohibited_claims)
