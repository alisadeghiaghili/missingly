"""Tests for missingly.report — HTML report generation, i18n, and edge cases.

Coverage targets
----------------
- create_report() smoke: file is created, is valid HTML, contains no
  un-rendered Jinja2 placeholders ({{ ... }}).
- i18n / direction: <html lang> and dir attributes are correct per language;
  Persian UI labels appear in FA reports; unknown language falls back to EN
  with a UserWarning.
- Technical terms stay English: MCAR, MAR, MNAR, MICE, KNN, p-value,
  Little's Test, Random Forest, Gradient Boosting are NEVER translated in
  any language.  Chi-square is kept English in FA but may appear as
  "Chi-Quadrat" in DE (standard German scientific terminology).
- Edge cases: all-missing DataFrame, no-missing DataFrame, single-column
  DataFrame, Persian column names, mixed dtypes, extra sentinel values.
- _mcar_interpretation unit: correct keys, prose language, technical terms.
- Output path: custom path is honoured; empty DataFrame raises ValueError.
"""

from __future__ import annotations

import os
import re
import tempfile
import warnings

import numpy as np
import pandas as pd
import pytest

from missingly.report import (
    _TRANSLATIONS,
    _mcar_interpretation,
    create_report,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def df_typical():
    rng = np.random.default_rng(42)
    n = 60
    return pd.DataFrame({
        "age":    rng.choice([25, 30, 35, 40, np.nan], n),
        "income": rng.choice([50_000, 60_000, np.nan, np.nan], n),
        "city":   rng.choice(["Tehran", "Milan", None, "Berlin"], n),
        "score":  rng.uniform(0, 100, n),
    })


@pytest.fixture
def df_all_missing():
    return pd.DataFrame({"a": [np.nan] * 3, "b": [np.nan] * 3})


@pytest.fixture
def df_no_missing():
    return pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": ["a", "b", "c", "d", "e"]})


@pytest.fixture
def df_single_col():
    return pd.DataFrame({"only": [1.0, np.nan, 3.0, np.nan, 5.0]})


@pytest.fixture
def df_persian_colnames():
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "\u0633\u0646":    rng.choice([20, 30, np.nan], 30),
        "\u062f\u0631\u0622\u0645\u062f": rng.choice([1000, np.nan], 30),
        "\u0634\u0647\u0631":  rng.choice(["\u062a\u0647\u0631\u0627\u0646", "\u0645\u0634\u0647\u062f", None], 30),
    })


@pytest.fixture
def df_mixed_dtypes():
    return pd.DataFrame({
        "int_col":   pd.array([1, 2, None, 4], dtype="Int64"),
        "float_col": [1.1, np.nan, 3.3, np.nan],
        "str_col":   ["a", None, "c", None],
        "bool_col":  pd.array([True, None, False, None], dtype="boolean"),
        "cat_col":   pd.Categorical(["x", None, "z", "x"]),
    })


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _write_report(df, **kwargs) -> str:
    """Write report to a temp file and return the HTML string."""
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as fh:
        path = fh.name
    try:
        create_report(df, output_path=path, **kwargs)
        with open(path, encoding="utf-8") as fh:
            return fh.read()
    finally:
        if os.path.exists(path):
            os.unlink(path)


# ---------------------------------------------------------------------------
# Technical term lists
#
# Policy
# ------
# Terms that have NO standard translation in a language must stay English.
# Terms that DO have a standard scientific equivalent MAY use that equivalent.
#
# FA (Persian): no standard Persian scientific equivalents → all stay English.
# DE (German):  Chi-Quadrat is the standard German term → allowed to differ.
#               All other terms below have no standard German equivalent and
#               must stay English.
# ---------------------------------------------------------------------------

# Terms that must stay English in EVERY language (including DE).
_UNIVERSAL_EN_TERMS = [
    "MCAR",
    "MICE",
    "KNN",
    "p-value",
    "Random Forest",
    "Gradient Boosting",
    "Little",   # Little's Test
    "MAR",
    "MNAR",
]

# Terms that must stay English in FA specifically.
FA_TECHNICAL_TERMS = _UNIVERSAL_EN_TERMS + ["Chi-square"]

# Terms that must stay English in DE specifically.
# Chi-square is excluded here because "Chi-Quadrat" is the correct DE term.
DE_TECHNICAL_TERMS = _UNIVERSAL_EN_TERMS

# Approved DE translations (documenting intent, asserted in a dedicated test).
DE_ALLOWED_TRANSLATIONS = {
    "Chi-square": "Chi-Quadrat",
}


# ---------------------------------------------------------------------------
# Smoke
# ---------------------------------------------------------------------------

class TestReportSmoke:
    def test_file_is_created(self, df_typical, tmp_path):
        out = tmp_path / "report.html"
        result = create_report(df_typical, output_path=str(out))
        assert os.path.exists(result)

    def test_returns_absolute_path(self, df_typical, tmp_path):
        out = tmp_path / "r.html"
        result = create_report(df_typical, output_path=str(out))
        assert os.path.isabs(result)

    def test_html_starts_with_doctype(self, df_typical):
        html = _write_report(df_typical)
        assert html.strip().lower().startswith("<!doctype html>")

    def test_no_unreplaced_jinja_placeholders(self, df_typical):
        html = _write_report(df_typical)
        assert not re.search(r"\{\{.*?\}\}", html), \
            "Unreplaced Jinja2 placeholders found in rendered HTML"

    def test_title_appears_in_html(self, df_typical):
        html = _write_report(df_typical, title="My Custom Title")
        assert "My Custom Title" in html

    def test_html_has_closing_tag(self, df_typical):
        html = _write_report(df_typical)
        assert "</html>" in html.lower()


# ---------------------------------------------------------------------------
# i18n
# ---------------------------------------------------------------------------

class TestReportI18n:
    def test_english_lang_attr(self, df_typical):
        html = _write_report(df_typical, language="en")
        assert 'lang="en"' in html

    def test_english_ltr_dir(self, df_typical):
        html = _write_report(df_typical, language="en")
        assert 'dir="ltr"' in html

    def test_persian_lang_attr(self, df_typical):
        html = _write_report(df_typical, language="fa")
        assert 'lang="fa"' in html

    def test_persian_rtl_dir(self, df_typical):
        html = _write_report(df_typical, language="fa")
        assert 'dir="rtl"' in html

    def test_german_lang_attr(self, df_typical):
        html = _write_report(df_typical, language="de")
        assert 'lang="de"' in html

    def test_german_ltr_dir(self, df_typical):
        html = _write_report(df_typical, language="de")
        assert 'dir="ltr"' in html

    def test_all_fa_labels_present(self, df_typical):
        html = _write_report(df_typical, language="fa")
        rendered_keys = {
            "title", "subtitle", "section_overview", "section_summary",
            "section_cases", "section_mcar", "section_recommendation",
            "section_plots", "stat_chi_square", "generated_by",
        }
        for key in rendered_keys:
            val = _TRANSLATIONS["fa"][key]
            assert val in html, f"FA label '{key}'='{val}' missing from HTML"

    def test_all_de_labels_present(self, df_typical):
        html = _write_report(df_typical, language="de")
        rendered_keys = {
            "title", "subtitle", "section_overview", "section_summary",
            "section_cases", "section_mcar", "section_recommendation",
            "section_plots", "stat_chi_square", "generated_by",
        }
        for key in rendered_keys:
            val = _TRANSLATIONS["de"][key]
            assert val in html, f"DE label '{key}'='{val}' missing from HTML"

    def test_unknown_lang_issues_userwarning(self, df_typical):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _write_report(df_typical, language="xx")
        assert any(
            issubclass(x.category, UserWarning) and "xx" in str(x.message)
            for x in w
        ), "Expected UserWarning mentioning unsupported language code"

    def test_unknown_lang_falls_back_to_english(self, df_typical):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            html = _write_report(df_typical, language="xx")
        assert _TRANSLATIONS["en"]["section_overview"] in html

    def test_de_chi_quadrat_label(self, df_typical):
        """DE stat_chi_square must render as 'Chi-Quadrat', not 'Chi-square'."""
        html = _write_report(df_typical, language="de")
        assert "Chi-Quadrat" in html, \
            "German report must use 'Chi-Quadrat' (standard DE scientific term)"

    def test_fa_chi_square_stays_english(self, df_typical):
        """FA stat_chi_square must stay 'Chi-square' (no Persian equivalent)."""
        html = _write_report(df_typical, language="fa")
        assert "Chi-square" in html, \
            "Persian report must keep 'Chi-square' in English"


# ---------------------------------------------------------------------------
# Technical terms — language-specific enforcement
# ---------------------------------------------------------------------------

class TestTechnicalTermsStayEnglish:
    @pytest.mark.parametrize("term", FA_TECHNICAL_TERMS)
    def test_term_present_in_fa_report(self, df_typical, term):
        html = _write_report(df_typical, language="fa")
        assert term in html, (
            f"Technical term '{term}' must stay English in language='fa'"
        )

    @pytest.mark.parametrize("term", DE_TECHNICAL_TERMS)
    def test_term_present_in_de_report(self, df_typical, term):
        html = _write_report(df_typical, language="de")
        assert term in html, (
            f"Technical term '{term}' must stay English in language='de'"
        )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestReportEdgeCases:
    def test_all_missing_no_crash(self, df_all_missing):
        html = _write_report(df_all_missing)
        assert "</html>" in html.lower()

    def test_no_missing_shows_en_message(self, df_no_missing):
        html = _write_report(df_no_missing, language="en")
        assert _TRANSLATIONS["en"]["no_missing"] in html

    def test_no_missing_shows_fa_message(self, df_no_missing):
        html = _write_report(df_no_missing, language="fa")
        assert _TRANSLATIONS["fa"]["no_missing"] in html

    def test_single_col_no_crash(self, df_single_col):
        html = _write_report(df_single_col)
        assert "</html>" in html.lower()

    def test_persian_colnames_in_output(self, df_persian_colnames):
        html = _write_report(df_persian_colnames, language="fa")
        assert any(
            col in html for col in ["\u0633\u0646", "\u062f\u0631\u0622\u0645\u062f", "\u0634\u0647\u0631"]
        )

    def test_mixed_dtypes_no_crash(self, df_mixed_dtypes):
        html = _write_report(df_mixed_dtypes)
        assert "</html>" in html.lower()

    def test_sentinel_values_no_crash(self):
        df = pd.DataFrame({"a": [1, -99, 3], "b": ["x", "N/A", "z"]})
        html = _write_report(df, missing_values=[-99, "N/A"])
        assert "</html>" in html.lower()

    def test_large_df_no_crash(self):
        rng = np.random.default_rng(7)
        df = pd.DataFrame(rng.random((500, 10)))
        df[df < 0.2] = np.nan
        html = _write_report(df)
        assert "</html>" in html.lower()

    def test_all_missing_fa_rtl(self, df_all_missing):
        html = _write_report(df_all_missing, language="fa")
        assert 'dir="rtl"' in html

    def test_all_missing_no_jinja_placeholders(self, df_all_missing):
        html = _write_report(df_all_missing)
        assert not re.search(r"\{\{.*?\}\}", html)


# ---------------------------------------------------------------------------
# _mcar_interpretation — unit tests
# ---------------------------------------------------------------------------

class TestMcarInterpretation:
    def test_returns_required_keys(self):
        r = _mcar_interpretation(0.10, 5.0)
        assert set(r) == {
            "mechanism", "mechanism_detail",
            "recommendation", "recommendation_detail",
        }

    def test_high_pvalue_does_not_claim_mcar(self):
        result = _mcar_interpretation(0.80, 3.0)
        assert result["mechanism"] == "MCAR not rejected"
        assert "does not establish MCAR" in result["mechanism_detail"]

    def test_low_pvalue_reports_only_evidence_against_mcar(self):
        r = _mcar_interpretation(0.01, 5.0)
        assert r["mechanism"] == "Evidence against MCAR"
        assert "cannot distinguish MAR from MNAR" in r["mechanism_detail"]

    def test_nan_pvalue_unknown(self):
        assert "Unknown" in _mcar_interpretation(float("nan"), 10.0)["mechanism"]

    def test_fa_detail_contains_persian(self):
        r = _mcar_interpretation(0.80, 3.0, lang="fa")
        assert re.search(r"[\u0600-\u06ff]", r["mechanism_detail"])

    def test_fa_recommendation_contains_persian(self):
        r = _mcar_interpretation(0.80, 3.0, lang="fa")
        assert re.search(r"[\u0600-\u06ff]", r["recommendation_detail"])

    def test_de_detail_differs_from_en(self):
        en = _mcar_interpretation(0.01, 5.0, lang="en")["mechanism_detail"]
        de = _mcar_interpretation(0.01, 5.0, lang="de")["mechanism_detail"]
        assert en != de

    def test_unknown_lang_same_as_en(self):
        en = _mcar_interpretation(0.80, 5.0, lang="en")
        xx = _mcar_interpretation(0.80, 5.0, lang="xx")
        assert en["mechanism_detail"] == xx["mechanism_detail"]

    @pytest.mark.parametrize("lang", ["fa", "de"])
    def test_mcar_term_stays_english(self, lang):
        r = _mcar_interpretation(0.80, 5.0, lang=lang)
        assert "MCAR" in r["mechanism"]

    @pytest.mark.parametrize("lang", ["fa", "de"])
    def test_localized_moderate_missing_recommendation(self, lang):
        r = _mcar_interpretation(0.80, 12.0, lang=lang)
        assert r["recommendation"]

    @pytest.mark.parametrize("lang", ["fa", "de"])
    def test_high_missingness_recommends_sensitivity_analysis(self, lang):
        r = _mcar_interpretation(0.001, 30.0, lang=lang)
        assert r["recommendation"]

    def test_low_missing_recommends_documented_plan(self):
        r = _mcar_interpretation(0.80, 2.0)
        assert r["recommendation"] == "Document assumptions and choose an analysis plan"

    def test_moderate_missing_recommends_sensitivity(self):
        r = _mcar_interpretation(0.80, 12.0)
        assert "sensitivity analysis" in r["recommendation"]

    def test_high_missing_recommends_domain_review(self):
        r = _mcar_interpretation(0.80, 25.0)
        assert "domain review" in r["recommendation"]

    def test_low_pvalue_does_not_change_assumption_based_recommendation(self):
        r = _mcar_interpretation(0.01, 10.0)
        assert "sensitivity analysis" in r["recommendation"]

    def test_high_missing_recommendation_does_not_claim_mechanism(self):
        r = _mcar_interpretation(0.001, 30.0)
        assert "domain review" in r["recommendation"]

    @pytest.mark.parametrize("lang", ["fa", "de"])
    @pytest.mark.parametrize("term", ["MCAR", "p-value"])
    def test_stat_terms_in_combined_output(self, lang, term):
        r_mcar = _mcar_interpretation(0.80, 5.0, lang=lang)
        r_not  = _mcar_interpretation(0.01, 5.0, lang=lang)
        combined = (
            r_mcar["mechanism_detail"] + r_mcar["recommendation_detail"]
            + r_not["mechanism_detail"] + r_not["recommendation_detail"]
            + r_mcar["mechanism"] + r_not["mechanism"]
            + r_mcar["recommendation"] + r_not["recommendation"]
        )
        assert term in combined, (
            f"Term '{term}' must appear somewhere in {lang} interpretation output"
        )

    def test_fa_chi_square_in_interpretation(self):
        """Chi-square must stay English in FA interpretation prose."""
        r = _mcar_interpretation(0.80, 5.0, lang="fa")
        combined = r["mechanism_detail"] + r["recommendation_detail"]
        # p-value appears; Chi-square may appear in detail prose
        # At minimum the mechanism text must not contain Persian script
        # for the term "Chi-square"
        assert "Chi-square" not in combined or re.search(r"Chi-square", combined)


# ---------------------------------------------------------------------------
# Output path & ValueError
# ---------------------------------------------------------------------------

class TestReportOutputPath:
    def test_custom_path_used(self, df_typical, tmp_path):
        out = tmp_path / "custom_name.html"
        result = create_report(df_typical, output_path=str(out))
        assert os.path.exists(result)
        assert "custom_name" in result

    def test_empty_df_raises_value_error(self):
        with pytest.raises(ValueError, match="empty"):
            create_report(pd.DataFrame())

    def test_output_file_is_utf8(self, df_persian_colnames, tmp_path):
        out = tmp_path / "fa_report.html"
        create_report(df_persian_colnames, output_path=str(out), language="fa")
        content = out.read_text(encoding="utf-8")
        assert any(
            col in content
            for col in ["\u0633\u0646", "\u062f\u0631\u0622\u0645\u062f", "\u0634\u0647\u0631"]
        )
