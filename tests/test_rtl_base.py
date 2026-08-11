"""Tests for missingly.visualisation._base RTL helpers and utilities.

All four fallback paths of ``_rtl_safe`` are exercised by monkeypatching
``_HAVE_RESHAPER`` and ``_HAVE_BIDI`` so the tests run correctly regardless
of whether ``arabic-reshaper`` and ``python-bidi`` are installed.
"""
from __future__ import annotations

import logging
import warnings
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

import missingly.visualisation._base as _base
from missingly.visualisation._base import (
    _is_rtl,
    _nullity,
    _pct_labels,
    _rtl_safe,
    _safe_labels,
    _split_runs,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RLM = "\u200F"
_LRM = "\u200E"


def _strip_markers(s: str) -> str:
    """Remove leading RLM and trailing LRM so assertions stay marker-agnostic."""
    return s.lstrip(_RLM).rstrip(_LRM)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def simple_df() -> pd.DataFrame:
    """Small DataFrame with Latin column names and NaN values."""
    return pd.DataFrame(
        {
            "age": [25, np.nan, 30, np.nan, 45],
            "income": [50_000, 60_000, np.nan, 70_000, np.nan],
            "city": ["Berlin", "Paris", "Berlin", None, "Rome"],
        }
    )


@pytest.fixture()
def persian_df() -> pd.DataFrame:
    """DataFrame with Persian (Farsi) column names and NaN values."""
    return pd.DataFrame(
        {
            "\u0633\u0646": [25, np.nan, 30, np.nan, 45],       # سن
            "\u062f\u0631\u0622\u0645\u062f": [1000, np.nan, np.nan, 4000, 5000],  # درآمد
            "\u0634\u0647\u0631": ["\u062a\u0647\u0631\u0627\u0646", None, "\u0627\u0635\u0641\u0647\u0627\u0646", None, "\u0634\u06cc\u0631\u0627\u0632"],  # شهر
        }
    )


@pytest.fixture()
def mixed_df() -> pd.DataFrame:
    """DataFrame with a mix of Latin and Persian column names."""
    return pd.DataFrame(
        {
            "age": [25, np.nan, 30],
            "\u0633ن": [np.nan, 35, np.nan],  # سن
            "score": [88.5, 92.0, np.nan],
        }
    )


# ---------------------------------------------------------------------------
# _is_rtl
# ---------------------------------------------------------------------------

class TestIsRtl:
    def test_latin_returns_false(self):
        assert _is_rtl("age") is False

    def test_latin_with_numbers_returns_false(self):
        assert _is_rtl("col_42") is False

    def test_pure_persian_returns_true(self):
        assert _is_rtl("\u0633\u0646") is True  # سن

    def test_pure_arabic_returns_true(self):
        assert _is_rtl("\u0645\u0631\u062d\u0628\u0627") is True  # مرحبا

    def test_mixed_rtl_ltr_returns_true(self):
        assert _is_rtl("\u0633\u0646 (42.5%)") is True

    def test_empty_string_returns_false(self):
        assert _is_rtl("") is False

    def test_numbers_only_returns_false(self):
        assert _is_rtl("3.14") is False

    def test_unicode_non_rtl_returns_false(self):
        # Greek mu (μ) is not Arabic/Persian
        assert _is_rtl("\u03bc") is False


# ---------------------------------------------------------------------------
# _split_runs
# ---------------------------------------------------------------------------

class TestSplitRuns:
    def test_latin_only_single_run(self):
        runs = _split_runs("age")
        assert runs == [(False, "age")]

    def test_persian_only_single_run(self):
        runs = _split_runs("\u0633\u0646")  # سن
        assert runs == [(True, "\u0633\u0646")]

    def test_mixed_two_runs(self):
        # "سن (42.5%)" -> RTL run + LTR run
        runs = _split_runs("\u0633\u0646 (42.5%)")
        assert len(runs) == 2
        assert runs[0] == (True, "\u0633\u0646")
        assert runs[1] == (False, " (42.5%)")

    def test_mixed_three_runs(self):
        # "col_سن_val" -> LTR + RTL + LTR
        runs = _split_runs("col_\u0633\u0646_val")
        assert len(runs) == 3
        assert runs[0] == (False, "col_")
        assert runs[1] == (True, "\u0633\u0646")
        assert runs[2] == (False, "_val")

    def test_empty_string(self):
        runs = _split_runs("")
        assert runs == [(False, "")]

    def test_single_char_latin(self):
        runs = _split_runs("x")
        assert runs == [(False, "x")]

    def test_single_char_persian(self):
        runs = _split_runs("\u0633")  # س
        assert runs == [(True, "\u0633")]

    def test_reconstructed_string_matches_input(self):
        """Joining all run chunks must reproduce the original string."""
        for text in ["age", "\u0633\u0646", "\u0633\u0646 (42.5%)", "col_\u0633\u0646_val", "", "12.5%"]:
            runs = _split_runs(text)
            assert "".join(chunk for _, chunk in runs) == text


# ---------------------------------------------------------------------------
# _rtl_safe — regression test for mixed RTL+LTR
# ---------------------------------------------------------------------------

class TestRtlSafeMixed:
    """The core regression: mixed strings must not produce reversed numbers."""

    def test_latin_passthrough(self):
        assert _rtl_safe("age") == "age"

    def test_latin_with_special_chars_passthrough(self):
        assert _rtl_safe("col_name (42.5%)") == "col_name (42.5%)"

    def test_mixed_ltr_segment_preserved_both_libraries(self):
        """With both reshaper+bidi: numbers/parens must not be reversed."""
        mock_reshaper = MagicMock()
        mock_reshaper.reshape.side_effect = lambda s: s  # identity
        mock_bidi = MagicMock()
        mock_bidi.side_effect = lambda s, **kw: s  # identity

        with (
            patch.object(_base, "_HAVE_RESHAPER", True),
            patch.object(_base, "_HAVE_BIDI", True),
            patch.object(_base, "_ar_reshaper", mock_reshaper),
            patch.object(_base, "_bidi_get_display", mock_bidi),
        ):
            result = _rtl_safe("\u0633\u0646 (42.5%)")

        # reshaper must only be called on the RTL run, NOT on " (42.5%)"
        call_args = [call.args[0] for call in mock_reshaper.reshape.call_args_list]
        assert all("42.5" not in arg for arg in call_args), (
            f"arabic_reshaper was called with LTR content: {call_args}"
        )
        # The LTR segment must appear intact in the output
        assert "42.5%" in result

    def test_mixed_reshaper_only_ltr_preserved(self):
        """With reshaper only (no bidi): LTR segment must survive."""
        mock_reshaper = MagicMock()
        mock_reshaper.reshape.side_effect = lambda s: s

        with (
            patch.object(_base, "_HAVE_RESHAPER", True),
            patch.object(_base, "_HAVE_BIDI", False),
            patch.object(_base, "_ar_reshaper", mock_reshaper),
        ):
            result = _rtl_safe("\u0633\u0646 (42.5%)")

        call_args = [call.args[0] for call in mock_reshaper.reshape.call_args_list]
        assert all("42.5" not in arg for arg in call_args), (
            f"arabic_reshaper was called with LTR content: {call_args}"
        )
        assert "42.5%" in result

    def test_mixed_bidi_only_does_not_crash(self):
        """With bidi only (no reshaper): must not crash; numbers intact."""
        mock_bidi = MagicMock()
        mock_bidi.side_effect = lambda s, **kw: s

        with (
            patch.object(_base, "_HAVE_RESHAPER", False),
            patch.object(_base, "_HAVE_BIDI", True),
            patch.object(_base, "_bidi_get_display", mock_bidi),
        ):
            result = _rtl_safe("\u0633\u0646 (42.5%)")

        assert "42.5%" in result

    def test_mixed_no_libraries_ltr_preserved(self):
        """With neither library: all original characters must be present.

        The output is wrapped with _RLM/_LRM directional markers, so we
        strip those before comparing character sets.
        """
        with (
            patch.object(_base, "_HAVE_RESHAPER", False),
            patch.object(_base, "_HAVE_BIDI", False),
        ):
            result = _rtl_safe("\u0633\u0646 (42.5%)")

        # Strip directional markers before comparing character content.
        core = _strip_markers(result)
        assert sorted(core) == sorted("\u0633\u0646 (42.5%)"), (
            f"Character set mismatch after stripping markers. core={core!r}"
        )

    def test_pure_latin_unaffected_by_any_flag_combo(self):
        """Latin-only strings bypass all processing regardless of flags."""
        for have_r, have_b in [(True, True), (True, False), (False, True), (False, False)]:
            with (
                patch.object(_base, "_HAVE_RESHAPER", have_r),
                patch.object(_base, "_HAVE_BIDI", have_b),
            ):
                assert _rtl_safe("income") == "income"


# ---------------------------------------------------------------------------
# _rtl_safe — all 4 library-availability paths
# ---------------------------------------------------------------------------

class TestRtlSafeFallbackPaths:
    """Each path must return a non-empty string and not raise."""

    PERSIAN = "\u0633\u0646"  # سن

    def test_both_libraries(self):
        mock_reshaper = MagicMock()
        mock_reshaper.reshape.side_effect = lambda s: s
        mock_bidi = MagicMock()
        mock_bidi.side_effect = lambda s, **kw: s

        with (
            patch.object(_base, "_HAVE_RESHAPER", True),
            patch.object(_base, "_HAVE_BIDI", True),
            patch.object(_base, "_ar_reshaper", mock_reshaper),
            patch.object(_base, "_bidi_get_display", mock_bidi),
        ):
            result = _rtl_safe(self.PERSIAN)

        assert isinstance(result, str) and result
        mock_reshaper.reshape.assert_called_once_with(self.PERSIAN)
        mock_bidi.assert_called_once()

    def test_reshaper_only(self):
        mock_reshaper = MagicMock()
        mock_reshaper.reshape.side_effect = lambda s: s

        with (
            patch.object(_base, "_HAVE_RESHAPER", True),
            patch.object(_base, "_HAVE_BIDI", False),
            patch.object(_base, "_ar_reshaper", mock_reshaper),
        ):
            result = _rtl_safe(self.PERSIAN)

        assert isinstance(result, str) and result
        mock_reshaper.reshape.assert_called_once_with(self.PERSIAN)

    def test_bidi_only(self):
        mock_bidi = MagicMock()
        mock_bidi.side_effect = lambda s, **kw: s

        with (
            patch.object(_base, "_HAVE_RESHAPER", False),
            patch.object(_base, "_HAVE_BIDI", True),
            patch.object(_base, "_bidi_get_display", mock_bidi),
        ):
            result = _rtl_safe(self.PERSIAN)

        assert isinstance(result, str) and result
        mock_bidi.assert_called_once_with(self.PERSIAN)

    def test_neither_library_reverses_string(self):
        """Neither-mode reverses the RTL run as last resort.

        The output is wrapped with _RLM/_LRM directional markers; strip
        those before comparing the core reversed content.
        """
        with (
            patch.object(_base, "_HAVE_RESHAPER", False),
            patch.object(_base, "_HAVE_BIDI", False),
        ):
            result = _rtl_safe(self.PERSIAN)

        assert _strip_markers(result) == self.PERSIAN[::-1], (
            f"Neither-mode must reverse the RTL run (ignoring markers): "
            f"got {result!r}"
        )


# ---------------------------------------------------------------------------
# _safe_labels
# ---------------------------------------------------------------------------

class TestSafeLabels:
    def test_latin_labels_unchanged(self):
        labels = ["age", "income", "city"]
        assert _safe_labels(labels) == labels

    def test_mixed_labels_returns_list_of_strings(self):
        labels = ["age", "\u0633\u0646", "score"]
        result = _safe_labels(labels)
        assert isinstance(result, list)
        assert len(result) == 3
        assert all(isinstance(s, str) for s in result)

    def test_non_string_inputs_coerced(self):
        result = _safe_labels([1, 2.5, None])
        assert all(isinstance(s, str) for s in result)

    def test_empty_sequence(self):
        assert _safe_labels([]) == []


# ---------------------------------------------------------------------------
# _pct_labels
# ---------------------------------------------------------------------------

class TestPctLabels:
    def test_latin_columns_contain_pct(self, simple_df):
        labels = _pct_labels(simple_df)
        assert len(labels) == 3
        # Each label must contain a percentage string
        assert all("%" in lbl for lbl in labels)

    def test_no_missing_shows_0_pct(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        labels = _pct_labels(df)
        assert all("0.0%" in lbl for lbl in labels)

    def test_all_missing_shows_100_pct(self):
        df = pd.DataFrame({"a": [np.nan, np.nan], "b": [np.nan, np.nan]})
        labels = _pct_labels(df)
        assert all("100.0%" in lbl for lbl in labels)

    def test_persian_columns_return_strings(self, persian_df):
        """Persian column names must produce valid label strings."""
        labels = _pct_labels(persian_df)
        assert len(labels) == 3
        assert all(isinstance(lbl, str) and lbl for lbl in labels)
        # Percentage values must be present in some form
        assert all("%" in lbl for lbl in labels)

    def test_sentinel_values_counted(self):
        df = pd.DataFrame({"a": [1, -99, 3], "b": [0, 0, 0]})
        labels = _pct_labels(df, missing_values=[-99])
        # Column 'a' has 1/3 missing
        assert "33.3%" in labels[0]
        # Column 'b' has 0 missing
        assert "0.0%" in labels[1]


# ---------------------------------------------------------------------------
# _nullity
# ---------------------------------------------------------------------------

class TestNullity:
    def test_no_missing_all_false(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        result = _nullity(df)
        assert not result.any().any()

    def test_nan_detected(self, simple_df):
        result = _nullity(simple_df)
        assert result["age"].sum() == 2
        assert result["income"].sum() == 2
        assert result["city"].sum() == 1

    def test_sentinel_values(self):
        df = pd.DataFrame({"a": [1, -99, 3, np.nan]})
        result = _nullity(df, missing_values=[-99])
        assert result["a"].tolist() == [False, True, False, True]

    def test_multiple_sentinels(self):
        df = pd.DataFrame({"a": ["N/A", "valid", "", -99]})
        result = _nullity(df, missing_values=["N/A", "", -99])
        assert result["a"].tolist() == [True, False, True, True]

    def test_output_shape_matches_input(self, simple_df):
        result = _nullity(simple_df)
        assert result.shape == simple_df.shape

    def test_all_missing_column(self):
        df = pd.DataFrame({"a": [np.nan, np.nan, np.nan]})
        result = _nullity(df)
        assert result["a"].all()

    def test_no_missing_values_arg_uses_nan_only(self):
        df = pd.DataFrame({"a": [-99, 0, 1]})
        result = _nullity(df)  # no missing_values kwarg
        assert not result.any().any()

    def test_mixed_dtypes(self):
        df = pd.DataFrame({
            "int_col": pd.array([1, pd.NA, 3], dtype="Int64"),
            "str_col": ["a", None, "c"],
            "float_col": [1.0, np.nan, 3.0],
        })
        result = _nullity(df)
        assert result["int_col"].tolist() == [False, True, False]
        assert result["str_col"].tolist() == [False, True, False]
        assert result["float_col"].tolist() == [False, True, False]


# ---------------------------------------------------------------------------
# _ensure_vazirmatn — corrupt cache invalidation
# ---------------------------------------------------------------------------

class TestEnsureVazirmatn:
    def test_corrupt_cache_is_removed_and_redownloaded(self, tmp_path):
        """A cached file with wrong magic bytes must be removed and
        a fresh download attempted."""
        font_path = tmp_path / "Vazirmatn-Regular.ttf"
        font_path.write_bytes(b"THIS IS NOT A FONT")

        valid_ttf = b"\x00\x01\x00\x00" + b"\x00" * 100

        with (
            patch.object(_base, "_FONT_CACHE_DIR", tmp_path),
            patch("urllib.request.urlopen") as mock_url,
        ):
            mock_response = MagicMock()
            mock_response.__enter__ = lambda s: s
            mock_response.__exit__ = MagicMock(return_value=False)
            mock_response.read.return_value = valid_ttf
            mock_url.return_value = mock_response

            result = _base._ensure_vazirmatn()

        assert result is not None
        assert result.read_bytes()[:4] == b"\x00\x01\x00\x00"

    def test_valid_cache_returned_without_download(self, tmp_path):
        font_path = tmp_path / "Vazirmatn-Regular.ttf"
        font_path.write_bytes(b"\x00\x01\x00\x00" + b"\x00" * 100)

        with (
            patch.object(_base, "_FONT_CACHE_DIR", tmp_path),
            patch("urllib.request.urlopen") as mock_url,
        ):
            result = _base._ensure_vazirmatn()
            mock_url.assert_not_called()

        assert result == font_path

    def test_network_failure_returns_none(self, tmp_path):
        with (
            patch.object(_base, "_FONT_CACHE_DIR", tmp_path),
            patch("urllib.request.urlopen", side_effect=OSError("no network")),
        ):
            result = _base._ensure_vazirmatn()

        assert result is None

    def test_invalid_download_returns_none(self, tmp_path):
        """A download that returns non-TTF data must not be cached."""
        with (
            patch.object(_base, "_FONT_CACHE_DIR", tmp_path),
            patch("urllib.request.urlopen") as mock_url,
        ):
            mock_response = MagicMock()
            mock_response.__enter__ = lambda s: s
            mock_response.__exit__ = MagicMock(return_value=False)
            mock_response.read.return_value = b"<html>404 not found</html>"
            mock_url.return_value = mock_response

            result = _base._ensure_vazirmatn()

        assert result is None
        assert not (tmp_path / "Vazirmatn-Regular.ttf").exists()


# ---------------------------------------------------------------------------
# _register_font — silent failure now logs a warning
# ---------------------------------------------------------------------------

class TestRegisterFont:
    def test_logs_warning_on_failure(self, tmp_path, caplog):
        bad_path = tmp_path / "not_a_font.ttf"
        bad_path.write_bytes(b"garbage")

        with caplog.at_level(logging.WARNING, logger="missingly.visualisation._base"):
            result = _base._register_font(bad_path)

        assert result is False
        assert any("Failed to register font" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# _apply_rtl_font — basic integration
# ---------------------------------------------------------------------------

class TestApplyRtlFont:
    def test_no_rtl_labels_early_return(self):
        """Non-RTL labels must not touch rcParams."""
        import matplotlib as mpl
        original = mpl.rcParams["font.family"]
        _base._apply_rtl_font(["age", "income", "city"])
        assert mpl.rcParams["font.family"] == original

    def test_rtl_labels_trigger_font_lookup(self):
        """RTL labels must trigger the font resolution logic without crashing."""
        try:
            _base._apply_rtl_font(["\u0633\u0646", "\u062f\u0631\u0622\u0645\u062f"])
        except Exception as exc:  # noqa: BLE001 - test reports any unexpected rendering failure.
            pytest.fail(f"_apply_rtl_font raised unexpectedly: {exc}")

    def test_warns_when_no_font_available(self):
        """When font resolution fails completely, a UserWarning is emitted."""
        original_warned = _base._RTL_WARNED
        original_registered = _base._RTL_FONT_REGISTERED
        _base._RTL_WARNED = False
        _base._RTL_FONT_REGISTERED = False

        try:
            with (
                patch("matplotlib.font_manager.fontManager") as mock_fm,
                patch.object(_base, "_ensure_vazirmatn", return_value=None),
            ):
                mock_fm.ttflist = []
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    _base._apply_rtl_font(["\u0633\u0646"])

            assert any(issubclass(w.category, UserWarning) for w in caught)
        finally:
            _base._RTL_WARNED = original_warned
            _base._RTL_FONT_REGISTERED = original_registered
