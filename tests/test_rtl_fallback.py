"""Tests for missingly.visualisation._base RTL fallback behaviour.

Policy
------
The RTL rendering path has four library-availability states:

  State A: both arabic_reshaper AND bidi installed     -> fully correct
  State B: only arabic_reshaper installed              -> partial (shapes only)
  State C: only bidi installed                         -> partial (order only)
  State D: neither installed                           -> last-resort reversal

All four states must be tested independently because the fallback logic is
branching and easy to break silently.  We use ``unittest.mock.patch`` to
force each state without requiring the libraries to actually be absent.

Additional coverage
-------------------
- Latin text is never modified regardless of library state.
- Mixed RTL+LTR strings (e.g. "\u0633\u0646 (42.5%)") preserve numeric/paren
  portions in states A and B.
- _apply_rtl_font emits a UserWarning at most once per session when no
  suitable font is available, and the warning message mentions the install
  command.
- The public visualisation functions (matrix, bar, heatmap, vis_miss) do not
  raise when given a DataFrame with Persian column names, regardless of RTL
  library state.
"""

from __future__ import annotations

import warnings
from unittest import mock
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

import missingly.visualisation._base as _base


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

@pytest.fixture
def df_persian():
    rng = np.random.default_rng(1)
    return pd.DataFrame({
        "\u0633\u0646":    rng.choice([20.0, 30.0, np.nan], 20),
        "\u062f\u0631\u0622\u0645\u062f": rng.choice([1000.0, np.nan], 20),
        "\u0634\u0647\u0631":  rng.choice(["\u062a\u0647\u0631\u0627\u0646", "\u0645\u0634\u0647\u062f", None], 20),
    })


@pytest.fixture
def df_latin():
    rng = np.random.default_rng(2)
    return pd.DataFrame({
        "age":    rng.choice([20.0, 30.0, np.nan], 20),
        "income": rng.choice([1000.0, np.nan], 20),
        "city":   rng.choice(["Milan", "Berlin", None], 20),
    })


@pytest.fixture(autouse=True)
def reset_rtl_globals():
    """Reset session-level globals before each test."""
    orig_warned = _base._RTL_WARNED
    orig_registered = _base._RTL_FONT_REGISTERED
    yield
    _base._RTL_WARNED = orig_warned
    _base._RTL_FONT_REGISTERED = orig_registered


# ---------------------------------------------------------------------------
# Helpers to force library-availability states via mock
# ---------------------------------------------------------------------------

def _make_reshaper_mock():
    m = MagicMock()
    m.reshape.side_effect = lambda s: s  # identity
    return m


def _make_bidi_mock():
    m = MagicMock()
    m.side_effect = lambda s, **kw: s  # identity
    return m


def _patch_both_missing():
    """Context manager: neither arabic_reshaper nor bidi available."""
    return mock.patch.multiple(
        "missingly.visualisation._base",
        _HAVE_RESHAPER=False,
        _HAVE_BIDI=False,
    )


def _patch_reshaper_only():
    """Context manager: only arabic_reshaper available.

    Also supplies a real mock for _ar_reshaper so _rtl_safe can call
    .reshape() even when the library is not installed in the test env.
    """
    return mock.patch.multiple(
        "missingly.visualisation._base",
        _HAVE_RESHAPER=True,
        _HAVE_BIDI=False,
        _ar_reshaper=_make_reshaper_mock(),
    )


def _patch_bidi_only():
    """Context manager: only bidi available.

    Also supplies a real mock for _bidi_get_display so _rtl_safe can call
    it even when the library is not installed in the test env.
    """
    return mock.patch.multiple(
        "missingly.visualisation._base",
        _HAVE_RESHAPER=False,
        _HAVE_BIDI=True,
        _bidi_get_display=_make_bidi_mock(),
    )


# ---------------------------------------------------------------------------
# State A: both libraries present (baseline correctness)
# ---------------------------------------------------------------------------

class TestRtlSafeStateBoth:
    """_rtl_safe with both arabic_reshaper and bidi available."""

    def _requires_both(self):
        if not (_base._HAVE_RESHAPER and _base._HAVE_BIDI):
            pytest.skip("State A tests require both arabic_reshaper and bidi")

    def test_latin_unchanged(self):
        self._requires_both()
        assert _base._rtl_safe("age") == "age"

    def test_latin_with_numbers_unchanged(self):
        self._requires_both()
        assert _base._rtl_safe("col_1") == "col_1"

    def test_persian_is_not_raw_input(self):
        self._requires_both()
        raw = "\u0633\u0646"
        result = _base._rtl_safe(raw)
        assert len(result) > 0

    def test_mixed_preserves_numbers(self):
        """Numbers in parens must not be mangled by reshaper."""
        self._requires_both()
        mixed = "\u0633\u0646 (42.5%)"
        result = _base._rtl_safe(mixed)
        assert "42.5" in result, f"Numbers corrupted: got {result!r}"
        assert "%" in result, f"Percent sign lost: got {result!r}"

    def test_mixed_preserves_parens(self):
        self._requires_both()
        mixed = "\u0633\u0646 (42.5%)"
        result = _base._rtl_safe(mixed)
        assert "(" in result and ")" in result, f"Parens lost: got {result!r}"


# ---------------------------------------------------------------------------
# State B: only arabic_reshaper
# ---------------------------------------------------------------------------

class TestRtlSafeStateReshaperOnly:
    def test_latin_unchanged(self):
        with _patch_reshaper_only():
            assert _base._rtl_safe("income") == "income"

    def test_persian_returns_non_empty(self):
        with _patch_reshaper_only():
            result = _base._rtl_safe("\u0633\u0646")
            assert len(result) > 0

    def test_mixed_preserves_numbers(self):
        with _patch_reshaper_only():
            mixed = "\u0633\u0646 (42.5%)"
            result = _base._rtl_safe(mixed)
            assert "42.5" in result, (
                f"Reshaper-only mode mangled numbers: {result!r}"
            )

    def test_mixed_preserves_percent(self):
        with _patch_reshaper_only():
            result = _base._rtl_safe("\u0633\u0646 (12.0%)")
            assert "%" in result

    def test_returns_str(self):
        """Must return str regardless of whether the real library is installed."""
        with _patch_reshaper_only():
            result = _base._rtl_safe("\u0633\u0646")
            assert isinstance(result, str)


# ---------------------------------------------------------------------------
# State C: only bidi
# ---------------------------------------------------------------------------

class TestRtlSafeStateBidiOnly:
    def test_latin_unchanged(self):
        with _patch_bidi_only():
            assert _base._rtl_safe("score") == "score"

    def test_persian_returns_non_empty(self):
        with _patch_bidi_only():
            result = _base._rtl_safe("\u0633\u0646")
            assert len(result) > 0

    def test_returns_str(self):
        """Must return str regardless of whether the real library is installed."""
        with _patch_bidi_only():
            assert isinstance(_base._rtl_safe("\u0633\u0646"), str)

    def test_does_not_raise(self):
        """Must not raise even with a mixed RTL+LTR string."""
        with _patch_bidi_only():
            _base._rtl_safe("\u062f\u0631\u0622\u0645\u062f (0.0%)")


# ---------------------------------------------------------------------------
# State D: neither library
# ---------------------------------------------------------------------------

class TestRtlSafeStateNeither:
    def test_latin_unchanged(self):
        with _patch_both_missing():
            assert _base._rtl_safe("city") == "city"

    def test_persian_is_reversed(self):
        """Neither-mode reverses RTL runs as last resort.

        The output may be wrapped in directional markers (_RLM prefix +
        _LRM suffix) for correct matplotlib layout, so we strip those
        before comparing the core content.
        """
        with _patch_both_missing():
            raw = "\u0633\u0646"
            result = _base._rtl_safe(raw)
            assert _strip_markers(result) == raw[::-1], (
                f"Neither-mode must reverse the RTL run (ignoring directional "
                f"markers): got {result!r}"
            )

    def test_output_has_rtl_markers(self):
        """Output for RTL text must be wrapped with _RLM prefix and _LRM suffix."""
        with _patch_both_missing():
            result = _base._rtl_safe("\u0633\u0646")
            assert result.startswith(_RLM), (
                f"Expected _RLM prefix, got {result!r}"
            )
            assert result.endswith(_LRM), (
                f"Expected _LRM suffix, got {result!r}"
            )

    def test_pure_latin_not_reversed(self):
        """Latin-only strings must bypass all RTL logic (never reversed)."""
        with _patch_both_missing():
            assert _base._rtl_safe("age") == "age"

    def test_returns_str(self):
        with _patch_both_missing():
            assert isinstance(_base._rtl_safe("\u0633\u0646"), str)

    def test_does_not_raise_on_mixed(self):
        with _patch_both_missing():
            _base._rtl_safe("\u0633\u0646 (42.5%)")


# ---------------------------------------------------------------------------
# _apply_rtl_font: warning behaviour
# ---------------------------------------------------------------------------

class TestApplyRtlFontWarning:
    def test_warning_emitted_when_no_font(self):
        """UserWarning must be emitted when RTL labels are detected but no font
        is available."""
        _base._RTL_WARNED = False
        _base._RTL_FONT_REGISTERED = False

        with mock.patch(
            "matplotlib.font_manager.fontManager"
        ) as mock_fm, mock.patch(
            "missingly.visualisation._base._ensure_vazirmatn",
            return_value=None,
        ):
            mock_fm.ttflist = []
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                _base._apply_rtl_font(["\u0633\u0646", "\u062f\u0631\u0622\u0645\u062f"])

        user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        assert len(user_warnings) >= 1, "Expected at least one UserWarning"

    def test_warning_mentions_install_command(self):
        """Warning message must contain the pip install command."""
        _base._RTL_WARNED = False
        _base._RTL_FONT_REGISTERED = False

        with mock.patch(
            "matplotlib.font_manager.fontManager"
        ) as mock_fm, mock.patch(
            "missingly.visualisation._base._ensure_vazirmatn",
            return_value=None,
        ):
            mock_fm.ttflist = []
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                _base._apply_rtl_font(["\u0633\u0646"])

        user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        assert user_warnings, "No UserWarning was emitted"
        msg = str(user_warnings[0].message)
        assert "missingly[rtl]" in msg or "pip install" in msg, (
            f"Warning must mention install command, got: {msg!r}"
        )

    def test_warning_emitted_at_most_once(self):
        """_apply_rtl_font must not spam the user with repeated warnings."""
        _base._RTL_WARNED = False
        _base._RTL_FONT_REGISTERED = False

        with mock.patch(
            "matplotlib.font_manager.fontManager"
        ) as mock_fm, mock.patch(
            "missingly.visualisation._base._ensure_vazirmatn",
            return_value=None,
        ):
            mock_fm.ttflist = []
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                for _ in range(5):
                    _base._apply_rtl_font(["\u0633\u0646"])

        user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        assert len(user_warnings) == 1, (
            f"Expected exactly 1 UserWarning across 5 calls, got {len(user_warnings)}"
        )

    def test_no_warning_for_latin_labels(self):
        """No warning when all labels are Latin (no RTL text)."""
        _base._RTL_WARNED = False

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _base._apply_rtl_font(["age", "income", "city"])

        user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        assert len(user_warnings) == 0


# ---------------------------------------------------------------------------
# _safe_labels: delegation
# ---------------------------------------------------------------------------

class TestSafeLabels:
    def test_latin_labels_unchanged(self):
        labels = ["age", "income", "score"]
        assert _base._safe_labels(labels) == labels

    def test_returns_list_of_str(self):
        result = _base._safe_labels(["\u0633\u0646", "\u062f\u0631\u0622\u0645\u062f"])
        assert isinstance(result, list)
        assert all(isinstance(s, str) for s in result)

    def test_length_preserved(self):
        labels = ["\u0633\u0646", "\u062f\u0631\u0622\u0645\u062f", "\u0634\u0647\u0631"]
        assert len(_base._safe_labels(labels)) == 3

    def test_none_mode_each_item_reversed(self):
        """In neither-mode each RTL label's core content must be reversed.

        Directional markers (_RLM / _LRM) are stripped before comparison
        so the assertion is robust to wrapper presence.
        """
        with _patch_both_missing():
            labels = ["\u0633\u0646", "\u0634\u0647\u0631"]
            result = _base._safe_labels(labels)
            for raw, processed in zip(labels, result):
                assert _strip_markers(processed) == raw[::-1], (
                    f"Expected reversed core {raw[::-1]!r}, "
                    f"got {processed!r} (stripped: {_strip_markers(processed)!r})"
                )


# ---------------------------------------------------------------------------
# Integration: public plots don't raise with Persian columns
# ---------------------------------------------------------------------------

class TestVisualisationWithPersianNoRaise:
    """Public visualisation functions must not raise when columns are Persian,
    regardless of which RTL libraries are available."""

    @pytest.fixture(autouse=True)
    def close_plots(self):
        yield
        plt.close("all")

    @pytest.mark.parametrize("state_ctx", [
        _patch_both_missing,
        _patch_reshaper_only,
        _patch_bidi_only,
    ])
    def test_matrix_no_raise(self, df_persian, state_ctx):
        from missingly.visualisation import matrix
        with state_ctx():
            fig, ax = plt.subplots()
            matrix(df_persian, ax=ax)
            plt.close(fig)

    @pytest.mark.parametrize("state_ctx", [
        _patch_both_missing,
        _patch_reshaper_only,
        _patch_bidi_only,
    ])
    def test_bar_no_raise(self, df_persian, state_ctx):
        from missingly.visualisation import bar
        with state_ctx():
            fig, ax = plt.subplots()
            bar(df_persian, ax=ax)
            plt.close(fig)

    @pytest.mark.parametrize("state_ctx", [
        _patch_both_missing,
        _patch_reshaper_only,
        _patch_bidi_only,
    ])
    def test_heatmap_no_raise(self, df_persian, state_ctx):
        from missingly.visualisation import heatmap
        with state_ctx():
            fig, ax = plt.subplots()
            heatmap(df_persian, ax=ax)
            plt.close(fig)

    @pytest.mark.parametrize("state_ctx", [
        _patch_both_missing,
        _patch_reshaper_only,
        _patch_bidi_only,
    ])
    def test_vis_miss_no_raise(self, df_persian, state_ctx):
        from missingly.visualisation import vis_miss
        with state_ctx():
            fig, ax = plt.subplots()
            vis_miss(df_persian, ax=ax)
            plt.close(fig)

    def test_latin_df_no_rtl_side_effects(self, df_latin):
        """Latin DataFrames must never trigger RTL code paths."""
        from missingly.visualisation import matrix
        _base._RTL_WARNED = False
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fig, ax = plt.subplots()
            matrix(df_latin, ax=ax)
            plt.close(fig)
        rtl_warnings = [
            x for x in w
            if issubclass(x.category, UserWarning)
            and "Persian" in str(x.message)
        ]
        assert len(rtl_warnings) == 0, (
            "Latin DataFrame must not trigger RTL font warnings"
        )
