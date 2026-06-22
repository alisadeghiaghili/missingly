"""HTML report generation for missing data analysis.

This module provides :func:`create_report`, which generates a
self-contained, standalone HTML report summarising all aspects of
missingness in a DataFrame.

All user-facing strings are resolved through :class:`~missingly.i18n.Translator`
so the report can be rendered in any supported locale.  Pass ``locale="fa"``
for a fully Persian report, ``locale="en"`` (default) for English, or any
other locale that has a corresponding JSON file in ``missingly/i18n/``.

Sections
--------
1. **Dataset overview** — shape, total missing count, percentage.
2. **Per-variable summary** — n_miss, pct_miss for every column.
3. **Per-case summary** — top 10 rows with the most missing values.
4. **MCAR test result** — Little’s test chi-square, df, p-value, and a
   locale-aware interpretation.
5. **Imputation recommendation** — based on MCAR test result and
   missingness fraction.
6. **Visualisations** — matrix, bar, heatmap, and vis_miss plots,
   embedded as base64 PNG images (no external dependencies).

Compatibility
-------------
Requires Python 3.9+, pandas 2.0+, matplotlib, and jinja2.
"""

from __future__ import annotations

import base64
import math
import os
import warnings
from io import BytesIO
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # headless backend — must be set before pyplot import
import matplotlib.pyplot as plt
import pandas as pd

from jinja2 import Environment, FileSystemLoader

from .i18n import Translator
from .summary import miss_var_summary, miss_case_summary, n_miss, pct_miss
from .visualise import matrix, bar, heatmap, vis_miss
from .stats import mcar_test


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fig_to_b64(fig: plt.Figure) -> str:
    """Encode a Matplotlib Figure as a base64 PNG string.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to encode.

    Returns
    -------
    str
        Base64-encoded PNG data (suitable for inline ``<img>`` src).
    """
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _safe_plot(plot_fn, df: pd.DataFrame, **kwargs) -> Optional[str]:
    """Call a plot function and return a base64 PNG, or None on failure.

    Swallows exceptions so a single failed visualisation does not crash
    the entire report.

    Parameters
    ----------
    plot_fn : callable
        A missingly visualisation function that accepts *df* and returns
        a Matplotlib Axes.
    df : pd.DataFrame
        DataFrame to visualise.
    **kwargs
        Additional keyword arguments forwarded to *plot_fn*.

    Returns
    -------
    str or None
        Base64-encoded PNG string, or ``None`` if the plot failed.
    """
    try:
        fig, ax = plt.subplots(figsize=(10, 4))
        plot_fn(df, ax=ax, **kwargs)
        b64 = _fig_to_b64(fig)
        plt.close(fig)
        return b64
    except Exception as exc:  # noqa: BLE001
        warnings.warn(
            f"Plot {plot_fn.__name__!r} failed: {exc}", UserWarning, stacklevel=2
        )
        plt.close("all")
        return None


def _mcar_interpretation(p_value: float, overall_pct: float, tr: Translator) -> dict:
    """Produce a locale-aware MCAR interpretation and imputation recommendation.

    Parameters
    ----------
    p_value : float
        p-value from Little's MCAR test.
    overall_pct : float
        Overall percentage of missing values in the DataFrame.
    tr : Translator
        Active :class:`~missingly.i18n.Translator` instance.

    Returns
    -------
    dict
        Keys: ``mechanism``, ``mechanism_detail``, ``recommendation``,
        ``recommendation_detail``.
    """
    if math.isnan(p_value):
        mechanism = tr.t("mcar.mechanism_unknown")
        mechanism_detail = tr.t("mcar.detail_unknown")
    elif p_value > 0.05:
        mechanism = tr.t("mcar.mechanism_mcar")
        mechanism_detail = tr.t("mcar.detail_mcar", p_value=p_value)
    else:
        mechanism = tr.t("mcar.mechanism_not_mcar")
        mechanism_detail = tr.t("mcar.detail_not_mcar", p_value=p_value)

    # Recommendation keys depend on mechanism + missingness fraction
    is_mcar = math.isnan(p_value) or p_value > 0.05
    if is_mcar:
        if overall_pct < 5:
            rec_key, det_key = "mcar_low", "mcar_low_detail"
        elif overall_pct < 20:
            rec_key, det_key = "mcar_moderate", "mcar_moderate_detail"
        else:
            rec_key, det_key = "mcar_high", "mcar_high_detail"
    else:
        if overall_pct < 20:
            rec_key, det_key = "not_mcar_moderate", "not_mcar_moderate_detail"
        else:
            rec_key, det_key = "not_mcar_high", "not_mcar_high_detail"

    return {
        "mechanism": mechanism,
        "mechanism_detail": mechanism_detail,
        "recommendation": tr.t(f"recommendation.{rec_key}"),
        "recommendation_detail": tr.t(f"recommendation.{det_key}"),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_report(
    df: pd.DataFrame,
    output_path: str = "missing_data_report.html",
    title: Optional[str] = None,
    missing_values: Optional[list] = None,
    locale: str = "en",
) -> str:
    """Generate a self-contained HTML report for missing data analysis.

    The report includes:

    1. **Dataset overview** (shape, total missing, overall pct missing).
    2. **Per-variable summary table** (n_miss, pct_miss per column).
    3. **Per-case summary** (top 10 most-missing rows).
    4. **MCAR test** (chi-square, df, p-value, locale-aware interpretation).
    5. **Imputation recommendation** (concrete, based on MCAR result and
       missingness fraction).
    6. **Visualisations** (matrix, bar chart, heatmap, vis_miss heatmap),
       all embedded as inline base64 images.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to analyse.  May contain missing values.
    output_path : str, optional
        File path where the HTML report is saved.  Default is
        ``"missing_data_report.html"`` in the current directory.
    title : str, optional
        Title displayed at the top of the HTML report.  When omitted the
        locale-specific default title is used (e.g. *"Missing Data Analysis
        Report"* for English).
    missing_values : list, optional
        Additional sentinel values treated as missing (e.g. ``[-99, "N/A"]``).
        These are replaced with ``NaN`` before any analysis.
    locale : str, optional
        Locale code for the report language.  Must match a JSON file in
        ``missingly/i18n/`` (e.g. ``"en"``, ``"fa"``).  Defaults to
        ``"en"``.

    Returns
    -------
    str
        The absolute path to the saved HTML file.

    Raises
    ------
    ValueError
        If *df* is empty or *locale* is not supported.

    Example
    -------
    >>> import pandas as pd, numpy as np
    >>> df = pd.DataFrame({'age': [25, np.nan, 35], 'city': ['A', None, 'C']})
    >>> create_report(df, output_path='/tmp/report.html')  # English (default)
    '/tmp/report.html'
    >>> create_report(df, output_path='/tmp/report_fa.html', locale='fa')  # Persian
    '/tmp/report_fa.html'
    """
    if df.empty:
        raise ValueError("Cannot generate a report for an empty DataFrame.")

    # Initialise translator — raises ValueError for unsupported locales
    tr = Translator(locale)

    # Resolve title from locale if not provided
    if title is None:
        title = tr.t("report.title_default")

    # Replace extra sentinels
    df_analysis = df.copy()
    if missing_values:
        df_analysis = df_analysis.replace(missing_values, pd.NA)

    # ---------------------------------------------------------------
    # 1. Dataset overview
    # ---------------------------------------------------------------
    n_rows, n_cols = df_analysis.shape
    total_miss = n_miss(df_analysis)
    overall_pct = pct_miss(df_analysis)
    overview = {
        "rows": n_rows,
        "cols": n_cols,
        "total_cells": n_rows * n_cols,
        "total_missing": total_miss,
        "overall_pct_missing": f"{overall_pct:.2f}%",
    }

    # ---------------------------------------------------------------
    # 2. Per-variable summary
    # ---------------------------------------------------------------
    var_summary_html = (
        miss_var_summary(df_analysis)
        .sort_values("pct_miss", ascending=False)
        .style
        .format({"pct_miss": "{:.2f}"})
        .set_table_attributes('class="summary-table"')
        .to_html()
    )

    # ---------------------------------------------------------------
    # 3. Per-case summary (top 10)
    # ---------------------------------------------------------------
    case_summary_html = (
        miss_case_summary(df_analysis)
        .sort_values("n_miss", ascending=False)
        .head(10)
        .style
        .format({"pct_miss": "{:.2f}"})
        .set_table_attributes('class="summary-table"')
        .to_html()
    )

    # ---------------------------------------------------------------
    # 4. MCAR test
    # ---------------------------------------------------------------
    mcar_result: Optional[dict] = None
    mcar_info: Optional[dict] = None
    numeric_df = df_analysis.select_dtypes(include=["number"])
    if numeric_df.shape[1] >= 2 and numeric_df.isnull().any().any():
        try:
            mcar_result = mcar_test(numeric_df)
            mcar_info = _mcar_interpretation(
                mcar_result["p_value"], overall_pct, tr
            )
        except Exception as exc:
            warnings.warn(
                f"MCAR test failed and will be omitted from report: {exc}",
                UserWarning,
                stacklevel=1,
            )

    # ---------------------------------------------------------------
    # 5. Visualisations
    # ---------------------------------------------------------------
    plots = {
        "matrix_plot":  _safe_plot(matrix,   df_analysis),
        "bar_plot":     _safe_plot(bar,      df_analysis),
        "heatmap_plot": _safe_plot(heatmap,  df_analysis),
        "vismiss_plot": _safe_plot(vis_miss, df_analysis),
    }

    # ---------------------------------------------------------------
    # 6. Render HTML
    # ---------------------------------------------------------------
    template_dir = os.path.join(os.path.dirname(__file__), "templates")
    env = Environment(loader=FileSystemLoader(template_dir), autoescape=False)
    template = env.get_template("report.html")

    html_content = template.render(
        title=title,
        tr=tr,
        locale=locale,
        overview=overview,
        var_summary=var_summary_html,
        case_summary=case_summary_html,
        mcar_result=mcar_result,
        mcar_info=mcar_info,
        **plots,
    )

    output_path = os.path.abspath(output_path)
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(html_content)

    return output_path
