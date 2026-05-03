"""HTML report generation for missing data analysis.

This module provides :func:`create_report`, which generates a
self-contained, standalone HTML report summarising all aspects of
missingness in a DataFrame:

1. **Dataset overview** — shape, total missing count, percentage.
2. **Per-variable summary** — n_miss, pct_miss for every column.
3. **Per-case summary** — top 10 rows with the most missing values.
4. **MCAR test result** — Little's test chi-square, df, p-value, and a
   plain-English interpretation.
5. **Imputation recommendation** — based on MCAR test result and
   missingness fraction, a concrete recommendation is provided.
6. **Visualisations** — matrix, bar, heatmap, and vis_miss plots,
   embedded as base64 PNG images (no external dependencies).

All plots are generated with a non-interactive Matplotlib backend so the
function works correctly in headless CI environments and notebooks alike.

Compatibility
-------------
Requires Python 3.9+, pandas 2.0+, matplotlib, and jinja2.
"""

from __future__ import annotations

import base64
import os
import warnings
from io import BytesIO
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # headless backend — must be set before pyplot import
import matplotlib.pyplot as plt
import pandas as pd

from jinja2 import Environment, FileSystemLoader

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
        warnings.warn(f"Plot {plot_fn.__name__!r} failed: {exc}", UserWarning, stacklevel=2)
        plt.close("all")
        return None


def _mcar_interpretation(p_value: float, overall_pct: float) -> dict:
    """Produce a plain-English MCAR interpretation and imputation recommendation.

    Parameters
    ----------
    p_value : float
        p-value from Little's MCAR test.
    overall_pct : float
        Overall percentage of missing values in the DataFrame.

    Returns
    -------
    dict with keys:
        ``mechanism`` (str), ``mechanism_detail`` (str),
        ``recommendation`` (str), ``recommendation_detail`` (str).
    """
    import math

    if math.isnan(p_value):
        mechanism = "Unknown"
        mechanism_detail = (
            "The MCAR test could not be computed (insufficient data or "
            "only one missing pattern). Treat the mechanism as unknown."
        )
    elif p_value > 0.05:
        mechanism = "MCAR (Missing Completely At Random)"
        mechanism_detail = (
            f"p = {p_value:.4f} > 0.05: cannot reject MCAR. The probability "
            "of a value being missing is unrelated to any observed or "
            "unobserved data. Simple imputation strategies are statistically "
            "justified."
        )
    else:
        mechanism = "MAR / MNAR (Not MCAR)"
        mechanism_detail = (
            f"p = {p_value:.4f} ≤ 0.05: MCAR is rejected. The pattern of "
            "missingness is related to the data. Simple mean/median "
            "imputation may introduce bias; model-based methods are preferred."
        )

    # Recommendation based on mechanism + overall missingness fraction
    if math.isnan(p_value) or p_value > 0.05:
        if overall_pct < 5:
            recommendation = "Complete-case analysis or mean/median imputation"
            recommendation_detail = (
                "Missingness is MCAR and low (< 5%). Dropping rows with "
                "missing values or using mean/median imputation will not "
                "meaningfully bias results."
            )
        elif overall_pct < 20:
            recommendation = "KNN or MICE imputation"
            recommendation_detail = (
                "Missingness is MCAR but moderate (5–20%). KNN or MICE "
            "preserves more statistical structure than mean/median."
            )
        else:
            recommendation = "MICE imputation + sensitivity analysis"
            recommendation_detail = (
                "Missingness is MCAR but high (≥ 20%). Use MICE and run a "
                "sensitivity analysis to check how much imputation choices "
                "affect your downstream results."
            )
    else:
        if overall_pct < 20:
            recommendation = "MICE or Random Forest imputation"
            recommendation_detail = (
                "Missingness is not MCAR. Model-based imputation (MICE, RF, GB) "
                "is recommended as it can leverage correlations between variables "
                "to produce less biased imputations."
            )
        else:
            recommendation = "Random Forest or Gradient Boosting imputation + domain review"
            recommendation_detail = (
                "Missingness is not MCAR and is high (≥ 20%). Use RF or GB "
                "imputation. Additionally, consider whether the high missingness "
                "indicates a data collection problem that should be fixed upstream."
            )

    return {
        "mechanism": mechanism,
        "mechanism_detail": mechanism_detail,
        "recommendation": recommendation,
        "recommendation_detail": recommendation_detail,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_report(
    df: pd.DataFrame,
    output_path: str = "missing_data_report.html",
    title: str = "Missing Data Analysis Report",
    missing_values: Optional[list] = None,
) -> str:
    """Generate a self-contained HTML report for missing data analysis.

    The report includes:

    1. **Dataset overview** (shape, total missing, overall pct missing).
    2. **Per-variable summary table** (n_miss, pct_miss per column).
    3. **Per-case summary** (top 10 most-missing rows).
    4. **MCAR test** (chi-square, df, p-value, interpretation).
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
        Title displayed at the top of the HTML report.
    missing_values : list, optional
        Additional sentinel values treated as missing (e.g. ``[-99, "N/A"]``).
        These are replaced with ``NaN`` before any analysis.

    Returns
    -------
    str
        The absolute path to the saved HTML file.

    Raises
    ------
    ValueError
        If *df* is empty.

    Example
    -------
    >>> import pandas as pd, numpy as np
    >>> df = pd.DataFrame({'age': [25, np.nan, 35], 'city': ['A', None, 'C']})
    >>> create_report(df, output_path='/tmp/report.html')
    '/tmp/report.html'
    """
    if df.empty:
        raise ValueError("Cannot generate a report for an empty DataFrame.")

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
                mcar_result["p_value"], overall_pct
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
        "matrix_plot":  _safe_plot(matrix,  df_analysis),
        "bar_plot":     _safe_plot(bar,     df_analysis),
        "heatmap_plot": _safe_plot(heatmap, df_analysis),
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
