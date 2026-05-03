"""Visualization utilities for missing data analysis.

This module provides a collection of plotting functions that work on
pandas DataFrames.  Every function follows the same convention:

* accepts an optional ``ax`` (matplotlib Axes) so callers can embed
  plots inside existing figures;
* accepts an optional ``missing_values`` list to treat arbitrary
  sentinel values (e.g. ``-99``, ``"N/A"``) as missing;
* returns the Axes object (or a dict of Axes for multi-panel plots)
  so callers can further customise the output.

Persian / right-to-left labels
-------------------------------
All functions use matplotlib's default Unicode text renderer, which
supports Persian (Farsi) and Arabic column/index labels out of the box.
No special configuration is required; however, for correct RTL glyph
ordering you may want to install a font that supports the required
script (e.g. ``DejaVu Sans``, ``Vazir``) and register it with
``matplotlib.font_manager`` before calling these functions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram as scipy_dendrogram, leaves_list
from scipy.spatial.distance import squareform


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _nullity(df: pd.DataFrame, missing_values: list | None) -> pd.DataFrame:
    """Return a boolean DataFrame indicating missing positions.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    missing_values : list or None
        Sentinel values to treat as missing in addition to ``NaN``.
        Pass ``None`` to treat only true ``NaN`` / ``None`` as missing.

    Returns
    -------
    pd.DataFrame
        Boolean DataFrame with the same shape and labels as *df*.
        ``True`` means the corresponding cell is considered missing.
    """
    if missing_values is None:
        return df.isnull()
    return df.isnull() | df.isin(missing_values)


# ---------------------------------------------------------------------------
# Existing visualisations
# ---------------------------------------------------------------------------

def matrix(df: pd.DataFrame, ax=None, missing_values: list = None, **kwargs):
    """A matrix plot to visualize the location of missing data.

    Creates a heatmap where each cell is coloured by whether the
    corresponding value is missing.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to visualize.
    ax : matplotlib.axes.Axes, optional
        A matplotlib Axes object.  If not provided, a new one is created.
    missing_values : list, optional
        A list of values to treat as missing in addition to ``NaN``.
    **kwargs
        Additional keyword arguments forwarded to ``seaborn.heatmap``.

    Returns
    -------
    matplotlib.axes.Axes
        The Axes containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))

    nullity_matrix = _nullity(df, missing_values)

    sns.heatmap(nullity_matrix, cbar=False, ax=ax, **kwargs)

    if df.shape[0] < 50:
        ax.set_yticks(np.arange(len(df)) + 0.5)
        ax.set_yticklabels(df.index)
    else:
        ax.set_yticks([])

    if df.shape[1] < 50:
        ax.set_xticks(np.arange(len(df.columns)) + 0.5)
        ax.set_xticklabels(df.columns, rotation=90)
    else:
        ax.set_xticks([])

    ax.set_title("Missing Data Matrix")
    return ax


def bar(df: pd.DataFrame, ax=None, missing_values: list = None, **kwargs):
    """A bar plot to visualize the number of missing values per column.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to visualize.
    ax : matplotlib.axes.Axes, optional
        A matplotlib Axes object.  If not provided, a new one is created.
    missing_values : list, optional
        A list of values to treat as missing in addition to ``NaN``.
    **kwargs
        Additional keyword arguments forwarded to ``DataFrame.plot``.

    Returns
    -------
    matplotlib.axes.Axes
        The Axes containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))

    miss_counts = _nullity(df, missing_values).sum()

    miss_counts.plot(kind='bar', ax=ax, **kwargs)
    ax.set_title("Missing Values per Column")
    ax.set_xlabel("Columns")
    ax.set_ylabel("Number of Missing Values")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return ax


def upset(df: pd.DataFrame, missing_values: list = None, **kwargs):
    """An UpSet plot to visualize combinations of missing values.

    Renders three panels in a single figure:

    * **top** — bar chart of intersection sizes, sorted by cardinality
      (descending).
    * **middle** — dot-and-line matrix showing which columns participate
      in each intersection.
    * **left** — horizontal bar chart of per-column missing totals.

    This is a pure-matplotlib implementation that carries no dependency on
    the ``upsetplot`` package, which has an unfixed bug under
    pandas >= 3.0 / Python 3.11+.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to visualize.
    missing_values : list, optional
        A list of values to treat as missing in addition to ``NaN``.
    **kwargs
        Ignored.  Present for API compatibility with the previous
        ``upsetplot``-based signature.

    Returns
    -------
    dict
        A dictionary with keys ``'intersections'``, ``'matrix'``, and
        ``'totals'``, each mapping to the corresponding
        ``matplotlib.axes.Axes``.

    Example
    -------
    >>> import pandas as pd
    >>> from missingly import visualise
    >>> df = pd.DataFrame({'A': [None, 1, None], 'B': [None, None, 1]})
    >>> axes = visualise.upset(df)
    >>> isinstance(axes, dict)
    True
    """
    nullity_matrix = _nullity(df, missing_values)

    missing_cols = list(nullity_matrix.columns[nullity_matrix.any()])
    if not missing_cols:
        print("No missing values to plot.")
        return {}

    nullity_matrix = nullity_matrix[missing_cols].astype(bool)
    n_cols = len(missing_cols)

    combos: dict = {}
    for row in nullity_matrix.itertuples(index=False):
        key = tuple(row)
        combos[key] = combos.get(key, 0) + 1

    combos = {k: v for k, v in combos.items() if any(k)}
    if not combos:
        print("No missing combinations to plot.")
        return {}

    sorted_combos = sorted(combos.items(), key=lambda x: x[1], reverse=True)
    combo_keys = [c[0] for c in sorted_combos]
    combo_counts = [c[1] for c in sorted_combos]
    n_combos = len(combo_keys)

    col_totals = [nullity_matrix[c].sum() for c in missing_cols]

    fig = plt.figure(figsize=(max(8, n_combos * 1.2), max(6, n_cols * 0.9 + 3)))
    gs = gridspec.GridSpec(
        2, 2,
        width_ratios=[1, n_combos],
        height_ratios=[2, n_cols],
        hspace=0.05,
        wspace=0.05,
    )
    ax_bar = fig.add_subplot(gs[0, 1])
    ax_mat = fig.add_subplot(gs[1, 1])
    ax_tot = fig.add_subplot(gs[1, 0])
    fig.add_subplot(gs[0, 0]).set_visible(False)

    x_pos = np.arange(n_combos)
    y_pos = np.arange(n_cols)

    ax_bar.bar(x_pos, combo_counts, color="steelblue", edgecolor="white")
    ax_bar.set_xlim(-0.5, n_combos - 0.5)
    ax_bar.set_xticks([])
    ax_bar.set_ylabel("Intersection size")
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)

    dot_color_on = "#333333"
    dot_color_off = "#cccccc"

    for xi, key in enumerate(combo_keys):
        active_rows = [yi for yi, active in enumerate(key) if active]
        if len(active_rows) > 1:
            ax_mat.plot(
                [xi, xi],
                [min(active_rows), max(active_rows)],
                color=dot_color_on,
                linewidth=2,
                zorder=1,
            )
        for yi, active in enumerate(key):
            ax_mat.scatter(
                xi, yi,
                s=150,
                color=dot_color_on if active else dot_color_off,
                zorder=2,
            )

    ax_mat.set_xlim(-0.5, n_combos - 0.5)
    ax_mat.set_ylim(-0.5, n_cols - 0.5)
    ax_mat.set_xticks([])
    ax_mat.set_yticks(y_pos)
    ax_mat.set_yticklabels(missing_cols)
    ax_mat.spines["top"].set_visible(False)
    ax_mat.spines["right"].set_visible(False)
    ax_mat.spines["bottom"].set_visible(False)

    ax_tot.barh(y_pos, col_totals, color="steelblue", edgecolor="white")
    ax_tot.set_ylim(-0.5, n_cols - 0.5)
    ax_tot.set_yticks([])
    ax_tot.set_xlabel("Set size")
    ax_tot.invert_xaxis()
    ax_tot.spines["top"].set_visible(False)
    ax_tot.spines["left"].set_visible(False)

    fig.suptitle("UpSet Plot of Missing Value Combinations", y=1.01)
    plt.tight_layout()

    return {"intersections": ax_bar, "matrix": ax_mat, "totals": ax_tot}


def scatter_miss(
    df: pd.DataFrame,
    x: str,
    y: str,
    ax=None,
    missing_values: list = None,
    **kwargs,
):
    """A scatter plot that highlights missing values.

    Missing values in either axis variable are imputed slightly below
    the observed minimum so they remain visible on the plot, and are
    coloured differently via the ``hue`` parameter.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to visualize.
    x : str
        Column name for the x-axis.
    y : str
        Column name for the y-axis.
    ax : matplotlib.axes.Axes, optional
        A matplotlib Axes object.  If not provided, a new one is created.
    missing_values : list, optional
        A list of values to treat as missing in addition to ``NaN``.
    **kwargs
        Additional keyword arguments forwarded to ``seaborn.scatterplot``.

    Returns
    -------
    matplotlib.axes.Axes
        The Axes containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    plot_df = df[[x, y]].copy()
    null_df = _nullity(plot_df, missing_values)

    plot_df[f'{x}_NA'] = null_df[x]
    plot_df[f'{y}_NA'] = null_df[y]

    any_x_na = plot_df[f'{x}_NA'].any()
    any_y_na = plot_df[f'{y}_NA'].any()

    if any_x_na:
        x_min = pd.to_numeric(plot_df[x], errors='coerce').min()
        plot_df[x] = pd.to_numeric(plot_df[x], errors='coerce').fillna(
            x_min - abs(x_min) * 0.1
        )
    if any_y_na:
        y_min = pd.to_numeric(plot_df[y], errors='coerce').min()
        plot_df[y] = pd.to_numeric(plot_df[y], errors='coerce').fillna(
            y_min - abs(y_min) * 0.1
        )

    hue = None
    if any_x_na and any_y_na:
        hue = plot_df[f'{x}_NA'].astype(str) + "_" + plot_df[f'{y}_NA'].astype(str)
        hue.name = "Missingness"
    elif any_x_na:
        hue = plot_df[f'{x}_NA']
        hue.name = f"Missing {x}"
    elif any_y_na:
        hue = plot_df[f'{y}_NA']
        hue.name = f"Missing {y}"

    sns.scatterplot(data=plot_df, x=x, y=y, hue=hue, ax=ax, **kwargs)
    ax.set_title(f"Scatter Plot of {x} vs {y} with Missing Values")
    return ax


def miss_case(df: pd.DataFrame, ax=None, missing_values: list = None, **kwargs):
    """A bar plot to visualize the number of missing values per case (row).

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to visualize.
    ax : matplotlib.axes.Axes, optional
        A matplotlib Axes object.  If not provided, a new one is created.
    missing_values : list, optional
        A list of values to treat as missing in addition to ``NaN``.
    **kwargs
        Additional keyword arguments forwarded to ``DataFrame.plot``.

    Returns
    -------
    matplotlib.axes.Axes
        The Axes containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))

    miss_counts = _nullity(df, missing_values).sum(axis=1)

    miss_counts.plot(kind='bar', ax=ax, **kwargs)
    ax.set_title("Missing Values per Case")
    ax.set_xlabel("Cases (Rows)")
    ax.set_ylabel("Number of Missing Values")
    plt.tight_layout()
    return ax


def vis_impute_dist(
    original_df: pd.DataFrame,
    imputed_df: pd.DataFrame,
    column: str,
    ax=None,
    **kwargs,
):
    """Visualize the distribution of original and imputed data for a column.

    Parameters
    ----------
    original_df : pd.DataFrame
        The original dataframe with missing values.
    imputed_df : pd.DataFrame
        The dataframe after imputation.
    column : str
        The column to visualize.
    ax : matplotlib.axes.Axes, optional
        A matplotlib Axes object.  If not provided, a new one is created.
    **kwargs
        Additional keyword arguments forwarded to ``seaborn.kdeplot``.

    Returns
    -------
    matplotlib.axes.Axes
        The Axes containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    sns.kdeplot(original_df[column].dropna(), ax=ax, label='Original', **kwargs)
    sns.kdeplot(imputed_df[column], ax=ax, label='Imputed', **kwargs)
    ax.set_title(f"Distribution of Original vs. Imputed Data for {column}")
    ax.legend()
    return ax


def vis_miss_fct(
    df: pd.DataFrame,
    fct: str,
    ax=None,
    missing_values: list = None,
    **kwargs,
):
    """Visualize missingness by a factor (categorical) variable.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to visualize.
    fct : str
        Name of the categorical column to group by.
    ax : matplotlib.axes.Axes, optional
        A matplotlib Axes object.  If not provided, a new one is created.
    missing_values : list, optional
        A list of values to treat as missing in addition to ``NaN``.
    **kwargs
        Additional keyword arguments forwarded to ``DataFrame.plot``.

    Returns
    -------
    matplotlib.axes.Axes
        The Axes containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))

    null_indicator = _nullity(df, missing_values)
    miss_by_fct = (
        null_indicator
        .drop(columns=fct, errors="ignore")
        .groupby(df[fct])
        .sum()
    )

    miss_by_fct.plot(kind='bar', stacked=True, ax=ax, **kwargs)
    ax.set_title(f"Missing Values by {fct}")
    ax.set_xlabel(fct)
    ax.set_ylabel("Number of Missing Values")
    plt.tight_layout()
    return ax


def vis_miss_cumsum_var(
    df: pd.DataFrame,
    ax=None,
    missing_values: list = None,
    **kwargs,
):
    """Visualize the cumulative sum of missing values per variable.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to visualize.
    ax : matplotlib.axes.Axes, optional
        A matplotlib Axes object.  If not provided, a new one is created.
    missing_values : list, optional
        A list of values to treat as missing in addition to ``NaN``.
    **kwargs
        Additional keyword arguments forwarded to ``Series.plot``.

    Returns
    -------
    matplotlib.axes.Axes
        The Axes containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))

    miss_cumsum = _nullity(df, missing_values).sum().cumsum()

    miss_cumsum.plot(kind='line', ax=ax, **kwargs)
    ax.set_title("Cumulative Sum of Missing Values per Variable")
    ax.set_xlabel("Variables")
    ax.set_ylabel("Cumulative Sum of Missing Values")
    plt.tight_layout()
    return ax


def vis_miss_cumsum_case(
    df: pd.DataFrame,
    ax=None,
    missing_values: list = None,
    **kwargs,
):
    """Visualize the cumulative sum of missing values per case (row).

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to visualize.
    ax : matplotlib.axes.Axes, optional
        A matplotlib Axes object.  If not provided, a new one is created.
    missing_values : list, optional
        A list of values to treat as missing in addition to ``NaN``.
    **kwargs
        Additional keyword arguments forwarded to ``Series.plot``.

    Returns
    -------
    matplotlib.axes.Axes
        The Axes containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))

    miss_cumsum = _nullity(df, missing_values).sum(axis=1).cumsum()

    miss_cumsum.plot(kind='line', ax=ax, **kwargs)
    ax.set_title("Cumulative Sum of Missing Values per Case")
    ax.set_xlabel("Cases (Rows)")
    ax.set_ylabel("Cumulative Sum of Missing Values")
    plt.tight_layout()
    return ax


def vis_miss_span(
    df: pd.DataFrame,
    column: str,
    span: int,
    ax=None,
    missing_values: list = None,
    **kwargs,
):
    """Visualize the number of missings in a rolling span for a single variable.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to visualize.
    column : str
        The column to visualize.
    span : int
        Rolling window size.
    ax : matplotlib.axes.Axes, optional
        A matplotlib Axes object.  If not provided, a new one is created.
    missing_values : list, optional
        A list of values to treat as missing in addition to ``NaN``.
    **kwargs
        Additional keyword arguments forwarded to ``Series.plot``.

    Returns
    -------
    matplotlib.axes.Axes
        The Axes containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))

    miss_span = _nullity(df[[column]], missing_values)[column].rolling(span).sum()

    miss_span.plot(kind='line', ax=ax, **kwargs)
    ax.set_title(f"Missing Values in Spans of {span} for {column}")
    ax.set_xlabel("Index")
    ax.set_ylabel("Number of Missing Values in Span")
    plt.tight_layout()
    return ax


def vis_parallel_coords(df: pd.DataFrame, missing_values: list = None, **kwargs):
    """A parallel coordinates plot to visualize missingness patterns.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to visualize.
    missing_values : list, optional
        A list of values to treat as missing in addition to ``NaN``.
    **kwargs
        Additional keyword arguments forwarded to
        ``pandas.plotting.parallel_coordinates``.

    Returns
    -------
    matplotlib.axes.Axes
        The Axes containing the plot.
    """
    df_miss = _nullity(df, missing_values).astype(int)
    df_miss['missing_count'] = df_miss.sum(axis=1)

    fig, ax = plt.subplots(figsize=(12, 8))
    pd.plotting.parallel_coordinates(df_miss, 'missing_count', ax=ax, **kwargs)
    ax.set_title("Parallel Coordinates Plot of Missingness")
    plt.tight_layout()
    return ax


def dendrogram(
    df: pd.DataFrame,
    ax=None,
    missing_values: list = None,
    method: str = 'ward',
    **kwargs,
):
    """A dendrogram to cluster variables by their nullity correlation.

    Builds a hierarchical clustering of the DataFrame columns based on
    how similarly they are missing across rows.  Columns that tend to be
    missing together will appear close in the dendrogram.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to visualize.
    ax : matplotlib.axes.Axes, optional
        A matplotlib Axes object.  If not provided, a new one is created.
    missing_values : list, optional
        A list of values to treat as missing in addition to ``NaN``.
    method : str, optional
        Linkage method passed to ``scipy.cluster.hierarchy.linkage``.
        Default is ``'ward'``.
    **kwargs
        Additional keyword arguments forwarded to
        ``scipy.cluster.hierarchy.dendrogram``.

    Returns
    -------
    matplotlib.axes.Axes
        The Axes containing the plot.

    Raises
    ------
    ValueError
        If fewer than two columns have variable missingness patterns.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))

    nullity_matrix = _nullity(df, missing_values).astype(int)

    variable_cols = nullity_matrix.columns[nullity_matrix.var() > 0]
    if len(variable_cols) < 2:
        raise ValueError(
            "dendrogram requires at least 2 columns with variable missingness "
            "(i.e. not always-missing or never-missing). "
            f"Only {len(variable_cols)} such column(s) found."
        )
    nullity_matrix = nullity_matrix[variable_cols]

    corr_matrix = nullity_matrix.corr()
    distance_matrix = 1 - corr_matrix.abs()
    condensed_distances = squareform(distance_matrix.values, checks=False)
    linkage_matrix = linkage(condensed_distances, method=method)

    scipy_dendrogram(
        linkage_matrix,
        labels=variable_cols.tolist(),
        ax=ax,
        orientation='top',
        **kwargs,
    )

    ax.set_title("Dendrogram of Variables by Missing Data Patterns")
    ax.set_xlabel("Variables")
    ax.set_ylabel("Distance")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return ax


# ---------------------------------------------------------------------------
# New visualisations
# ---------------------------------------------------------------------------

def heatmap(
    df: pd.DataFrame,
    ax=None,
    missing_values: list = None,
    **kwargs,
):
    """Nullity correlation heatmap between columns.

    Shows the Pearson correlation of the binary missingness indicators
    for each pair of columns.  A value of +1 means the two columns are
    always missing together; -1 means they are never missing at the same
    time; 0 means missingness is independent.

    Inspired by ``missingno.heatmap``.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to visualize.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into.  Created automatically if omitted.
    missing_values : list, optional
        Sentinel values treated as missing in addition to ``NaN``.
    **kwargs
        Additional keyword arguments forwarded to ``seaborn.heatmap``.

    Returns
    -------
    matplotlib.axes.Axes
        The Axes containing the heatmap.

    Notes
    -----
    Columns that are never missing or always missing have zero variance
    and produce ``NaN`` correlations.  These cells are rendered as blank
    (masked) rather than raising an error.

    Example
    -------
    >>> import pandas as pd, numpy as np
    >>> from missingly import visualise
    >>> df = pd.DataFrame({'A': [np.nan, 1, np.nan], 'B': [np.nan, np.nan, 1]})
    >>> ax = visualise.heatmap(df)
    """
    if ax is None:
        n = df.shape[1]
        fig, ax = plt.subplots(figsize=(max(6, n), max(5, n - 1)))

    nullity_matrix = _nullity(df, missing_values).astype(float)

    corr = nullity_matrix.corr()
    mask = np.isnan(corr.values)

    cmap = kwargs.pop("cmap", "RdBu")
    annot = kwargs.pop("annot", True)
    fmt = kwargs.pop("fmt", ".2f")
    vmin = kwargs.pop("vmin", -1)
    vmax = kwargs.pop("vmax", 1)
    center = kwargs.pop("center", 0)
    linewidths = kwargs.pop("linewidths", 0.5)

    sns.heatmap(
        corr,
        mask=mask,
        ax=ax,
        cmap=cmap,
        annot=annot,
        fmt=fmt,
        vmin=vmin,
        vmax=vmax,
        center=center,
        linewidths=linewidths,
        **kwargs,
    )
    ax.set_title("Nullity Correlation Heatmap")
    plt.tight_layout()
    return ax


def vis_miss(
    df: pd.DataFrame,
    ax=None,
    missing_values: list = None,
    show_pct: bool = True,
    cluster: bool = False,
    **kwargs,
):
    """Annotated missingness matrix with per-column percentage labels.

    An enhanced version of :func:`matrix` that annotates each column
    header with its missingness percentage, making it easy to read
    both location and magnitude of missing data at a glance.

    Inspired by ``naniar::vis_miss``.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to visualize.  Column and index labels may contain
        Unicode characters (including Persian / Arabic).
    ax : matplotlib.axes.Axes, optional
        Axes to draw into.  Created automatically if omitted.
    missing_values : list, optional
        Sentinel values treated as missing in addition to ``NaN``.
    show_pct : bool, optional
        If ``True`` (default), append the missingness percentage to each
        column tick label, e.g. ``"salary (12.5%)"``.
    cluster : bool, optional
        If ``True``, reorder rows using hierarchical clustering on the
        missingness pattern so that rows with similar missing profiles
        are adjacent.  Default is ``False``.
    **kwargs
        Additional keyword arguments forwarded to ``seaborn.heatmap``.

    Returns
    -------
    matplotlib.axes.Axes
        The Axes containing the plot.

    Example
    -------
    >>> import pandas as pd, numpy as np
    >>> from missingly import visualise
    >>> df = pd.DataFrame({'نام': ['علی', None, 'رضا'], 'سن': [25, 30, None]})
    >>> ax = visualise.vis_miss(df)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(max(8, df.shape[1] * 1.2), 6))

    null_df = _nullity(df, missing_values).astype(float)

    if cluster and null_df.shape[0] > 1:
        from scipy.spatial.distance import pdist
        row_dist = pdist(null_df.values, metric="hamming")
        row_linkage = linkage(row_dist, method="ward")
        row_order = leaves_list(row_linkage)
        null_df = null_df.iloc[row_order]

    if show_pct:
        pct = _nullity(df, missing_values).mean() * 100
        col_labels = [
            f"{col} ({pct[col]:.1f}%)" for col in df.columns
        ]
    else:
        col_labels = list(df.columns.astype(str))

    cmap = kwargs.pop("cmap", ["#f0f0f0", "#d62728"])
    cbar = kwargs.pop("cbar", False)

    sns.heatmap(
        null_df,
        ax=ax,
        cmap=cmap,
        cbar=cbar,
        xticklabels=col_labels,
        yticklabels=False,
        **kwargs,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_title("Missing Data Overview")
    plt.tight_layout()
    return ax


def miss_var_pct(
    df: pd.DataFrame,
    ax=None,
    missing_values: list = None,
    sort: bool = True,
    **kwargs,
):
    """Horizontal bar chart of missingness percentage per variable.

    Each bar shows what percentage of that variable's values are
    missing.  Variables are sorted by missingness percentage (descending)
    by default.

    Inspired by ``naniar::miss_var_plot``.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to visualize.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into.  Created automatically if omitted.
    missing_values : list, optional
        Sentinel values treated as missing in addition to ``NaN``.
    sort : bool, optional
        If ``True`` (default), sort variables by descending missingness
        percentage before plotting.
    **kwargs
        Additional keyword arguments forwarded to
        ``matplotlib.axes.Axes.barh``.

    Returns
    -------
    matplotlib.axes.Axes
        The Axes containing the plot.

    Example
    -------
    >>> import pandas as pd, numpy as np
    >>> from missingly import visualise
    >>> df = pd.DataFrame({'درآمد': [None, 2, None, 4], 'سن': [1, None, 3, 4]})
    >>> ax = visualise.miss_var_pct(df)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, max(4, df.shape[1] * 0.5)))

    pct = _nullity(df, missing_values).mean() * 100

    if sort:
        pct = pct.sort_values(ascending=True)

    color = kwargs.pop("color", "steelblue")
    ax.barh(pct.index.astype(str), pct.values, color=color, **kwargs)
    ax.set_xlabel("% Missing")
    ax.set_xlim(0, 100)
    ax.axvline(x=0, color="black", linewidth=0.8)

    for i, (val, label) in enumerate(zip(pct.values, pct.index)):
        ax.text(
            val + 0.5, i, f"{val:.1f}%",
            va="center", ha="left", fontsize=8,
        )

    ax.set_title("Missing Values per Variable (%)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return ax


def miss_cluster(
    df: pd.DataFrame,
    ax=None,
    missing_values: list = None,
    method: str = "ward",
    **kwargs,
):
    """Heatmap with rows reordered by hierarchical clustering on missingness.

    Groups rows with similar missingness patterns together, making it
    easy to spot systematic missing data structures (e.g. an entire
    survey section skipped by one cohort).

    Inspired by ``missingno`` sorted matrix and ``naniar`` clustering.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to visualize.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into.  Created automatically if omitted.
    missing_values : list, optional
        Sentinel values treated as missing in addition to ``NaN``.
    method : str, optional
        Linkage method for ``scipy.cluster.hierarchy.linkage``.
        Default is ``'ward'``.
    **kwargs
        Additional keyword arguments forwarded to ``seaborn.heatmap``.

    Returns
    -------
    matplotlib.axes.Axes
        The Axes containing the heatmap.

    Notes
    -----
    Rows that are identical in their missingness pattern will be placed
    adjacent to each other.  The y-axis shows the *original* row index
    only when the DataFrame has fewer than 50 rows; otherwise it is
    suppressed for readability.

    When all rows have identical missingness patterns (zero variance in
    the Hamming distance), clustering is skipped and rows are shown in
    original order.

    Example
    -------
    >>> import pandas as pd, numpy as np
    >>> from missingly import visualise
    >>> rng = np.random.default_rng(0)
    >>> df = pd.DataFrame(rng.choice([1.0, np.nan], size=(20, 4)),
    ...                   columns=list('ABCD'))
    >>> ax = visualise.miss_cluster(df)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(max(8, df.shape[1] * 1.2), 6))

    null_df = _nullity(df, missing_values).astype(float)

    if null_df.shape[0] > 1:
        from scipy.spatial.distance import pdist
        row_dist = pdist(null_df.values, metric="hamming")
        if row_dist.max() > 0:
            row_linkage = linkage(row_dist, method=method)
            row_order = leaves_list(row_linkage)
            null_df = null_df.iloc[row_order]

    yticklabels = (
        list(null_df.index.astype(str))
        if null_df.shape[0] < 50
        else False
    )

    cmap = kwargs.pop("cmap", ["#f0f0f0", "#d62728"])
    cbar = kwargs.pop("cbar", False)

    sns.heatmap(
        null_df,
        ax=ax,
        cmap=cmap,
        cbar=cbar,
        yticklabels=yticklabels,
        **kwargs,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_title("Clustered Missing Data Matrix")
    plt.tight_layout()
    return ax


def miss_which(
    df: pd.DataFrame,
    ax=None,
    missing_values: list = None,
    **kwargs,
):
    """Binary tile plot showing which variables contain missing values.

    Each column is represented by a single tile.  The tile is filled
    (dark) if the variable has **any** missing values, and empty (light)
    if it is fully observed.

    Inspired by ``naniar::miss_which`` and ``naniar::vis_miss``.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to visualize.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into.  Created automatically if omitted.
    missing_values : list, optional
        Sentinel values treated as missing in addition to ``NaN``.
    **kwargs
        Additional keyword arguments forwarded to ``seaborn.heatmap``.

    Returns
    -------
    matplotlib.axes.Axes
        The Axes containing the plot.

    Notes
    -----
    The underlying heatmap data is a (1 x n_columns) numeric DataFrame
    with dtype ``float64``.  Annotation labels ("Missing" / "Complete")
    are stored in a separate object-dtype array passed via the ``annot``
    parameter to avoid the pandas FutureWarning about incompatible dtype
    assignment.

    Example
    -------
    >>> import pandas as pd, numpy as np
    >>> from missingly import visualise
    >>> df = pd.DataFrame({'A': [1, None], 'B': [1, 1], 'ستون': [None, None]})
    >>> ax = visualise.miss_which(df)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(max(6, df.shape[1] * 0.8), 2.5))

    null_df = _nullity(df, missing_values)
    # 1-row numeric DataFrame: 1.0 = has missing, 0.0 = fully observed.
    has_missing = null_df.any().astype(float).to_frame(name="has_missing").T
    pct = null_df.mean() * 100

    col_labels = [
        f"{col}\n({pct[col]:.1f}%)" for col in df.columns
    ]

    # Build annotation array as object dtype to avoid dtype-incompatibility
    # FutureWarning when mixing strings into a float DataFrame.
    annot_arr = np.where(
        has_missing.values.astype(bool),
        "Missing",
        "Complete",
    )

    cmap = kwargs.pop("cmap", ["#f0f0f0", "#d62728"])
    cbar = kwargs.pop("cbar", False)

    sns.heatmap(
        has_missing,
        ax=ax,
        cmap=cmap,
        cbar=cbar,
        annot=annot_arr,
        fmt="",
        xticklabels=col_labels,
        yticklabels=False,
        **kwargs,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_title("Which Variables Have Missing Data?")
    plt.tight_layout()
    return ax
