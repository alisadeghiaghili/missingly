"""Visualization utilities for missing data analysis.

This module provides a collection of plotting functions that work on
pandas DataFrames.  Every function follows the same convention:

* accepts an optional ``ax`` (matplotlib Axes) so callers can embed
  plots inside existing figures;
* accepts an optional ``missing_values`` list to treat arbitrary
  sentinel values (e.g. ``-99``, ``"N/A"``) as missing;
* returns the Axes object (or a dict of Axes for multi-panel plots)
  so callers can further customise the output.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram as scipy_dendrogram
from scipy.spatial.distance import squareform


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

    if missing_values is None:
        nullity_matrix = df.isnull()
    else:
        nullity_matrix = df.isin(missing_values)

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

    if missing_values is None:
        miss_counts = df.isnull().sum()
    else:
        miss_counts = df.isin(missing_values).sum()

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

    Notes
    -----
    The previous implementation delegated to ``upsetplot.UpSet``, which
    calls ``fillna(inplace=True)`` on a DataFrame slice.  Under
    pandas >= 3.0 Copy-on-Write this is a silent no-op, leaving ``NaN``
    values in matplotlib colour arrays and raising
    ``ValueError: Invalid RGBA argument: nan`` on Python 3.11/3.12.

    Example
    -------
    >>> import pandas as pd
    >>> from missingly import visualise
    >>> df = pd.DataFrame({'A': [None, 1, None], 'B': [None, None, 1]})
    >>> axes = visualise.upset(df)
    >>> isinstance(axes, dict)
    True
    """
    if missing_values is None:
        nullity_matrix = df.isnull()
    else:
        nullity_matrix = df.isin(missing_values)

    missing_cols = list(nullity_matrix.columns[nullity_matrix.any()])
    if not missing_cols:
        print("No missing values to plot.")
        return {}

    nullity_matrix = nullity_matrix[missing_cols].astype(bool)
    n_cols = len(missing_cols)

    # --- build combination counts -----------------------------------------
    # Represent each row as a tuple of booleans, count occurrences.
    combos: dict = {}
    for row in nullity_matrix.itertuples(index=False):
        key = tuple(row)
        combos[key] = combos.get(key, 0) + 1

    # Only keep combinations where at least one column is missing.
    combos = {k: v for k, v in combos.items() if any(k)}
    if not combos:
        print("No missing combinations to plot.")
        return {}

    # Sort by count descending (cardinality order).
    sorted_combos = sorted(combos.items(), key=lambda x: x[1], reverse=True)
    combo_keys = [c[0] for c in sorted_combos]
    combo_counts = [c[1] for c in sorted_combos]
    n_combos = len(combo_keys)

    col_totals = [nullity_matrix[c].sum() for c in missing_cols]

    # --- layout -------------------------------------------------------------
    # GridSpec: 2 rows × 2 cols
    #   [0, 1]  intersection bar chart   (top-right)
    #   [1, 0]  column-totals h-bar      (bottom-left)
    #   [1, 1]  dot-matrix               (bottom-right)
    fig = plt.figure(figsize=(max(8, n_combos * 1.2), max(6, n_cols * 0.9 + 3)))
    gs = gridspec.GridSpec(
        2, 2,
        width_ratios=[1, n_combos],
        height_ratios=[2, n_cols],
        hspace=0.05,
        wspace=0.05,
    )
    ax_bar   = fig.add_subplot(gs[0, 1])          # intersection sizes
    ax_mat   = fig.add_subplot(gs[1, 1])          # dot matrix
    ax_tot   = fig.add_subplot(gs[1, 0])          # column totals
    fig.add_subplot(gs[0, 0]).set_visible(False)  # empty corner

    x_pos = np.arange(n_combos)
    y_pos = np.arange(n_cols)

    # --- top bar chart (intersection sizes) --------------------------------
    ax_bar.bar(x_pos, combo_counts, color="steelblue", edgecolor="white")
    ax_bar.set_xlim(-0.5, n_combos - 0.5)
    ax_bar.set_xticks([])
    ax_bar.set_ylabel("Intersection size")
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)

    # --- dot-and-line matrix -----------------------------------------------
    dot_color_on  = "#333333"
    dot_color_off = "#cccccc"

    for xi, key in enumerate(combo_keys):
        # Vertical connector between the topmost and bottommost active dots.
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

    # --- left horizontal bar chart (per-column totals) ---------------------
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


def scatter_miss(df: pd.DataFrame, x: str, y: str, ax=None, missing_values: list = None, **kwargs):
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

    if missing_values is None:
        plot_df[f'{x}_NA'] = plot_df[x].isnull()
        plot_df[f'{y}_NA'] = plot_df[y].isnull()
    else:
        plot_df[f'{x}_NA'] = plot_df[x].isin(missing_values)
        plot_df[f'{y}_NA'] = plot_df[y].isin(missing_values)

    any_x_na = plot_df[f'{x}_NA'].any()
    any_y_na = plot_df[f'{y}_NA'].any()

    if any_x_na:
        x_min = plot_df[x].min()
        plot_df[x] = plot_df[x].fillna(x_min - (abs(x_min) * 0.1))
    if any_y_na:
        y_min = plot_df[y].min()
        plot_df[y] = plot_df[y].fillna(y_min - (abs(y_min) * 0.1))

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

    if missing_values is None:
        miss_counts = df.isnull().sum(axis=1)
    else:
        miss_counts = df.isin(missing_values).sum(axis=1)

    miss_counts.plot(kind='bar', ax=ax, **kwargs)
    ax.set_title("Missing Values per Case")
    ax.set_xlabel("Cases (Rows)")
    ax.set_ylabel("Number of Missing Values")
    plt.tight_layout()
    return ax


def vis_impute_dist(original_df: pd.DataFrame, imputed_df: pd.DataFrame, column: str, ax=None, **kwargs):
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


def vis_miss_fct(df: pd.DataFrame, fct: str, ax=None, missing_values: list = None, **kwargs):
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

    if missing_values is None:
        miss_by_fct = df.drop(columns=fct).groupby(df[fct]).apply(lambda x: x.isnull().sum())
    else:
        miss_by_fct = df.drop(columns=fct).groupby(df[fct]).apply(lambda x: x.isin(missing_values).sum())

    miss_by_fct.plot(kind='bar', stacked=True, ax=ax, **kwargs)
    ax.set_title(f"Missing Values by {fct}")
    ax.set_xlabel(fct)
    ax.set_ylabel("Number of Missing Values")
    plt.tight_layout()
    return ax


def vis_miss_cumsum_var(df: pd.DataFrame, ax=None, missing_values: list = None, **kwargs):
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

    if missing_values is None:
        miss_cumsum = df.isnull().sum().cumsum()
    else:
        miss_cumsum = df.isin(missing_values).sum().cumsum()

    miss_cumsum.plot(kind='line', ax=ax, **kwargs)
    ax.set_title("Cumulative Sum of Missing Values per Variable")
    ax.set_xlabel("Variables")
    ax.set_ylabel("Cumulative Sum of Missing Values")
    plt.tight_layout()
    return ax


def vis_miss_cumsum_case(df: pd.DataFrame, ax=None, missing_values: list = None, **kwargs):
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

    if missing_values is None:
        miss_cumsum = df.isnull().sum(axis=1).cumsum()
    else:
        miss_cumsum = df.isin(missing_values).sum(axis=1).cumsum()

    miss_cumsum.plot(kind='line', ax=ax, **kwargs)
    ax.set_title("Cumulative Sum of Missing Values per Case")
    ax.set_xlabel("Cases (Rows)")
    ax.set_ylabel("Cumulative Sum of Missing Values")
    plt.tight_layout()
    return ax


def vis_miss_span(df: pd.DataFrame, column: str, span: int, ax=None, missing_values: list = None, **kwargs):
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

    if missing_values is None:
        miss_span = df[column].isnull().rolling(span).sum()
    else:
        miss_span = df[column].isin(missing_values).rolling(span).sum()

    miss_span.plot(kind='line', ax=ax, **kwargs)
    ax.set_title(f"Missing Values in Spans of {span} for {column}")
    ax.set_xlabel("Index")
    ax.set_ylabel(f"Number of Missing Values in Span")
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
    if missing_values is None:
        df_miss = df.isnull().astype(int)
    else:
        df_miss = df.isin(missing_values).astype(int)

    df_miss['missing_count'] = df_miss.sum(axis=1)

    fig, ax = plt.subplots(figsize=(12, 8))
    pd.plotting.parallel_coordinates(df_miss, 'missing_count', ax=ax, **kwargs)
    ax.set_title("Parallel Coordinates Plot of Missingness")
    plt.tight_layout()
    return ax


def dendrogram(df: pd.DataFrame, ax=None, missing_values: list = None, method='ward', **kwargs):
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
        Linkage method passed to
        ``scipy.cluster.hierarchy.linkage``.  One of ``'ward'``,
        ``'single'``, ``'complete'``, ``'average'``, ``'weighted'``,
        ``'centroid'``, ``'median'``.  Default is ``'ward'``.
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
        If fewer than two columns have variable missingness patterns
        (i.e. all columns are either always-missing or never-missing),
        making clustering impossible.

    Notes
    -----
    Columns with zero variance in their nullity indicator (always-missing
    or never-missing) are dropped before computing the correlation matrix.
    Such columns produce ``NaN`` in the Pearson correlation and would make
    the distance matrix non-finite, causing scipy linkage to fail.

    The distance between two columns is defined as
    ``1 - |correlation of their missingness indicators|``.
    ``scipy.spatial.distance.squareform`` converts the square distance
    matrix to the condensed 1-D vector required by ``linkage``.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))

    if missing_values is None:
        nullity_matrix = df.isnull().astype(int)
    else:
        nullity_matrix = df.isin(missing_values).astype(int)

    # Drop columns with zero variance: always-missing or never-missing columns
    # produce NaN in the Pearson correlation, making the distance matrix
    # non-finite and causing linkage() to raise ValueError.
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

    # squareform converts a symmetric square distance matrix to the
    # condensed 1-D vector that linkage() expects.
    condensed_distances = squareform(distance_matrix.values, checks=False)

    linkage_matrix = linkage(condensed_distances, method=method)

    scipy_dendrogram(
        linkage_matrix,
        labels=variable_cols.tolist(),
        ax=ax,
        orientation='top',
        **kwargs
    )

    ax.set_title("Dendrogram of Variables by Missing Data Patterns")
    ax.set_xlabel("Variables")
    ax.set_ylabel("Distance")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return ax
