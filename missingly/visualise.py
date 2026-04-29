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
import seaborn as sns
from upsetplot import UpSet, from_memberships
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

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to visualize.
    missing_values : list, optional
        A list of values to treat as missing in addition to ``NaN``.
    **kwargs
        Additional keyword arguments forwarded to ``upsetplot.UpSet``.

    Returns
    -------
    dict
        A dict of matplotlib Axes objects returned by ``UpSet.plot()``.

    Notes
    -----
    Uses ``upsetplot.from_memberships()`` instead of a manual
    ``groupby``-based MultiIndex.  The ``groupby`` approach produces
    ``NaN`` entries in the index under pandas >= 2 for unobserved
    combinations, which propagate into upsetplot's internal colour
    arrays and cause ``ValueError: Invalid RGBA argument: nan`` in
    matplotlib >= 3.8.  ``from_memberships`` is the canonical upsetplot
    API for building a series from a boolean membership matrix and
    avoids this issue entirely.
    """
    if missing_values is None:
        nullity_matrix = df.isnull()
    else:
        nullity_matrix = df.isin(missing_values)

    missing_cols = nullity_matrix.columns[nullity_matrix.any()]
    if missing_cols.empty:
        print("No missing values to plot.")
        return

    nullity_matrix = nullity_matrix[missing_cols].astype(bool)

    # Build memberships: for each row, the list of columns that ARE missing.
    # from_memberships is the correct upsetplot API — it never produces NaN
    # in the resulting Series index, unlike a manual groupby on bool columns.
    memberships = [
        [col for col in missing_cols if row[col]]
        for _, row in nullity_matrix.iterrows()
    ]
    upset_data = from_memberships(memberships)
    return UpSet(upset_data, sort_by='cardinality', **kwargs).plot()


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
