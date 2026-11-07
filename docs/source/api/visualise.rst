Visualization Functions
=======================

Static functions return the :class:`matplotlib.axes.Axes` they drew on and
accept an optional ``ax`` keyword. ``upset`` returns a dict of axes,
``vis_miss_fct`` has no ``ax`` and returns the last facet's axes, and
``miss_impute_compare`` returns a :class:`matplotlib.figure.Figure`.

Only ``vis_miss``, ``heatmap``, ``matrix``, ``bar``, ``miss_var_pct``,
``miss_cooccurrence``, ``miss_case``, ``upset`` and ``miss_patterns`` have a
Plotly backend (``backend="plotly"`` or ``interactive=True``). Styling
arguments the Plotly backend does not honour are reported with a warning.

Keyword arguments a function does not support are ignored but emit a
``FutureWarning`` naming the argument; a future release will raise
``TypeError``. ``upset`` already raises ``TypeError``.

.. automodule:: missingly.visualise
   :members:
