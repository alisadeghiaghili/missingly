Polars
======

The optional Polars APIs operate natively: they do not silently convert input
to pandas. Install support with ``pip install missingly[polars]``.

``polars_missing_summary`` accepts a Polars ``DataFrame`` or ``LazyFrame`` and
returns the same execution mode. It counts only native Polars nulls; sentinel
value handling remains an explicit pandas-oriented workflow for now.

.. automodule:: missingly.polars_adapter
   :members:
