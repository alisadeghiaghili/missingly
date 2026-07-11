"""missingly.utils — internal utilities not part of the public missing-data API.

This subpackage contains helper modules that are out of scope for
``missingly``'s core charter (missingness analysis, imputation,
comparison, reporting) but are kept here for backward compatibility
during the deprecation window.

Public surface
--------------
Do **not** import from this subpackage in new user code.
Use the canonical ``missingly`` public API instead.

Deprecation timeline
--------------------
* ``utils.manipulation`` symbols deprecated in v0.3.0, removed in v0.4.0.
  Point users to ``data_quality_toolkit`` for data-cleaning utilities.
"""
