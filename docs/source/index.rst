.. missingly documentation master file, created by
   sphinx-quickstart on Mon Sep  8 07:52:43 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

missingly: Comprehensive Missing Data Analysis
==============================================

**missingly** is a Python package for comprehensive analysis, visualization, and imputation of missing data. Inspired by the R `naniar` package, it provides intuitive tools for understanding and handling missing data patterns in pandas DataFrames.

Features
--------

* **Summary Statistics**: Quick overviews of missing data patterns
* **Rich Visualizations**: Matrix plots, bar charts, dendrograms, upset plots, and more  
* **Statistical Tests**: MCAR tests and missing data mechanism analysis
* **Multiple Imputation Methods**: From simple mean imputation to advanced MICE
* **Automated Reporting**: Generate comprehensive HTML reports
* **Custom Missing Values**: Handle non-standard missing indicators

Quick Start
-----------

.. code-block:: python

   import pandas as pd
   import numpy as np
   import missingly as mi

   # Create sample data
   data = {'A': [1, np.nan, 3], 'B': [4, 5, np.nan]}
   df = pd.DataFrame(data)

   # Analyze missing patterns
   mi.miss_var_summary(df)
   mi.matrix(df)
   mi.dendrogram(df)

   # Generate report
   mi.create_report(df, "report.html")

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Documentation:

   api/index
   examples/index

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
