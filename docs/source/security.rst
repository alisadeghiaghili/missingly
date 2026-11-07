Security notes
==============

This page describes what the ``missingly`` code in this repository does with
respect to output escaping, network access, random numbers and hashing. Each
statement refers to the code or test that backs it.

HTML report escaping
--------------------

:func:`missingly.report.create_report` renders its HTML with a Jinja2
``Environment`` created with ``autoescape=True``. Every value taken from the
input DataFrame or from the caller (column names, row index labels, the
report ``title``) is inserted through a ``{{ ... }}`` expression, so HTML
characters are escaped. The template contains no ``|safe`` filter.

The tests in ``tests/test_report_security.py`` check that:

* the environment used by ``create_report`` has ``autoescape`` enabled, and
* column names, row index labels and the title that contain HTML appear in the
  output only in escaped form.

Plot images are embedded as base64 PNG data URIs produced by matplotlib, so
they do not add raw HTML from the data.

Network access
--------------

The ``missingly`` package does not import any networking module (``urllib``,
``socket``, ``http``, ``requests`` and similar). Generating a report therefore
makes no network calls; it reads the bundled template from disk and writes
one file.

The generated HTML is not fully offline. Its ``<head>`` links a Google Fonts
stylesheet, so a browser that opens the report requests
``fonts.googleapis.com`` and ``fonts.gstatic.com``.

Random numbers
--------------

``report.py`` uses no random numbers.

Across the package, random generators are created with
``numpy.random.default_rng``. The shared seed rule
``missingly._validation.as_generator`` accepts an integer, ``None``, a
``numpy.random.Generator`` or a ``numpy.random.RandomState``. For a
``RandomState`` it draws one seed from it and builds a ``Generator``. The
package does not call ``numpy.random.seed`` or the standard library ``random``
module.

Hashing
-------

SHA-256 is the only digest used in the package:

* :func:`missingly.provenance.dataframe_fingerprint` returns the SHA-256 hex
  digest of a canonical JSON serialisation of a DataFrame, its operation and
  its parameters.
* :meth:`missingly.benchmark.BenchmarkManifest.sha256` returns the SHA-256 hex
  digest of the manifest's JSON form.

The HTML report does not compute any fingerprint.
