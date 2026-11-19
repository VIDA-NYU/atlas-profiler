Atlas Profiler
==============

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License: MIT

.. image:: https://img.shields.io/badge/python-3.10+-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python 3.10+

.. image:: https://img.shields.io/pypi/v/atlas-profiler.svg
   :target: https://pypi.org/project/atlas-profiler/
   :alt: PyPI

.. image:: https://img.shields.io/badge/github-VIDA--NYU%2Fatlas--profiler-brightgreen.svg
   :target: https://github.com/VIDA-NYU/atlas-profiler
   :alt: GitHub

Atlas Profiler is a dataset profiling library. Given a CSV/TSV path, a file-like object, or a pandas DataFrame, it returns JSON-style metadata about the dataset: its columns and their types, statistics and histograms, where the data is in space and in time, and the time each profiling step took.

The package builds on the Datamart Profiler workflow and adds an ML-assisted spatial column classifier. That classifier is only one part of the profiler: non-spatial columns still go through the core rule-based type detection, statistics, plots, coverage, and dataset-summary pipeline.

What It Produces
----------------

``process_dataset(...)`` returns a metadata dictionary with fields such as:

- Dataset size, row count, profiled row count, and column count.
- Per-column structural type, semantic types, missing/unclean value ratios, distinct counts (exact when small), and optional histograms.
- Dataset-level type summary: numerical, categorical, spatial, and temporal.
- Spatial coverage from lat/long pairs, WKT points, resolved addresses, and administrative areas. A lat/long pair gets a robust bounding box, geohash cells, a median-split k-d tree, and an exact density grid of square cells.
- Temporal coverage for date and time columns: extent, precision, resolution, and spacing (typical interval, regularity, largest gap).
- ``computed_on``: which parts come from the profiled random sample and which from every row.
- Attribute keywords derived from column names.
- Optional random sample rows and per-step profiling timings.

Rows are profiled on a seeded random sample whose size adapts to the table (about 2 million values, at least 10,000 rows). A CSV/TSV path is streamed with polars, so distinct counts, numeric statistics, and spatial and temporal coverage are computed over every row of the file.

Core Type System
----------------

The profiler detects broad structural types for all columns:

.. list-table:: Structural Types
   :header-rows: 1

   * - Structural type
     - Meaning
   * - ``MissingData``
     - Empty column.
   * - ``Integer``
     - Integer-like values.
   * - ``Float``
     - Floating point values.
   * - ``Text``
     - String/text values.
   * - ``Boolean``
     - Boolean-like values such as true/false, yes/no, 0/1.
   * - ``GeoCoordinates``
     - Point geometry or coordinate-pair strings.
   * - ``GeoShape``
     - Polygon-like geometry.

It also annotates semantic types when evidence is available:

.. list-table:: Semantic Types
   :header-rows: 1

   * - Semantic type
     - Examples
   * - ``DateTime``
     - Dates, timestamps, and year columns.
   * - ``latitude``, ``longitude``
     - Coordinate columns, paired after profiling.
   * - ``address``, ``AdministrativeArea``
     - Address-like and admin-area text, optionally resolved with Nominatim or ``datamart_geo``.
   * - ``URL``, ``FileName``, ``identifier``, ``Enumeration``
     - URLs, file paths, IDs, and categorical columns.

Spatial ML Classifier
---------------------

When ``geo_classifier=True``, Atlas Profiler creates a ``HybridGeoClassifier(GeoClassifier())``. It samples values from each column, predicts spatial labels in one batch, validates sensitive predictions with rules, and passes accepted labels into the normal profiler type system.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   examples
   profiler-changes
   report
   api
   changelog

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

