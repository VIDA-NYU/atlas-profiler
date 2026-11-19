Changelog
=========

Version 0.1.0
-------------

- New HTML report of a profile, drawn with Autark (``profiler.report``, ``atlas-profiler-report``); see :doc:`report`
- CSV paths are sampled at random across the file, not from its first rows; the sample size adapts to the table (``profiled_rows``)
- ``computed_on`` records which parts of a profile come from the sample and which from every row
- Fast temporal min/max with polars, checked against the exact parser; support for polars 2
- Spatial coverage: (0, 0) placeholders counted apart, a 0.1-99.9% quantile bounding box, geohash cells built inside it with numpy, a median-split k-d tree (``kdtree``) and an exact density grid (``grid``)
- Robust histograms (``profiler.histograms``): robust range with counts below and above, one bar per value for small integer domains, calendar bins for times
- Temporal ``temporal_precision`` and ``sampling`` (interval, regularity, gaps)
- Values are stripped before type detection; hashed identifiers are detected; small distinct counts are exact; keywords are stripped and deduplicated
- PyTorch is imported only when the spatial classifier runs; a clear error when it is requested but not installed
- ``full_statistics``: histograms over every row
- The spatial classifier's diagnostics file is opt-in (``diagnostics_path``); logs no longer include data values
- Per-entry provenance (``computed_on``) for spatial and temporal coverage, and the sampling method, seed and size rule
- Inputs: ``os.PathLike`` paths, a ``delimiter`` option, string and unique DataFrame column labels; dates with no parseable values are skipped; NaN and infinity never reach the profile
- ``benchmarks/csv_column_stats.py``: time and memory of the whole-file column statistics, by columns per scan

See ``docs/profiler-changes.md`` for before/after comparisons.

Version 0.0.2
---------------

- Improved geoclassifier model, make the performance better

Version 0.0.1b0
---------------

- Adopt from DatamartProfiler
- Introduce GeoClassifier, trained from contrastive learning on synthetic data and hard negatives
- Improve the performances on CPU and GPU