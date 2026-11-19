Examples
========

Basic Usage
-----------

.. code-block:: python

   import json
   from atlas_profiler import process_dataset

   # A CSV/TSV path is streamed: statistics and coverage use every row of the file
   metadata = process_dataset("trips.csv", plots=True)
   json.dump(metadata, open("trips.profile.json", "w"), indent=2, default=str)

   print(metadata["nb_rows"], "rows,", metadata["nb_profiled_rows"], "profiled")
   print(metadata["computed_on"])   # which parts come from the sample, which from every row

Spatial coverage
----------------

``spatial_coverage`` is a list with one entry per lat/long pair (and per WKT point column, address
column, or administrative-area column).

.. code-block:: python

   for entry in metadata.get("spatial_coverage", []):
       if entry["type"] != "latlong":
           continue
       (min_long, max_lat), (max_long, min_lat) = entry["ranges"][0]["range"]["coordinates"]
       print(entry["column_names"], f"lon {min_long:.3f}..{max_long:.3f}, lat {min_lat:.3f}..{max_lat:.3f}")
       print("  points:", entry["number"], "at (0, 0):", entry.get("number_zero_coordinates", 0))

       grid = entry["grid"]          # exact counts in square cells
       print(f"  grid of {grid['columns']} x {grid['rows']} cells of {1000 * grid['cell_km']:.0f} m,",
             len(grid["cells"]), "non-empty")
       x0, y0 = grid["bounds"][:2]
       dx, dy = grid["cell_degrees"]
       column, row, count = max(grid["cells"], key=lambda c: c[2])
       print(f"  densest cell: {count} points at lon {x0 + (column + 0.5) * dx:.4f}, lat {y0 + (row + 0.5) * dy:.4f}")

       leaves = entry["kdtree"]["leaves"]   # each holds about the same number of points
       print("  k-d tree:", len(leaves), "leaves of about", leaves[0]["number"], "points")

Temporal coverage
-----------------

.. code-block:: python

   from datetime import datetime, timezone

   for entry in metadata.get("temporal_coverage", []):
       span = entry["ranges"][0]["range"]
       start = datetime.fromtimestamp(span["gte"], tz=timezone.utc)
       end = datetime.fromtimestamp(span["lte"], tz=timezone.utc)
       print(entry["column_names"], start, "to", end, "precision:", entry["temporal_precision"])
       sampling = entry.get("sampling")   # present for CSV paths and unsampled DataFrames
       if sampling:
           print("  median interval", sampling["median_interval_seconds"], "s, largest gap",
                 sampling["largest_gap_seconds"], "s, regular:", sampling["regular"])

Getting Started with NYC Traffic Data
-------------------------------------

.. code-block:: python

   from atlas_profiler import process_dataset
   import pandas as pd

   # Load sample NYC traffic volume data
   nyc_traffic_volume_sample_df = pd.read_csv(
       "hf://datasets/oscur/automated-traffic-volume-counts-sample/sample_traffic.csv"
   )

   nyc_traffic_volume_sample_df.head()

.. code-block:: python

   # Profile the dataset with geo classifier enabled
   profiled_columns = process_dataset(
       nyc_traffic_volume_sample_df,
       coverage=False,
       indexes=False,
       geo_classifier=True,
       load_max_size=1000000,   # profile at most 1 MB of in-memory data
   )

   profiled_columns

Output Structure
================

The ``process_dataset`` function returns a dictionary of metadata about the profiled dataset. The main fields:

**Dataset-level Information:**

- ``size``: Size of the input file in bytes (for paths and file objects)
- ``nb_rows``: Total number of rows in the dataset
- ``nb_profiled_rows``: Number of rows in the random sample that was profiled
- ``nb_columns``: Total number of columns
- ``nb_spatial_columns``, ``nb_temporal_columns``, ``nb_categorical_columns``, ``nb_numerical_columns``: Number of columns of each kind
- ``types``: Dataset-level type categories present (e.g., ``["categorical", "numerical", "spatial", "temporal"]``)
- ``computed_on``: How the profiled rows were chosen (``sampling``: ``method`` ``random`` or ``none``, ``seed``, ``size_rule`` ``adaptive`` or ``load_max_size``, ``rows`` profiled ``of`` all), and for each part of the profile (``column_types``, ``value_ranges``, ``histograms``, ``distinct_counts``, ``numeric_statistics``, ``spatial_coverage``, ``temporal_coverage``, ``temporal_resolution``) whether it comes from the ``sample``, from ``all_rows``, or is ``mixed``. Each spatial coverage entry has its own ``computed_on`` (lat/long pairs of a CSV path come from every row; WKT points, addresses, and administrative areas from the sample), and each temporal entry says where its ``ranges``, ``sampling``, ``temporal_resolution``, and ``temporal_precision`` come from

**Column-level Information:**

- ``columns``: List of column metadata dictionaries, each containing:

  - ``name``: Column name, as in the file
  - ``structural_type``: Detected structural type using schema.org URLs (e.g., ``"http://schema.org/Integer"``, ``"http://schema.org/Text"``, ``"http://schema.org/GeoCoordinates"``)
  - ``semantic_types``: List of semantic type annotations using schema.org URLs (e.g., ``["http://schema.org/identifier"]``, ``["http://schema.org/AdministrativeArea"]``, ``["http://schema.org/address"]``)
  - ``missing_values_ratio``: Ratio of empty values (0.0 to 1.0), when there are any
  - ``unclean_values_ratio``: Ratio of values that do not match the detected type (0.0 to 1.0)
  - ``num_distinct_values``: Number of distinct values; ``num_distinct_values_is_approximate`` says whether the count is an estimate (counts of up to 10,000 values in a CSV are exact)
  - ``mean``, ``stddev``, ``min``, ``max``: For numbers
  - ``coverage``: Value ranges (up to 3 clusters of values, outer 5% trimmed)
  - ``plot`` (with ``plots=True``): A histogram. ``histogram_numerical`` bins a robust range and counts values ``below`` and ``above`` it, or has one bar per value (``discrete``) for small integer domains; ``histogram_temporal`` uses calendar bins (``unit``); ``histogram_categorical`` and ``histogram_text`` list the most frequent values or words
  - ``geo_classifier`` (optional): Geo classifier results for spatial columns, containing:

    - ``label``: Predicted spatial label (e.g., ``"borough"``, ``"point"``, ``"address"``)
    - ``confidence``: Confidence score (0.0 to 1.0)
    - ``source``: Source of the classification (e.g., ``"ml+validated"``, ``"ml"``)

**Coverage:**

- ``spatial_coverage``: One entry per spatial column or lat/long pair; a lat/long pair has ``ranges``, ``geohashes4``, ``kdtree``, ``grid`` and point counts (see `Spatial coverage`_)
- ``temporal_coverage``: One entry per date or time column, with ``ranges``, ``temporal_resolution``, ``temporal_precision`` and ``sampling`` (see `Temporal coverage`_)

**Additional Metadata:**

- ``attribute_keywords``: Keywords from the column names, stripped and without duplicates
- ``_profiling_times``: Performance timing information:

  - ``steps``: Dictionary of timing for each profiling step (in seconds)
  - ``total``: Total profiling time (in seconds)
