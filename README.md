# atlas-profiler

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/pypi/v/atlas-profiler.svg)](https://pypi.org/project/atlas-profiler/)
[![Documentation](https://img.shields.io/badge/docs-ReadTheDocs-blue.svg)](https://atlas-profiler.readthedocs.io/en/latest/)
[![GitHub](https://img.shields.io/badge/github-VIDA--NYU%2Fatlas--profiler-brightgreen.svg)](https://github.com/VIDA-NYU/atlas-profiler)

**Atlas Profiler** is a dataset profiling library that detects and annotates data types, including spatial and temporal ones. Given a CSV/TSV path, a file-like object or a pandas DataFrame, it returns JSON-style metadata about the dataset: its columns and their types, statistics and histograms, where the data is in space (a bounding box, a k-d tree and a density grid of each lat/long pair) and in time (extent, precision and spacing of each date column), and the time each profiling step took.

## Quick Start

### Installation

Install from PyPI:

```bash
pip install atlas-profiler
```

Or install from source for development:

```bash
git clone https://github.com/VIDA-NYU/atlas-profiler.git
cd atlas-profiler
pip install -e .
```

Requirements: Python 3.10+, and polars 1.44 or later (2.x works). The spatial classifier uses PyTorch and transformers; its model files are downloaded on first use (see [Model Files](#model-files)).

### Basic Usage

```python
import json
from atlas_profiler import process_dataset

# Profile a CSV file; plots=True adds a histogram per column
metadata = process_dataset("data.csv", plots=True)
json.dump(metadata, open("data.profile.json", "w"), indent=2, default=str)

# Or profile a pandas DataFrame
import pandas as pd
df = pd.read_csv("data.csv")
metadata = process_dataset(df, geo_classifier=True, coverage=True)
```

Prefer a path for large files: the profiler streams a CSV/TSV path with polars, so distinct counts, numeric statistics and spatial and temporal coverage come from every row. Column types and histograms always come from a random sample of the rows, whose size adapts to the table (see `load_max_size` below). The profile's `computed_on` says which part comes from where.

## Documentation

- **[Complete Documentation](https://atlas-profiler.readthedocs.io/en/latest/)**: guides, API reference, examples and configuration.
- **[How the profiler changed](docs/profiler-changes.md)**: sampling, speed, histograms, and spatial and temporal coverage, before and after, with images.
- **[HTML report](docs/report.md)**: turn a profile into a standalone report drawn with Autark, with the columns, their distributions and maps of the spatial coverage.

## Table of Contents

- [Features](#features)
- [What It Produces](#what-it-produces)
  - [Spatial coverage](#spatial-coverage)
  - [Temporal coverage](#temporal-coverage)
- [Type System](#type-system)
- [Architecture](#architecture)
  - [Pipeline](#pipeline)
  - [Spatial ML Classifier](#spatial-ml-classifier)
- [Advanced Usage](#advanced-usage)
  - [Configuration Parameters](#configuration-parameters)
  - [Manual Annotations](#manual-annotations)
  - [Model Files](#model-files)
- [Project Structure](#project-structure)
- [Related Projects](#related-projects)

## Features

✨ **Automatic Type Detection**: Identifies structural types (Integer, Float, Text, Boolean, GeoCoordinates, GeoShape) and semantic types (DateTime, Address, URL, ID, etc.)

🌍 **Spatial Intelligence**: ML-powered spatial column classifier trained on synthetic data, recognizing coordinates, addresses, geospatial identifiers, and administrative areas

🗺️ **Spatial Coverage**: For every lat/long pair, a robust bounding box, a k-d tree whose cells follow the density of the data, and an exact density grid of square cells (about 150–200 m for a city); placeholder (0, 0) coordinates and stray points are counted apart

⏰ **Temporal Analysis**: For every date column, its extent, precision, and spacing (typical interval, regularity, largest gap)

📊 **Rich Metadata**: Comprehensive dataset profiling including:
- Column-level statistics and distinct value counts (exact when small)
- Dataset-level type summaries
- Histograms that hold up with outliers: a robust range with counts below and above it, one bar per value for small integer domains, calendar bins for times
- A random sample of rows whose size adapts to the table, and a record of what was computed on it
- Profiling performance metrics

## What It Produces

`process_dataset(...)` returns a metadata dictionary with:

- **Dataset metrics**: row count, column count, profiled row count
- **Per-column analysis**: structural type, semantic types, missing value ratios, distinct counts, sample values
- **Dataset summary**: numerical, categorical, spatial, and temporal type counts
- **Coverage information**: spatial bounding boxes, geohash cells, a k-d tree and an exact density grid of each lat/long pair; temporal ranges, precision and spacing
- **Provenance**: `computed_on` says how the profiled rows were chosen (method, seed, size rule, rows of total) and, for each part of the profile, whether it comes from the sample or from every row; each spatial and temporal coverage entry has its own `computed_on`
- **Attribute keywords**: automatically extracted from column names
- **Performance metrics**: per-step profiling timings

### Spatial coverage

Each entry of `spatial_coverage` describes one lat/long pair (other entries describe WKT points, addresses and administrative areas):

| Field | Meaning |
| --- | --- |
| `ranges` | Bounding box, as `[[min_long, max_lat], [max_long, min_lat]]`; for a CSV path it spans the 0.1–99.9% quantiles of each coordinate |
| `number` | Points with valid coordinates, excluding (0, 0) |
| `number_zero_coordinates` | Points at (0, 0), a common placeholder for missing coordinates, left out |
| `number_outside_envelope` | Points outside the 0.1–99.9% box, kept out of the cells below |
| `geohashes4` | Base-4 geohash cells and their counts, at most 100 cells |
| `kdtree` | Leaves of a median-split k-d tree (4,096 by default, `spatial_leaves`): each leaf holds about the same number of points, so leaves are small where data is dense |
| `grid` | Exact point counts in square cells, 200 along the longer side of the area holding most of the points (chosen with the k-d tree); non-empty cells as `[column, row, count]` |

### Temporal coverage

Each entry of `temporal_coverage` describes one date or time column:

| Field | Meaning |
| --- | --- |
| `ranges` | Extent; for a CSV path, the exact min and max over every row |
| `temporal_precision` | Finest unit the timestamps use (`second`, `minute`, `day`, ...) |
| `temporal_resolution` | Finest calendar unit that separates the *sampled* timestamps; depends on the sample size |
| `sampling` | Spacing of the distinct timestamps over every row: median and percentile intervals, most common interval and whether it dominates (`regular`), gaps over 10× the median, largest gap |

## Type System

### Structural Types

The profiler recognizes these broad structural types:

| Type | Meaning |
| --- | --- |
| `Integer` | Integer-like values |
| `Float` | Floating point values |
| `Text` | String/text values |
| `Boolean` | Boolean-like values (true/false, yes/no, 0/1) |
| `GeoCoordinates` | Point geometry or coordinate-pair strings |
| `GeoShape` | Polygon-like geometry |
| `MissingData` | Empty column |

### Semantic Types

The profiler also annotates semantic meaning when evidence is available:

| Type | Examples |
| --- | --- |
| `DateTime` | Dates, timestamps, year columns |
| `latitude`, `longitude` | Coordinate columns (paired after profiling) |
| `address`, `AdministrativeArea` | Address text or admin areas (optionally resolved via Nominatim or `datamart_geo`) |
| `URL`, `FileName`, `identifier`, `Enumeration` | URLs, file paths, IDs, categorical values |

## Architecture

### Pipeline

`process_dataset` executes a consistent workflow for every dataset:

1. **Load data** from a path, file object or DataFrame. A CSV/TSV path is streamed with polars: distinct counts and numeric statistics are computed over every row, then a seeded random sample of rows is read for the next steps.
2. **Predict spatial labels** (optional) using batch ML inference.
3. **Process columns** on the sample: rule-based type detection, informed by the spatial predictions, plus value ranges and histograms.
4. **Pair lat/long columns** and compute dataset-level type summaries.
5. **Compute spatial coverage** (optional): bounding box, geohash cells, k-d tree and density grid, over every row of a CSV path.
6. **Compute temporal coverage** (optional): extent, precision and spacing, over every row of a CSV path.

### Spatial ML Classifier

When `geo_classifier=True`, Atlas Profiler uses a `HybridGeoClassifier` that:

- Samples values from each column
- Predicts spatial labels in a single batch
- Validates predictions using rule-based checks
- Maps predictions to the profiler's type system

**Supported spatial labels:**

| Label Family | Mapped Type |
| --- | --- |
| `latitude`, `longitude` | Float + semantic types |
| `x_coord`, `y_coord` | Projected coordinates |
| `point`, `polygon`, `line` | Geometry types |
| `address`, `zip5`, `zip9` | Address/postal codes |
| `borough`, `city`, `state`, `country` | Administrative areas |
| `bbl`, `bin` | NYC spatial identifiers |
| `non_spatial` | Falls back to standard detection |

Manual annotations take precedence over ML predictions. Low-confidence or rejected predictions fall back to rule-based detection.

## Advanced Usage

### Configuration Parameters

Key parameters for `process_dataset()`:

| Parameter | Default | Description |
| --- | --- | --- |
| `data` | required | Path (`str` or `os.PathLike`), file-like object, or pandas DataFrame. DataFrame column labels are converted to strings, and duplicates renamed `name`, `name.1`, … |
| `delimiter` | `None` | Field separator of a CSV/TSV path or file object; by default a comma, or a tab for `.tsv`/`.tab` paths |
| `geo_classifier` | `True` | Enable spatial ML classifier |
| `geo_classifier_threshold` | `0.5` | Confidence cutoff for predictions |
| `coverage` | `True` | Compute numerical ranges and spatial/temporal coverage |
| `plots` | `False` | Include a histogram per column (`plot`) |
| `full_statistics` | `False` | With `plots=True`, compute the histograms over every row instead of the profiled sample (adds about 25% to the time of a large CSV) |
| `include_sample` | `False` | Include sample rows in output |
| `indexes` | `True` | Preserve DataFrame indexes as columns |
| `load_max_size` | `None` | Bytes of in-memory data to profile. By default the sample size adapts to the table: about 2 million values (rows × columns), at least 10,000 rows, at most 256 MB. Larger inputs are randomly sampled |
| `spatial_leaves` | `4096` | Leaves of the k-d tree that summarizes each lat/long pair; it also chooses the area of the density grid |
| `diagnostics_path` | `None` | CSV file to which the spatial classifier's predictions, with the sampled values it saw, are appended. Off by default: profiling writes no file, and logs never include data values |
| `metadata` | `None` | Optional seed metadata with manual annotations |
| `nominatim` | `None` | Nominatim endpoint for address resolution |
| `datamart_geo_data` | `None` | GeoData instance for admin-area resolution |

### Manual Annotations

Supply manual type annotations through the `metadata` argument. Useful when upstream processes or domain knowledge already identifies column types:

```python
metadata = {
    "columns": [
        {
            "name": "latitude",
            "semantic_types": ["http://schema.org/latitude"]
        },
        {
            "name": "longitude", 
            "semantic_types": ["http://schema.org/longitude"]
        }
    ]
}

result = process_dataset(df, metadata=metadata)
```

Manually annotated columns skip the spatial ML classifier and are reconciled with observed values during processing.

### Model Files

The spatial ML classifier uses these model files, downloaded from NYU Box the first time the classifier runs:

- `model.pt` — PyTorch model weights
- `config.json` — Model configuration
- `label_encoder.json` — Label encoding
- `tokenizer.json`, `tokenizer_config.json` — Tokenizer

They are read from `profiler/model/` when present there, and otherwise cached in `$XDG_CACHE_HOME/atlas_profiler/model` (`~/.cache/atlas_profiler/model` by default). To use the classifier directly, `GeoClassifier(model_dir=None, auto_download=True)` is in `profiler.geo_model`. PyTorch and transformers are imported only when the classifier is used, so `geo_classifier=False` does not load them.

For model training details, see [`training/README.md`](training/README.md).

## Project Structure

```
atlas-profiler/
├── atlas_profiler/          # Public API: from atlas_profiler import process_dataset
├── profiler/                # Core profiling package
│   ├── core.py              # process_dataset(), data loading, column pipeline
│   ├── profile_types.py     # Rule-based type detection
│   ├── spatial.py           # Spatial coverage helpers, lat/long pairing, geohashes
│   ├── spatial_index.py     # k-d tree, density grid, geohash cells (numpy)
│   ├── geo_model.py         # GeoClassifier (PyTorch), loaded on first use
│   ├── histograms.py        # Histograms, temporal precision and spacing
│   ├── temporal.py          # Temporal analysis
│   ├── numerical.py         # Numerical profiling
│   └── types.py             # Type constants
├── training/                # Model training & synthetic data generation
├── tests/                   # Unit tests
├── examples/                # Example notebooks
├── docs/                    # Sphinx documentation and Markdown pages
└── pyproject.toml           # Project configuration
```

## Related Projects

This project builds upon and extends [Datamart Profiler](https://gitlab.com/ViDA-NYU/auctus/auctus) with additional spatial intelligence via ML-assisted column type classification.

- **Datamart Profiler**: https://pypi.org/project/datamart-profiler/
- **Research Background**: Developed by the [NYU Visualization and Data Analytics Lab](https://vida-nyu.github.io/)

## License

Atlas Profiler is released under the [MIT License](LICENSE).
