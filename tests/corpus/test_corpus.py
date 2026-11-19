"""Corpus integration tests (tests/TEST_PLAN.md), on GeoJSON files from the UrbanTrace data lake.

Opt-in: set ATLAS_PROFILER_TEST_DATA_DIR to the folder that holds ``geojson/`` (for example
``../urbantrace/data``). Without it these tests are skipped; with it, missing files or a changed
snapshot fail. Each feature's ``properties`` become one row, in order, keeping strings and nulls; the
profiler only sees tabular input. Expected values are computed from the files with plain Python.
"""
import csv
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy
import pandas
import pytest

DATA_DIR = os.environ.get("ATLAS_PROFILER_TEST_DATA_DIR")
pytestmark = pytest.mark.skipif(not DATA_DIR, reason="set ATLAS_PROFILER_TEST_DATA_DIR to run corpus tests")

from profiler.core import process_dataset  # noqa: E402

# Pinned snapshot (SHA-256 of each file)
SNAPSHOT = {
    "NYC_libraries": "e98a9b72cc6bd04b5832f2f0bfee68102916d37664045123b8e95d706babd0c2",
    "NYC_bicycle_counts": "690b648c51c3c06b0c4abab0d84bf93c9d1178b305cf06dc7957bf79b171202a",
    "2018_central_park_squirrel_census_squirrel_data": "eefb56ae72b320e9c1381c7b0c5e275cb2b13d32111af5505eab95f46160b7d0",
    "NTA_Neighborhood_Tabulation_Areas": "e76cd6f6247d963b8f7ed2011ff44e184681dad7151ab11038478d3c019649a7",
    "NYC_Bike_Routes": "93c64b256360e9131fa36d752f1e8e912b004383edb33a9871995ad5b3ec17e6",
}


def _features(name):
    path = Path(DATA_DIR) / "geojson" / f"{name}.geojson"
    assert path.exists(), f"corpus file missing: {path}"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == SNAPSHOT[name], f"{path.name} differs from the pinned snapshot"
    return json.loads(path.read_text())["features"]


def _rows(features, limit=None):
    rows = [dict(f["properties"]) for f in features[:limit]]
    columns = list(dict.fromkeys(k for r in rows for k in r))
    return rows, columns


def _frame(features, limit=None):
    rows, columns = _rows(features, limit)
    return pandas.DataFrame(rows, columns=columns)


def _csv(features, path, limit=None, transform=None):
    rows, columns = _rows(features, limit)
    if transform:
        rows = [transform(r) for r in rows]
        columns = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for r in rows:
            w.writerow({k: "" if r.get(k) is None else r[k] for k in columns})
    return str(path)


def _latlong(m):
    return [e for e in m.get("spatial_coverage", []) if e["type"] == "latlong"]


# --- NYC libraries -----------------------------------------------------------------------------

@pytest.mark.parametrize("source", ["frame", "csv"])
def test_libraries(tmp_path, source):
    features = _features("NYC_libraries")
    assert len(features) == 253
    data = _frame(features) if source == "frame" else _csv(features, tmp_path / "libraries.csv")
    m = process_dataset(data, geo_classifier=False, indexes=False)

    pairs = _latlong(m)
    assert [p["column_names"] for p in pairs] == [["latitude", "longitude"]]  # xcoord/ycoord make no pair
    assert pairs[0]["number"] == 253
    cols = {c["name"]: c for c in m["columns"]}
    assert cols["captype"]["structural_type"].endswith("/MissingData")  # an all-null column

    if source == "csv":
        # Envelope: the 0.1% and 99.9% quantiles of each coordinate, polars' default "nearest" method
        lat = numpy.array([f["properties"]["latitude"] for f in features])
        lon = numpy.array([f["properties"]["longitude"] for f in features])
        lat_lo, lat_hi = numpy.quantile(lat, (0.001, 0.999), method="nearest")
        lon_lo, lon_hi = numpy.quantile(lon, (0.001, 0.999), method="nearest")
        (min_lon, max_lat), (max_lon, min_lat) = pairs[0]["ranges"][0]["range"]["coordinates"]
        assert (min_lon, max_lon, min_lat, max_lat) == pytest.approx((lon_lo, lon_hi, lat_lo, lat_hi), abs=1e-9)


def test_library_properties_agree_with_geometry():
    for f in _features("NYC_libraries"):
        lon, lat = f["geometry"]["coordinates"][:2]   # GeoJSON order: longitude, latitude
        assert abs(lon - f["properties"]["longitude"]) < 1e-6 and abs(lat - f["properties"]["latitude"]) < 1e-6


# --- NYC bicycle counts ------------------------------------------------------------------------

def test_bicycle_latlong_and_wkt_identify_the_same_points():
    features = _features("NYC_bicycle_counts")
    m = process_dataset(_frame(features, limit=3_000), geo_classifier=False, indexes=False)
    latlong = _latlong(m)[0]
    wkt = next(e for e in m["spatial_coverage"] if e["column_names"] == ["geom_wkt"])
    assert latlong["number"] == wkt["number"] == 3_000
    assert sum(c["number"] for c in wkt["geohashes4"]) == 3_000
    assert sum(c["number"] for c in latlong["geohashes4"]) == 3_000 - latlong["number_outside_envelope"]
    flat = lambda e: [v for rng in e["ranges"] for pt in rng["range"]["coordinates"] for v in pt]  # noqa: E731
    assert flat(latlong) == pytest.approx(flat(wkt), abs=1e-6)


def test_bicycle_counts_whole_file(tmp_path):
    features = _features("NYC_bicycle_counts")
    assert len(features) == 28_459
    path = _csv(features, tmp_path / "bicycle.csv")
    m = process_dataset(path, geo_classifier=False, load_max_size=500_000)   # force a sample
    assert 0 < m["nb_profiled_rows"] < m["nb_rows"] == 28_459

    latlong = _latlong(m)[0]
    wkt = next(e for e in m["spatial_coverage"] if e["column_names"] == ["geom_wkt"])
    assert latlong["number"] == 28_459 and latlong["computed_on"] == "all_rows"
    assert wkt["number"] == m["nb_profiled_rows"] and wkt["computed_on"] == "sample"
    assert m["computed_on"]["spatial_coverage"] == "mixed"

    counts = [f["properties"]["counts"] for f in features]
    cols = {c["name"]: c for c in m["columns"]}
    assert (cols["counts"]["min"], cols["counts"]["max"]) == (min(counts), max(counts)) == (0, 759)
    assert cols["counts"]["mean"] == pytest.approx(sum(counts) / len(counts)) == pytest.approx(719_179 / 28_459)

    times = [datetime.strptime(f["properties"]["date"], "%m/%d/%Y %I:%M:%S %p").replace(tzinfo=timezone.utc)
             for f in features]
    assert (min(times), max(times)) == (datetime(2012, 9, 3, 7, 30, tzinfo=timezone.utc),
                                         datetime(2025, 8, 17, 3, 0, tzinfo=timezone.utc))
    date = next(e for e in m["temporal_coverage"] if e["column_names"] == ["date"])
    r = date["ranges"][0]["range"]
    assert (r["gte"], r["lte"]) == (min(times).timestamp(), max(times).timestamp())


# --- Central Park squirrel census --------------------------------------------------------------

def test_squirrel_booleans():
    m = process_dataset(_frame(_features("2018_central_park_squirrel_census_squirrel_data")), geo_classifier=False,
                        indexes=False)
    running = next(c for c in m["columns"] if c["name"] == "running")
    assert "http://schema.org/Boolean" in (running["semantic_types"] + [running["structural_type"]])


def test_squirrel_normalized_dates_and_coordinates(tmp_path):
    """Companion case: the fixture declares the date format (%m%d%Y integers) and maps x/y to
    longitude/latitude. Recognizing the raw forms automatically is a separate feature decision."""
    features = _features("2018_central_park_squirrel_census_squirrel_data")
    assert len(features) == 3_023
    for f in features:   # x/y agree with the geometry (longitude, latitude)
        lon, lat = f["geometry"]["coordinates"][:2]
        assert abs(lon - f["properties"]["x"]) < 1e-9 and abs(lat - f["properties"]["y"]) < 1e-9

    def normalize(row):
        row = dict(row)
        row["date"] = datetime.strptime(f"{row['date']:08d}", "%m%d%Y").strftime("%Y-%m-%d")
        row["longitude"], row["latitude"] = row.pop("x"), row.pop("y")
        return row

    m = process_dataset(_csv(features, tmp_path / "squirrels.csv", transform=normalize), geo_classifier=False)
    pairs = _latlong(m)
    assert len(pairs) == 1 and pairs[0]["number"] == 3_023
    date = next(e for e in m["temporal_coverage"] if e["column_names"] == ["date"])
    r = date["ranges"][0]["range"]
    utc = lambda *a: datetime(*a, tzinfo=timezone.utc).timestamp()  # noqa: E731
    assert (r["gte"], r["lte"]) == (utc(2018, 10, 6), utc(2018, 10, 20))


# --- Polygons and lines: properties must not create points ---------------------------------------

@pytest.mark.parametrize("name, limit, expected_rows", [
    ("NTA_Neighborhood_Tabulation_Areas", None, 262),
    ("NYC_Bike_Routes", 5_000, 5_000),
])
def test_identifiers_and_measures_create_no_point_coverage(name, limit, expected_rows):
    features = _features(name)
    m = process_dataset(_frame(features, limit=limit), geo_classifier=False, indexes=False)
    assert m["nb_rows"] == expected_rows
    kinds = {e["type"] for e in m.get("spatial_coverage", [])}
    assert not kinds & {"latlong", "point", "point_latlong"}, m.get("spatial_coverage")
    assert math.isfinite(len(m["columns"]))
