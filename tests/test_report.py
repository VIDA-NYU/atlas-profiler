import pytest

from profiler.report import _coverage_cells, _warnings, decode_geohash


METADATA = {
    "nb_rows": 1000,
    "nb_profiled_rows": 100,
    "nb_columns": 3,
    "size": 2048,
    "columns": [
        {
            "name": " distance",
            "structural_type": "http://schema.org/Float",
            "semantic_types": [],
            "mean": 2.5,
            "stddev": 1.0,
            "min": 0.0,
            "max": 90.0,
            "coverage": [{"range": {"gte": 0.5, "lte": 4.0}}],
            "plot": {
                "type": "histogram_numerical",
                "data": [{"count": 5, "bin_start": 0.0, "bin_end": 45.0}, {"count": 1, "bin_start": 45.0, "bin_end": 90.0}],
            },
        },
        {"name": "<b>note</b>", "structural_type": "http://schema.org/Text", "semantic_types": [], "missing_values_ratio": 0.9},
        {"name": "when", "structural_type": "http://schema.org/Text", "semantic_types": ["http://schema.org/DateTime"]},
    ],
    "spatial_coverage": [
        {
            "type": "latlong",
            "column_names": ["lat", "lon"],
            "ranges": [{"range": {"type": "envelope", "coordinates": [[-88.0, 51.0], [0.0, -0.5]]}}],
            "number": 1000,
            "geohashes4": [{"hash": "121130231331", "number": 995}, {"hash": "3", "number": 5}],
        }
    ],
    "temporal_coverage": [
        {
            "type": "datetime",
            "column_names": ["when"],
            "ranges": [{"range": {"gte": 1356998400.0, "lte": 1359676799.0}}],
            "temporal_resolution": "minute",
        }
    ],
    "_profiling_times": {"steps": {"1_load_data": 0.2, "6_temporal_coverage": 8.9}, "total": 9.1},
}


def test_decode_geohash_quadrants():
    assert decode_geohash("") == (-90.0, 90.0, -180.0, 180.0)
    # The first bit splits longitude, the second latitude: "3" is the north-east quadrant.
    assert decode_geohash("3") == (0.0, 90.0, 0.0, 180.0)
    assert decode_geohash("0") == (-90.0, 0.0, -180.0, 0.0)


def test_decode_geohash_nyc_cell():
    min_lat, max_lat, min_lon, max_lon = decode_geohash("121130231331")
    assert min_lat < 40.75 < max_lat
    assert min_lon < -73.95 < max_lon


def test_coverage_cells_drop_sparse_far_cells():
    features, out_cells, out_points, frame = _coverage_cells(METADATA["spatial_coverage"][0])
    assert frame[0] < -73.95 < frame[1] and frame[2] < 40.75 < frame[3]
    assert [f["properties"]["cell"] for f in features] == ["121130231331"]
    assert (out_cells, out_points) == (1, 5)
    ring = features[0]["geometry"]["coordinates"][0]
    assert ring[0] == ring[-1] and len(ring) == 5


def test_warnings():
    text = " ".join(_warnings(METADATA))
    assert "sample of 100 of 1,000 rows" in text
    assert "leading or trailing whitespace" in text
    assert "90.0% missing" in text
    assert "contains (0, 0)" in text
    assert "<b>" not in text and "&lt;b&gt;" in text


def test_histogram_flags_flattened_bins():
    from profiler.report import _histogram

    records, caption = _histogram(METADATA["columns"][0])
    assert [r["bin"] for r in records] == ["0–45", "45–90"]
    assert caption == ""  # 5 of 6 in one bin is under the 90% threshold
    flat = dict(METADATA["columns"][0], plot={"type": "histogram_numerical", "data": [
        {"count": 95, "bin_start": 0.0, "bin_end": 45.0}, {"count": 1, "bin_start": 45.0, "bin_end": 90.0}]})
    assert "99% of the values fall in one bin" in _histogram(flat)[1]


def test_views():
    pytest.importorskip("autark")
    from profiler.report import _distributions_spec, _maps_spec

    spec, captions, missing = _distributions_spec(METADATA)
    assert [type(v).__name__ for v in spec.views] == ["Barchart"]
    assert missing == ["<b>note</b>", "when"]
    maps, notes = _maps_spec(METADATA)
    assert [type(v).__name__ for v in maps.views] == ["Map"]
    assert "not shown" in notes[0]


def test_page_structure():
    pytest.importorskip("autark")
    from profiler.report import render_html

    page = render_html(METADATA, title="Test")
    for section in ('id="overview"', 'id="columns"', 'id="distributions"', 'id="spatial"', 'id="temporal"', 'id="timing"'):
        assert section in page
    assert page.count("<iframe") == 2
    assert "&lt;b&gt;note&lt;/b&gt;" in page and "<b>note</b>" not in page.split("<iframe")[0]


NEW_PROFILE_COLUMN = {
    "name": "passengers",
    "structural_type": "http://schema.org/Integer",
    "semantic_types": [],
    "plot": {"type": "histogram_numerical", "discrete": True, "below": 0, "above": 0, "data": [
        {"count": 70, "bin_start": 0.5, "bin_end": 1.5}, {"count": 30, "bin_start": 1.5, "bin_end": 2.5}]},
}


def test_discrete_histogram_labels_values():
    from profiler.report import _histogram

    records, caption = _histogram(NEW_PROFILE_COLUMN)
    assert [r["bin"] for r in records] == ["1", "2"]
    assert caption == "One bar per value."


def test_values_outside_range_become_end_bars():
    from profiler.report import _histogram

    column = {"name": "lat", "plot": {"type": "histogram_numerical", "below": 124, "above": 30, "data": [
        {"count": 5, "bin_start": 40.64, "bin_end": 40.7}, {"count": 9, "bin_start": 40.7, "bin_end": 40.81}]}}
    records, caption = _histogram(column)
    assert records[0] == {"bin": "< 40.64", "count": 124}
    assert records[-1] == {"bin": "> 40.81", "count": 30}
    assert "124 below, 30 above" in caption


def test_provenance_and_temporal_sampling_in_page():
    pytest.importorskip("autark")
    from profiler.report import render_html

    meta = dict(METADATA)
    meta["computed_on"] = {"column_types": "sample", "histograms": "sample", "spatial_coverage": "all_rows"}
    meta["temporal_coverage"] = [dict(METADATA["temporal_coverage"][0], temporal_precision="second", sampling={
        "median_interval_seconds": 23.0, "regular": False, "largest_gap_seconds": 1265.0})]
    page = render_html(meta)
    assert "random sample of 100 of 1,000 rows: column types, histograms come from the sample; spatial coverage from every row" in page
    assert "Median interval" in page and "23 s" in page and "21.1 min" in page and "irregular" in page


def test_maps_share_one_zoom():
    pytest.importorskip("autark")
    from profiler.report import _maps_spec

    near = {"type": "latlong", "column_names": ["a_lat", "a_lon"], "geohashes4": [{"hash": "121130231331", "number": 50}]}
    wide = {"type": "latlong", "column_names": ["b_lat", "b_lon"], "geohashes4": [{"hash": "1211302313", "number": 50}]}
    spec, _ = _maps_spec({"spatial_coverage": [near, wide]})
    zooms = {v.camera.zoom for v in spec.views}
    assert len(zooms) == 1


def test_profile_grid_is_drawn_as_is():
    pytest.importorskip("autark")
    from profiler.report import _coverage_cells, _maps_spec

    grid = {"bounds": [-74.0, 40.7, -73.9, 40.8], "columns": 2, "rows": 2, "cell_degrees": [0.05, 0.05],
            "cell_km": 4.2, "outside": 7, "cells": [[0, 0, 1000], [1, 1, 10]]}
    entry = {"type": "latlong", "column_names": ["lat", "lon"], "grid": grid, "kdtree": {"leaves": []}}
    features, _, outside, frame = _coverage_cells(entry)
    assert [f["properties"]["points"] for f in features] == [1000, 10]  # exact counts, no spreading
    assert features[1]["geometry"]["coordinates"][0][0] == [-73.95, 40.75]
    assert outside == 7 and frame == (-74.0, -73.9, 40.7, 40.8)
    spec, captions = _maps_spec({"spatial_coverage": [entry]})
    assert "counted exactly" in captions[0] and "7 points fall outside" in captions[0]


def test_long_histograms_are_merged_for_display():
    from profiler.report import MAX_BARS, _histogram

    data = [{"count": 1, "date_start": f"{1900 + i}-01-01T00:00:00", "date_end": f"{1901 + i}-01-01T00:00:00"}
            for i in range(118)]
    records, caption = _histogram({"name": "t", "plot": {"type": "histogram_temporal", "unit": "year", "data": data}})
    assert len(records) <= MAX_BARS and sum(r["count"] for r in records) == 118
    assert caption.startswith("3 bins per bar")


def test_long_category_labels_are_cut():
    from profiler.report import _histogram

    plot = {"type": "histogram_categorical", "data": [{"bin": "A very long category name indeed", "count": 3},
                                                       {"bin": "short", "count": 1}]}
    records, caption = _histogram({"name": "c", "plot": plot})
    assert records[0]["bin"] == "A very long c…" and records[1]["bin"] == "short"
    assert "cut to 14 characters" in caption


def test_multi_year_steps_are_explained():
    from profiler.report import _histogram

    data = [{"count": 2, "date_start": f"{1905 + 5 * i}-01-01T00:00:00", "date_end": f"{1910 + 5 * i}-01-01T00:00:00"}
            for i in range(24)]
    records, caption = _histogram({"name": "t", "plot": {"type": "histogram_temporal", "unit": "year", "step": 5,
                                                         "below": 0, "above": 0, "data": data}})
    assert records[0]["bin"] == "1905" and "Bars of 5 years" in caption
