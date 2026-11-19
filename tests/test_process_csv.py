"""End-to-end profiling of a CSV path."""
import hashlib

import pytest

pytest.importorskip("polars")

from profiler.core import process_dataset  # noqa: E402


@pytest.fixture
def trips_csv(tmp_path):
    rows = ["medallion, rate_code, pickup_datetime, pickup_latitude, pickup_longitude"]
    for i in range(3000):
        medallion = hashlib.md5(str(i % 400).encode()).hexdigest().upper()
        lat, lon = (0.0, 0.0) if i % 50 == 0 else (40.70 + (i % 100) / 1000, -74.00 + (i % 97) / 1000)
        minute = i * 7
        rows.append(f"{medallion}, {1 if i % 20 else 2}, 2013-01-{1 + minute // 1440 % 28:02d} "
                    f"{minute // 60 % 24:02d}:{minute % 60:02d}:{i % 60:02d}, {lat}, {lon}")
    path = tmp_path / "trips.csv"
    path.write_text("\n".join(rows) + "\n")
    return str(path)


def test_profile_of_a_csv_path(trips_csv):
    m = process_dataset(trips_csv, geo_classifier=False, plots=True, load_max_size=50_000)
    cols = {c["name"].strip(): c for c in m["columns"]}

    # Sampled, and the profile says which parts come from the sample
    assert m["nb_profiled_rows"] < m["nb_rows"] == 3000
    assert m["computed_on"]["histograms"] == "sample"
    assert m["computed_on"]["spatial_coverage"] == "all_rows"

    # Hashed identifiers; exact distinct counts when small
    assert "http://schema.org/identifier" in cols["medallion"]["semantic_types"]
    assert cols["medallion"]["num_distinct_values"] == 400
    assert cols["medallion"]["num_distinct_values_is_approximate"] is False

    # Values after ", " are stripped, so numbers are numbers; a small integer domain gets one bar per value
    assert cols["rate_code"]["structural_type"] == "http://schema.org/Integer"
    assert cols["rate_code"]["plot"].get("discrete") is True

    # (0, 0) is counted apart and kept out of the envelope
    spatial = next(c for c in m["spatial_coverage"] if c["type"] == "latlong")
    assert spatial["number_zero_coordinates"] == 60
    grid = spatial["grid"]
    assert sum(n for _, _, n in grid["cells"]) + grid["outside"] == spatial["number"] - spatial["number_outside_envelope"]
    leaves = spatial["kdtree"]["leaves"]
    assert len(leaves) > 64 and sum(leaf["number"] for leaf in leaves) + spatial["number_outside_envelope"] == spatial["number"]
    (min_lon, max_lat), (max_lon, min_lat) = spatial["ranges"][0]["range"]["coordinates"]
    assert min_lat > 40 and max_lon < -73

    # Time: precision and spacing over all rows
    temporal = m["temporal_coverage"][0]
    assert temporal["temporal_precision"] == "second"
    assert temporal["sampling"]["distinct_timestamps"] > 2000

    # Keywords are stripped and deduplicated
    assert all(k == k.strip() for k in m["attribute_keywords"])
    assert len(m["attribute_keywords"]) == len(set(m["attribute_keywords"]))


def test_stray_points_do_not_coarsen_geohash_cells(tmp_path):
    # 120,000 points in a city and 110 strays scattered over the world (under 0.1% of the points): the
    # strays fill the per-level cell budget at fine levels, which used to leave continent-sized cells.
    import numpy

    rng = numpy.random.default_rng(0)
    city = numpy.column_stack([rng.uniform(40.60, 40.85, 120_000), rng.uniform(-74.05, -73.75, 120_000)])
    stray = numpy.column_stack([rng.uniform(-60, 60, 110), rng.uniform(-170, 170, 110)])
    points = numpy.vstack([city, stray])
    path = tmp_path / "points.csv"
    path.write_text("latitude,longitude\n" + "\n".join(f"{a:.5f},{b:.5f}" for a, b in points) + "\n")

    m = process_dataset(str(path), geo_classifier=False)
    spatial = next(c for c in m["spatial_coverage"] if c["type"] == "latlong")
    from profiler.spatial import decode_hash

    widths = [decode_hash(c["hash"], base=4)[3] - decode_hash(c["hash"], base=4)[2] for c in spatial["geohashes4"]]
    assert max(widths) < 0.2  # degrees of longitude: city-scale cells
    assert spatial["number_outside_envelope"] >= 100
