"""Generated regression cases from tests/TEST_PLAN.md: delimited input, sample scope, coordinates, time.

Expected values are computed independently of the profiler (Python's csv module, datetime.strptime,
plain arithmetic).
"""
import csv
import io
import json
import math
import pathlib
from datetime import datetime, timezone

import numpy
import pandas
import pytest

pytest.importorskip("polars")

from profiler.core import process_dataset  # noqa: E402

QUOTED = 'name,note\n"Smith, J","line one\nline two"\n"Müller","漢字, ok"\n,""\n'


def _profile(data, **kw):
    return process_dataset(data, geo_classifier=False, **kw)


# --- Delimited input contract ------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["str", "path", "text_stream", "bytes_stream", "frame"])
def test_quoted_fields_newlines_and_utf8(tmp_path, kind):
    file = tmp_path / "quoted.csv"
    file.write_text(QUOTED, encoding="utf-8")
    data = {
        "str": str(file), "path": pathlib.Path(file),
        "text_stream": io.StringIO(QUOTED), "bytes_stream": io.BytesIO(QUOTED.encode("utf-8")),
        "frame": pandas.read_csv(io.StringIO(QUOTED), dtype=str, keep_default_na=False),
    }[kind]
    m = _profile(data, indexes=False)
    assert m["nb_rows"] == 3
    assert [c["name"] for c in m["columns"]] == ["name", "note"]
    json.dumps(m, allow_nan=False, default=str)


def test_tsv_path_and_stream(tmp_path):
    text = "a\tb\n1\tx, y\n2\tz\n"
    path = tmp_path / "data.tsv"
    path.write_text(text)
    for data, kw in ((str(path), {}), (io.StringIO(text), {"delimiter": "\t"})):
        m = _profile(data, **kw)
        assert m["nb_rows"] == 2 and [c["name"] for c in m["columns"]] == ["a", "b"]


@pytest.mark.parametrize("kind", ["path", "stream"])
def test_header_only_input(tmp_path, kind):
    path = tmp_path / "header.csv"
    path.write_text("a,b\n")
    m = _profile(str(path) if kind == "path" else io.StringIO("a,b\n"))
    assert m["nb_rows"] == 0 and [c["name"] for c in m["columns"]] == ["a", "b"]


@pytest.mark.parametrize("kind", ["path", "stream"])
def test_empty_input_gives_an_empty_profile(tmp_path, kind):
    path = tmp_path / "empty.csv"
    path.write_text("")
    m = _profile(str(path) if kind == "path" else io.StringIO(""))
    assert m["nb_rows"] == 0 and m["columns"] == []


# --- Sample scope and late extremes ------------------------------------------------------------

def test_whole_file_metrics_match_an_independent_calculation(tmp_path):
    """Rows sorted so the second region, the late timestamps and the extreme values all come after
    any prefix the profiler could have read."""
    rng = numpy.random.default_rng(3)
    n = 30_000
    rows = []
    for i in range(n):
        late = i >= n * 2 // 3
        lat = 40.65 + rng.uniform(0, 0.02) if late else 40.78 + rng.uniform(0, 0.02)
        lon = -73.95 + rng.uniform(0, 0.02) if late else -73.99 + rng.uniform(0, 0.02)
        when = datetime(2014 if late else 2013, 1, 1) + pandas.Timedelta(minutes=int(i))
        value = 1_000.0 + i if late else float(i % 100)
        rows.append([f"{lat:.6f}", f"{lon:.6f}", when.strftime("%Y-%m-%d %H:%M:%S"), f"{value:.1f}"])
    path = tmp_path / "sorted.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["latitude", "longitude", "when", "value"])
        w.writerows(rows)

    m = _profile(str(path), plots=True, load_max_size=300_000)
    assert 0 < m["nb_profiled_rows"] < m["nb_rows"] == n

    # Independent full-file values
    with open(path, newline="") as f:
        data = list(csv.DictReader(f))
    values = [float(r["value"]) for r in data]
    times = [datetime.strptime(r["when"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc) for r in data]

    cols = {c["name"]: c for c in m["columns"]}
    assert cols["value"]["min"] == min(values) and cols["value"]["max"] == max(values)
    assert cols["value"]["mean"] == pytest.approx(sum(values) / n)
    latlong = next(e for e in m["spatial_coverage"] if e["type"] == "latlong")
    assert latlong["number"] == n  # every valid pair, late region included
    r = m["temporal_coverage"][0]["ranges"][0]["range"]
    assert (r["gte"], r["lte"]) == (min(times).timestamp(), max(times).timestamp())

    # Histograms describe the profiled rows, and say so
    total = sum(d["count"] for d in cols["value"]["plot"]["data"]) + cols["value"]["plot"].get("below", 0) \
        + cols["value"]["plot"].get("above", 0)
    assert total == m["nb_profiled_rows"]
    assert m["computed_on"]["histograms"] == "sample"
    assert m["computed_on"]["numeric_statistics"] == latlong["computed_on"] == "all_rows"


# --- Coordinate validity and pairing ------------------------------------------------------------

def test_coordinate_validity_and_pairing(tmp_path):
    n = 200
    lines = ["pickup_latitude,pickup_longitude,dropoff_latitude,dropoff_longitude,extra_lat,wkt"]
    lines += ["40.7,-73.9,40.8,-73.8,40.1,POINT (-73.95 40.75)"] * (n - 3)
    lines.append("95.0,-73.9,40.8,,40.1,POINT (-73.95 40.75)")       # out-of-range pickup; dropoff half missing
    lines.append("90.0,-73.9,40.8,-73.8,40.1,POINT (-73.95 40.75)")  # on the bound: excluded (strictly inside)
    lines.append("40.7,-73.9,,-73.8,40.1,POINT (-73.95 40.75)")      # dropoff half missing
    path = tmp_path / "pairs.csv"
    path.write_text("\n".join(lines) + "\n")

    m = _profile(str(path))
    entries = {tuple(e["column_names"]): e for e in m["spatial_coverage"]}
    pickup = entries[("pickup_latitude", "pickup_longitude")]
    dropoff = entries[("dropoff_latitude", "dropoff_longitude")]
    assert pickup["number"] == n - 2       # the 95 and the 90 rows are out
    assert dropoff["number"] == n - 2      # two rows with a missing half
    assert not any("extra_lat" in names for names in entries)  # an unmatched axis makes no pair

    (min_lon, max_lat), (max_lon, min_lat) = pickup["ranges"][0]["range"]["coordinates"]
    assert min_lon <= -73.9 <= max_lon and min_lat <= 40.7 <= max_lat  # longitude first, then latitude

    point = entries[("wkt",)]
    (w_min_lon, w_max_lat), (w_max_lon, w_min_lat) = point["ranges"][0]["range"]["coordinates"]
    assert w_min_lon < -73.9 and w_min_lat > 40  # POINT (lon lat), not swapped


# --- Numeric and temporal robustness ------------------------------------------------------------

def test_timestamps_with_offsets_and_naive_timestamps(tmp_path):
    path = tmp_path / "times.csv"
    path.write_text(
        "with_offset,naive\n"
        "2020-01-01T10:00:00+02:00,2020-01-01 10:00:00\n"
        "2020-01-02T10:00:00+02:00,2020-01-02 10:00:00\n"
        "2020-01-03T00:00:00+02:00,2020-01-03 00:00:00\n"
    )
    m = _profile(str(path))
    by_name = {e["column_names"][0]: e["ranges"][0]["range"] for e in m["temporal_coverage"]}
    utc = lambda *a: datetime(*a, tzinfo=timezone.utc).timestamp()  # noqa: E731
    assert (by_name["with_offset"]["gte"], by_name["with_offset"]["lte"]) == (utc(2020, 1, 1, 8), utc(2020, 1, 2, 22))
    assert (by_name["naive"]["gte"], by_name["naive"]["lte"]) == (utc(2020, 1, 1, 10), utc(2020, 1, 3, 0))


def test_mixed_non_finite_and_empty_numbers(tmp_path):
    path = tmp_path / "numbers.csv"
    # A numeric column with a few non-finite values (under the 2% type detection tolerates) and blanks
    values = ["1", "2", "3", ""] * 100 + ["nan", "inf", "-inf"]
    path.write_text("x\n" + "\n".join(values) + "\n")
    m = _profile(str(path), plots=True)
    json.dumps(m, allow_nan=False, default=str)
    x = m["columns"][0]
    assert x["structural_type"].endswith(("Integer", "Float"))
    assert (x["min"], x["max"], x["mean"]) == (1.0, 3.0, 2.0)   # over the finite values only
    assert all(math.isfinite(x[k]) for k in ("mean", "stddev"))
