"""Which rows each part of a profile comes from, on a file whose early and late rows differ."""
import numpy
import pandas
import pytest

pytest.importorskip("polars")

from profiler.core import process_dataset  # noqa: E402


@pytest.fixture
def shifting_csv(tmp_path):
    """First half: phase 'early', values near 0, January, Manhattan. Second half: 'late', near 100,
    June, Brooklyn. A sample from the start of the file would see only the first half."""
    rng = numpy.random.default_rng(0)
    n = 20_000
    half = n // 2
    early = numpy.arange(n) < half
    frame = pandas.DataFrame({
        "phase": numpy.where(early, "early", "late"),
        "value": numpy.where(early, rng.normal(0, 1, n), rng.normal(100, 1, n)).round(3),
        "when": numpy.where(
            early,
            pandas.date_range("2013-01-01", periods=n, freq="2min").strftime("%Y-%m-%d %H:%M:%S"),
            pandas.date_range("2013-06-01", periods=n, freq="2min").strftime("%Y-%m-%d %H:%M:%S"),
        ),
        "latitude": numpy.where(early, rng.uniform(40.75, 40.80, n), rng.uniform(40.65, 40.70, n)).round(5),
        "longitude": numpy.where(early, rng.uniform(-74.00, -73.95, n), rng.uniform(-73.97, -73.92, n)).round(5),
    })
    path = tmp_path / "shifting.csv"
    frame.to_csv(path, index=False)
    return str(path)


def test_sample_sees_both_halves(shifting_csv):
    m = process_dataset(shifting_csv, geo_classifier=False, plots=True, load_max_size=200_000)
    assert m["nb_profiled_rows"] < m["nb_rows"]
    cols = {c["name"]: c for c in m["columns"]}
    assert {d["bin"] for d in cols["phase"]["plot"]["data"]} == {"early", "late"}
    ranges = [(r["range"]["gte"], r["range"]["lte"]) for r in cols["value"]["coverage"]]
    assert any(hi < 10 for _, hi in ranges) and any(lo > 90 for lo, _ in ranges)
    months = {d["date_start"][:7] for d in cols["when"]["plot"]["data"] if d["count"]}
    assert "2013-01" in months and "2013-06" in months


def test_provenance_of_each_part(shifting_csv):
    m = process_dataset(shifting_csv, geo_classifier=False, plots=True, load_max_size=200_000)
    c = m["computed_on"]
    assert c["sampling"] == {"method": "random", "seed": 89, "size_rule": "load_max_size",
                             "rows": m["nb_profiled_rows"], "of": 20_000}
    assert c["column_types"] == c["histograms"] == c["value_ranges"] == c["temporal_resolution"] == "sample"
    assert c["numeric_statistics"] == c["spatial_coverage"] == c["temporal_coverage"] == "all_rows"

    latlong = next(e for e in m["spatial_coverage"] if e["type"] == "latlong")
    assert latlong["computed_on"] == "all_rows"
    assert latlong["number"] == 20_000  # both halves
    temporal = m["temporal_coverage"][0]
    assert temporal["computed_on"] == {"ranges": "all_rows", "sampling": "all_rows",
                                       "temporal_resolution": "sample", "temporal_precision": "sample"}
    start, end = temporal["ranges"][0]["range"]["gte"], temporal["ranges"][0]["range"]["lte"]
    assert end - start > 150 * 86400  # January to June: every row, not a prefix
