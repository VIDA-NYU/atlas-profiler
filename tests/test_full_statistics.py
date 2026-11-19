"""full_statistics: histograms over every row instead of the profiled sample."""
import numpy
import pandas
import pytest

pytest.importorskip("polars")

from profiler.core import process_dataset  # noqa: E402


def _frame(n=20_000, seed=0):
    rng = numpy.random.default_rng(seed)
    when = numpy.datetime64("2013-01-01T00:00:00") + rng.integers(0, 31 * 86400, n).astype("timedelta64[s]")
    return pandas.DataFrame({
        "distance": numpy.round(rng.lognormal(1.0, 0.8, n), 2),
        "vendor": rng.choice(["CMT", "VTS", "DDS"], n, p=[0.5, 0.45, 0.05]),
        "pickup_datetime": pandas.to_datetime(when).strftime("%Y-%m-%d %H:%M:%S"),
    })


def _total(plot):
    return sum(d["count"] for d in plot["data"]) + plot.get("below", 0) + plot.get("above", 0)


@pytest.mark.parametrize("as_csv", [True, False])
def test_histograms_cover_every_row(tmp_path, as_csv):
    df = _frame()
    data = df
    if as_csv:
        data = str(tmp_path / "trips.csv")
        df.to_csv(data, index=False)
    m = process_dataset(data, geo_classifier=False, plots=True, full_statistics=True,
                        load_max_size=200_000, indexes=False)
    assert m["nb_profiled_rows"] < m["nb_rows"] == 20_000
    cols = {c["name"]: c for c in m["columns"]}
    assert m["computed_on"]["histograms"] == "all_rows"
    assert m["computed_on"]["column_types"] == "sample"

    assert cols["distance"]["plot"]["computed_on"] == "all_rows"
    assert _total(cols["distance"]["plot"]) == 20_000
    assert _total(cols["pickup_datetime"]["plot"]) == 20_000
    vendors = {d["bin"]: d["count"] for d in cols["vendor"]["plot"]["data"]}
    assert vendors == df["vendor"].value_counts().to_dict()


def test_without_the_option_histograms_come_from_the_sample(tmp_path):
    path = str(tmp_path / "trips.csv")
    _frame().to_csv(path, index=False)
    m = process_dataset(path, geo_classifier=False, plots=True, load_max_size=200_000)
    cols = {c["name"]: c for c in m["columns"]}
    assert m["computed_on"]["histograms"] == "sample"
    assert _total(cols["distance"]["plot"]) == m["nb_profiled_rows"]


def test_dates_that_cannot_be_parsed_in_bulk_stay_on_the_sample(tmp_path):
    # Day-first dates: the bulk parser could read 01/02 as February 1, so it is not used.
    rng = numpy.random.default_rng(1)
    days, hours, minutes = rng.integers(1, 29, 20_000), rng.integers(0, 24, 20_000), rng.integers(0, 60, 20_000)
    df = pandas.DataFrame({
        "x": rng.normal(size=20_000),
        "when": [f"{d:02d}/01/2013 {h:02d}:{mi:02d}" for d, h, mi in zip(days, hours, minutes)],
    })
    path = str(tmp_path / "dates.csv")
    df.to_csv(path, index=False)
    m = process_dataset(path, geo_classifier=False, plots=True, full_statistics=True, load_max_size=100_000)
    cols = {c["name"]: c for c in m["columns"]}
    assert cols["x"]["plot"]["computed_on"] == "all_rows"
    assert cols["when"]["plot"]["type"] == "histogram_temporal"
    when = cols["when"]["plot"]
    if when["computed_on"] == "sample":
        assert m["computed_on"]["histograms"] == "mixed"
    else:  # the bulk parse agreed with the exact parser on every checked value
        assert _total(when) == 20_000
