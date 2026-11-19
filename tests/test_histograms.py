import numpy
import pytest

from profiler.histograms import numerical_histogram, temporal_histogram, temporal_precision, temporal_sampling


def test_outliers_do_not_flatten_numerical_histogram():
    values = list(numpy.linspace(0, 10, 1000)) + [95.0, -50.0]
    h = numerical_histogram(values)
    assert len(h["data"]) == 20
    assert 1 <= h["below"] <= 10 and 1 <= h["above"] <= 10  # the outlier plus a thin tail
    assert max(d["count"] for d in h["data"]) < 0.2 * len(values)


def test_placeholders_above_one_percent_do_not_flatten_histogram():
    # 2% of zeros (missing-coordinate placeholders) among latitudes around 40.7
    values = [0.0] * 20 + list(numpy.linspace(40.60, 40.90, 980))
    h = numerical_histogram(values)
    assert h["below"] == 20
    assert h["data"][0]["bin_start"] > 40
    assert max(d["count"] for d in h["data"]) < 0.2 * len(values)


def test_small_integer_domain_gets_one_bar_per_value():
    h = numerical_histogram([1] * 50 + [2] * 30 + [5] * 3)
    assert h["discrete"] is True
    assert [(d["bin_start"] + d["bin_end"]) / 2 for d in h["data"]] == [1, 2, 5]
    assert [d["count"] for d in h["data"]] == [50, 30, 3]


def test_temporal_histogram_uses_calendar_days():
    start = 1356998400  # 2013-01-01 UTC
    t = [start + i * 3600 for i in range(31 * 24)]
    h = temporal_histogram(t)
    assert h["unit"] == "day"
    assert h["data"][0]["date_start"].startswith("2013-01-01")
    assert sum(d["count"] for d in h["data"]) + h["below"] + h["above"] == len(t)
    assert all(d["count"] == 24 for d in h["data"][1:-1])


def test_precision_does_not_depend_on_sampling():
    assert temporal_precision([0, 60, 120]) == "minute"
    assert temporal_precision([0, 61]) == "second"
    assert temporal_precision([0, 86400]) == "day"


def test_sampling_of_regular_series():
    s = temporal_sampling([i * 900 for i in range(100)] + [200 * 900])
    assert s["median_interval_seconds"] == 900
    assert s["regular"] is True
    assert s["gaps_over_10x_median"] == 1


@pytest.mark.parametrize("values", [[], [float("nan")]])
def test_empty_inputs(values):
    assert numerical_histogram(values) is None
    assert temporal_histogram(values) is None


@pytest.mark.parametrize("years, unit, step", [(118, "year", 5), (14, "month", 6), (3, "month", 1)])
def test_temporal_bins_stay_readable(years, unit, step):
    from profiler.histograms import MAX_TEMPORAL_BINS

    start = 1356998400 - years * 365.25 * 86400
    t = numpy.linspace(start, 1356998400, 20_000)
    h = temporal_histogram(t)
    assert (h["unit"], h["step"]) == (unit, step)
    assert len(h["data"]) <= MAX_TEMPORAL_BINS
    assert sum(d["count"] for d in h["data"]) + h["below"] + h["above"] == len(t)


def test_multi_year_and_week_bins_are_aligned():
    year = 365.25 * 86400
    h = temporal_histogram(numpy.linspace(-2034979200.0, 1665532800.0, 1000))   # 1905-07 to 2022-10
    assert h["data"][0]["date_start"].startswith("1905-01-01") and h["step"] == 5
    week = temporal_histogram(numpy.linspace(1357344000.0, 1357344000.0 + 0.5 * year, 1000))  # from Sat 2013-01-05
    assert week["step"] == 7 and week["data"][0]["date_start"].startswith("2012-12-31")  # a Monday


def test_placeholder_dates_do_not_stretch_the_range():
    real = numpy.linspace(1453161600.0, 1700000000.0, 1822)        # 2016-01-19 to 2023-11
    placeholder = numpy.full(1497, 253370764800.0)                  # 9999-09-09, 45% of the values
    h = temporal_histogram(numpy.concatenate([real, placeholder]))
    assert (h["unit"], h["step"]) == ("month", 3)   # about 8 years of real dates: quarters
    assert h["above"] == 1497 and h["data"][0]["date_start"].startswith("2016")


def test_dates_beyond_pandas_limits_do_not_crash():
    h = temporal_histogram(numpy.array([253370764800.0, 253370764800.0 + 86400 * 400, 253370764800.0 + 86400 * 800]))
    assert sum(d["count"] for d in h["data"]) + h["below"] + h["above"] == 3
