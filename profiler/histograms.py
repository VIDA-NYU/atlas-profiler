"""Histograms and temporal summaries that hold up on real data.

Equal-width bins over a column's full min to max put almost every value in one bin as soon as a few
values are extreme (a 95-mile taxi trip, a (0, 0) coordinate, a year-1900 date). These histograms bin a
robust range instead, report how many values fall below and above it, give integer columns with few
values one bar per value, and bin times on calendar units.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy
import pandas

NUMERICAL_BINS = 20
NUMERICAL_QUANTILES = (0.005, 0.995)
IQR_FENCE = 3.0
# Times outside these years never set a histogram's range (placeholders such as 09/09/9999); they are
# counted below or above it.
PLAUSIBLE_YEARS = (1700, 2200)  # bins span at most this many interquartile ranges beyond the quartiles
MAX_DISCRETE_VALUES = 20
TEMPORAL_QUANTILES = (0.005, 0.995)
MAX_TEMPORAL_BINS = 40  # a chart can label about this many bars

# Calendar steps, finest first: (unit, step, pandas frequency, approximate length in seconds)
_TEMPORAL_STEPS = (
    [("hour", n, f"{n}h", n * 3600) for n in (1, 2, 3, 6, 12)]
    + [("day", n, f"{n}D", n * 86400) for n in (1, 2, 7)]
    + [("month", n, f"{n}MS", n * 30.44 * 86400) for n in (1, 3, 6)]
    + [("year", n, f"{n}YS", n * 365.25 * 86400) for n in (1, 2, 5, 10, 20, 50, 100, 1000)]
)


def _robust_range(values: numpy.ndarray, quantiles: tuple[float, float]) -> tuple[float, float]:
    lo, hi = (float(q) for q in numpy.quantile(values, quantiles))
    if hi <= lo:
        lo, hi = float(values.min()), float(values.max())
    return lo, hi


def numerical_histogram(values) -> dict | None:
    """Histogram of numbers: one bar per value for small integer domains, else bins over a robust range."""
    v = numpy.asarray(values, dtype="float64")
    v = v[numpy.isfinite(v)]
    if not len(v):
        return None

    distinct = numpy.unique(v)
    if len(distinct) <= MAX_DISCRETE_VALUES and numpy.all(distinct == numpy.round(distinct)):
        counts = [int(numpy.count_nonzero(v == d)) for d in distinct]
        return {
            "type": "histogram_numerical",
            "discrete": True,
            "below": 0,
            "above": 0,
            "data": [
                {"count": c, "bin_start": float(d) - 0.5, "bin_end": float(d) + 0.5}
                for d, c in zip(distinct, counts)
            ],
        }

    # The 0.5-99.5% range, narrowed to 3 interquartile ranges around the quartiles: placeholders such as
    # (0, 0) coordinates can be a few percent of the values, more than percentiles alone would trim.
    lo, hi = _robust_range(v, NUMERICAL_QUANTILES)
    q1, q3 = (float(q) for q in numpy.quantile(v, (0.25, 0.75)))
    if q3 > q1:
        lo = max(lo, q1 - IQR_FENCE * (q3 - q1))
        hi = min(hi, q3 + IQR_FENCE * (q3 - q1))
    if hi <= lo:  # a single value
        lo, hi = lo - 0.5, hi + 0.5
    inside = v[(v >= lo) & (v <= hi)]
    counts, edges = numpy.histogram(inside, bins=NUMERICAL_BINS, range=(lo, hi))
    return {
        "type": "histogram_numerical",
        "below": int(numpy.count_nonzero(v < lo)),
        "above": int(numpy.count_nonzero(v > hi)),
        "data": [
            {"count": int(c), "bin_start": float(edges[i]), "bin_end": float(edges[i + 1])}
            for i, c in enumerate(counts)
        ],
    }


_SECONDS = {"hour": 3600, "day": 86400}


def _calendar_edges(lo: float, hi: float, unit: str, step: int) -> numpy.ndarray:
    """Bin edges in POSIX seconds covering [lo, hi], on calendar boundaries: whole hours of the day,
    weeks from Monday, quarters and half-years from January, years on multiples of ``step``.

    Computed with numpy datetimes (any year), not pandas timestamps (1677-2262 only).
    """
    t0 = numpy.datetime64(int(numpy.floor(lo)), "s")
    if unit in _SECONDS:
        size = step * _SECONDS[unit]
        day = t0.astype("datetime64[D]").astype("datetime64[s]").astype("int64")
        if unit == "hour":
            first = day + (int(lo) - day) // size * size
        elif step == 7:
            days = day // 86400
            first = (days - (days + 3) % 7) * 86400  # 1970-01-01 was a Thursday; weeks start on Monday
        else:
            first = day
        n = int((hi - first) // size) + 1
        return first + numpy.arange(n + 1, dtype="int64") * size
    if unit == "month":
        m0 = t0.astype("datetime64[M]").astype("int64")   # months since 1970-01
        m0 = (m0 + 1970 * 12) // step * step - 1970 * 12  # align on January for 3 and 6 months
        m1 = numpy.datetime64(int(numpy.floor(hi)), "s").astype("datetime64[M]").astype("int64")
        months = numpy.arange(m0, m1 + step + 1, step)
        return months.astype("datetime64[M]").astype("datetime64[s]").astype("int64")
    y0 = t0.astype("datetime64[Y]").astype("int64") + 1970
    y0 = y0 // step * step
    y1 = numpy.datetime64(int(numpy.floor(hi)), "s").astype("datetime64[Y]").astype("int64") + 1970
    years = numpy.arange(y0, y1 + step + 1, step)
    return (years - 1970).astype("datetime64[Y]").astype("datetime64[s]").astype("int64")


def temporal_histogram(timestamps) -> dict | None:
    """Histogram of times (POSIX seconds) on calendar bins over a robust range.

    The range is the 0.5-99.5% range narrowed to 3 interquartile ranges around the quartiles, as for
    numbers, computed on the times between PLAUSIBLE_YEARS; placeholder dates such as 09/09/9999 are
    counted below or above it instead of stretching it, even when they are a large share of the values. The bins are the finest calendar step (1-12 hours, 1, 2 or 7 days, 1, 3 or 6 months,
    or 1-1000 years) that keeps the histogram to at most MAX_TEMPORAL_BINS bars; ``unit`` and ``step``
    say which.
    """
    t = numpy.asarray(timestamps, dtype="float64")
    t = t[numpy.isfinite(t)]
    if not len(t):
        return None
    y0, y1 = (float((numpy.datetime64(f"{y}-01-01") - numpy.datetime64("1970-01-01")) / numpy.timedelta64(1, "s"))
              for y in PLAUSIBLE_YEARS)
    plausible = t[(t >= y0) & (t < y1)]
    basis = plausible if len(plausible) else t
    lo, hi = _robust_range(basis, TEMPORAL_QUANTILES)
    q1, q3 = (float(q) for q in numpy.quantile(basis, (0.25, 0.75)))
    if q3 > q1:
        lo = max(lo, q1 - IQR_FENCE * (q3 - q1))
        hi = min(hi, q3 + IQR_FENCE * (q3 - q1))
    for unit, step, _freq, length in _TEMPORAL_STEPS:
        last = length >= _TEMPORAL_STEPS[-1][3]
        if (hi - lo) / length + 2 > MAX_TEMPORAL_BINS and not last:
            continue
        edges = _calendar_edges(lo, hi, unit, step)
        if len(edges) - 1 <= MAX_TEMPORAL_BINS or last:
            break
    edges = edges.astype("float64")
    inside = t[(t >= edges[0]) & (t < edges[-1])]
    counts, _ = numpy.histogram(inside, bins=edges)
    iso = numpy.datetime_as_string(edges.astype("int64").astype("datetime64[s]"), unit="s")
    return {
        "type": "histogram_temporal",
        "unit": unit,
        "step": step,
        "below": int(numpy.count_nonzero(t < edges[0])),
        "above": int(numpy.count_nonzero(t >= edges[-1])),
        "data": [
            {"count": int(c), "date_start": str(iso[i]), "date_end": str(iso[i + 1])}
            for i, c in enumerate(counts)
        ],
    }


def temporal_precision(timestamps) -> str | None:
    """Finest unit the timestamps use: a time with any non-zero seconds is at ``second`` precision, etc.

    Unlike ``temporal_resolution``, this does not depend on how many timestamps were sampled.
    """
    t = numpy.asarray(timestamps, dtype="float64")
    t = t[numpy.isfinite(t)]
    if not len(t):
        return None
    if numpy.any(numpy.mod(t, 1) != 0):
        return "subsecond"
    for unit, seconds in (("second", 60), ("minute", 3600), ("hour", 86400)):
        if numpy.any(numpy.mod(t, seconds) != 0):
            return unit
    days = pandas.to_datetime(t, unit="s", utc=True)
    if numpy.any(days.day != 1):
        return "day"
    if numpy.any(days.month != 1):
        return "month"
    return "year"


def temporal_sampling(timestamps) -> dict | None:
    """How the timestamps are spaced: typical interval, regularity and gaps.

    Computed on the distinct timestamps in order. Meaningful only on all the timestamps of a column; a
    random sample of rows spaces them further apart than the data does.
    """
    t = numpy.unique(numpy.asarray(timestamps, dtype="float64"))
    t = t[numpy.isfinite(t)]
    if len(t) < 3:
        return None
    gaps = numpy.diff(t)
    median = float(numpy.median(gaps))
    steps, step_counts = numpy.unique(numpy.round(gaps, 3), return_counts=True)
    top = int(numpy.argmax(step_counts))
    regular_share = float(step_counts[top] / len(gaps))
    large = gaps > 10 * median if median > 0 else numpy.zeros(len(gaps), dtype=bool)
    largest = int(numpy.argmax(gaps))
    return {
        "distinct_timestamps": int(len(t)),
        "median_interval_seconds": median,
        "p10_interval_seconds": float(numpy.quantile(gaps, 0.10)),
        "p90_interval_seconds": float(numpy.quantile(gaps, 0.90)),
        "most_common_interval_seconds": float(steps[top]),
        "most_common_interval_share": round(regular_share, 4),
        "regular": regular_share >= 0.9,
        "gaps_over_10x_median": int(numpy.count_nonzero(large)),
        "largest_gap_seconds": float(gaps[largest]),
        "largest_gap_start": datetime.fromtimestamp(float(t[largest]), tz=timezone.utc).isoformat(),
    }
