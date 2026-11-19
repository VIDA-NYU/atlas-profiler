import csv
from datetime import datetime, timedelta, timezone

import pytest

pl = pytest.importorskip("polars")

from profiler.core import _csv_temporal_minmax, _load_csv_path, _sample_csv_rows  # noqa: E402

START = datetime(2013, 1, 1, tzinfo=timezone.utc)


def _write(path, rows, header):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    return str(path)


def test_large_csv_is_sampled_across_the_file(tmp_path):
    n = 20000
    path = _write(tmp_path / "big.csv", ([i, f"payload-{i:06d}" * 3] for i in range(n)), ["i", "payload"])
    data, metadata, _, _ = _load_csv_path(path, load_max_size=50_000)
    ids = data["i"].astype(int)
    assert metadata["nb_rows"] == n
    assert len(ids) < n
    # The first rows of the file would give ids 0..len-1; a random sample reaches the end of the file.
    assert ids.max() > 0.9 * n and ids.min() < 0.1 * n
    assert list(ids) == sorted(ids)  # file order is kept


def test_sample_is_reproducible(tmp_path):
    path = _write(tmp_path / "rows.csv", ([i] for i in range(1000)), ["i"])
    a = _sample_csv_rows(pl, path, ",", 1000, 50)["i"].to_list()
    b = _sample_csv_rows(pl, path, ",", 1000, 50)["i"].to_list()
    assert a == b and len(a) == 50


def _times(n, fmt):
    return [[(START + timedelta(minutes=37 * i)).strftime(fmt)] for i in range(n)]


@pytest.mark.parametrize("fmt", ["%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"])
def test_fast_temporal_minmax_matches_exact(tmp_path, fmt):
    rows = _times(500, fmt)
    path = _write(tmp_path / "t.csv", rows, ["when"])
    lo, hi = _csv_temporal_minmax(path, ",", "when")
    expected_lo = datetime.strptime(rows[0][0], fmt).replace(tzinfo=timezone.utc)
    expected_hi = datetime.strptime(rows[-1][0], fmt).replace(tzinfo=timezone.utc)
    assert (lo, hi) == (expected_lo, expected_hi)


def test_ambiguous_day_month_matches_exact_parser(tmp_path):
    # 01/02/2013 is January 2 for dateutil (month first); the result must not change with the fast path.
    rows = [["01/02/2013"], ["03/04/2013"], ["12/01/2013"]]
    path = _write(tmp_path / "md.csv", rows, ["when"])
    lo, hi = _csv_temporal_minmax(path, ",", "when")
    assert lo == datetime(2013, 1, 2, tzinfo=timezone.utc)
    assert hi == datetime(2013, 12, 1, tzinfo=timezone.utc)


def test_unparseable_values_fall_back_to_exact(tmp_path):
    rows = _times(50, "%Y-%m-%d %H:%M:%S") + [["June 2020"], [""], ["not a date"]]
    path = _write(tmp_path / "mixed.csv", rows, ["when"])
    lo, hi = _csv_temporal_minmax(path, ",", "when")
    assert lo == START
    assert hi == datetime(2020, 6, 1, tzinfo=timezone.utc)  # "June 2020", which only the exact parser reads


def test_sample_size_adapts_to_the_table():
    from profiler.core import profiled_rows

    assert profiled_rows(14_776_615, 14, 840) == 142_857          # about 2 million values
    assert profiled_rows(5_000_000, 400, 4_000) == 10_000          # wide table: the row floor
    assert profiled_rows(3_000, 14, 840) == 3_000                  # small table: every row
    assert profiled_rows(10_000_000, 2, 1_000_000) == 256          # huge rows: the memory cap
    assert profiled_rows(14_776_615, 14, 840, load_max_size=5_000_000) == 5_952  # explicit byte budget
