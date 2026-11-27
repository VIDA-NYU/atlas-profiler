"""Profiling a pandas DataFrame, and package-level behaviour (imports, polars compatibility)."""
import subprocess
import sys

import numpy
import pandas
import pytest

from profiler.core import process_dataset


def _trips(n, seed=0):
    rng = numpy.random.default_rng(seed)
    lat = rng.uniform(40.60, 40.85, n)
    lon = rng.uniform(-74.05, -73.75, n)
    lat[::50], lon[::50] = 0.0, 0.0  # missing coordinates recorded as (0, 0)
    start = numpy.datetime64("2013-01-01T00:00:00")
    when = start + rng.integers(0, 31 * 86400, n).astype("timedelta64[s]")
    return pandas.DataFrame({
        "pickup_latitude": lat,
        "pickup_longitude": lon,
        "pickup_datetime": pandas.to_datetime(when).strftime("%Y-%m-%d %H:%M:%S"),
        "passengers": rng.integers(1, 7, n),
    })


def test_sampled_dataframe_records_what_comes_from_the_sample():
    m = process_dataset(_trips(20_000), geo_classifier=False, plots=True, load_max_size=300_000)
    assert m["nb_profiled_rows"] < m["nb_rows"] == 20_000
    sampling = m["computed_on"].pop("sampling")
    assert sampling["method"] == "random" and sampling["size_rule"] == "load_max_size"
    assert sampling["rows"] == m["nb_profiled_rows"] and sampling["of"] == 20_000
    assert m["computed_on"] == {
        "column_types": "sample",
        "value_ranges": "sample",
        "histograms": "sample",
        "distinct_counts": "all_rows",
        "numeric_statistics": "sample",
        "spatial_coverage": "sample",
        "temporal_coverage": "sample",
        "temporal_resolution": "sample",
    }
    assert all(e["computed_on"] == "sample" for e in m["spatial_coverage"])
    # A sample spaces timestamps further apart than the data does: no spacing summary
    assert "sampling" not in m["temporal_coverage"][0]
    assert m["temporal_coverage"][0]["temporal_precision"] == "second"


def test_small_dataframe_is_profiled_whole():
    m = process_dataset(_trips(3_000), geo_classifier=False, plots=True)
    assert m["nb_profiled_rows"] == m["nb_rows"] == 3_000
    computed_on = dict(m["computed_on"])
    assert computed_on.pop("sampling")["method"] == "none"
    assert set(computed_on.values()) == {"all_rows"}
    assert m["temporal_coverage"][0]["sampling"]["distinct_timestamps"] > 2_000


def test_dataframe_spatial_coverage():
    m = process_dataset(_trips(3_000), geo_classifier=False)
    spatial = next(c for c in m["spatial_coverage"] if c["type"] == "latlong")
    assert spatial["number_zero_coordinates"] == 60
    assert spatial["number"] == 3_000 - 60
    grid = spatial["grid"]
    assert sum(n for _, _, n in grid["cells"]) + grid["outside"] == spatial["number"] - spatial["number_outside_envelope"]
    assert sum(leaf["number"] for leaf in spatial["kdtree"]["leaves"]) == spatial["number"] - spatial["number_outside_envelope"]


def test_dataframe_histograms():
    m = process_dataset(_trips(3_000), geo_classifier=False, plots=True)
    cols = {c["name"]: c for c in m["columns"]}
    assert cols["passengers"]["plot"].get("discrete") is True
    assert [round((d["bin_start"] + d["bin_end"]) / 2) for d in cols["passengers"]["plot"]["data"]] == [1, 2, 3, 4, 5, 6]
    lat = cols["pickup_latitude"]["plot"]
    assert lat["below"] >= 60  # the (0, 0) placeholders fall below the robust range
    assert lat["data"][0]["bin_start"] > 40
    assert cols["pickup_datetime"]["plot"]["unit"] == "day"


def test_profiling_without_the_classifier_does_not_import_torch():
    code = (
        "import sys, pandas\n"
        "from atlas_profiler import process_dataset\n"
        "process_dataset(pandas.DataFrame({'a': [1, 2, 3]}), geo_classifier=False)\n"
        "print('torch' in sys.modules)\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip().splitlines()[-1] == "False"


def test_missing_csv_fields_read_as_empty_strings(tmp_path):
    # polars 1.x does this with missing_utf8_is_empty_string; polars 2 removed it and nulls are filled.
    pl = pytest.importorskip("polars")
    from profiler.core import _scan_csv

    path = tmp_path / "gaps.csv"
    path.write_text("a,b\n1,\n,2\n")
    frame = _scan_csv(pl, str(path), ",").collect()
    assert frame.to_dict(as_series=False) == {"a": ["1", ""], "b": ["", "2"]}


def test_missing_classifier_dependencies_give_a_clear_error():
    code = (
        "import sys, pandas\n"
        "class Missing:  # as if PyTorch and transformers were not installed\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] in ('torch', 'transformers'):\n"
        "            raise ModuleNotFoundError(f'No module named {name!r}')\n"
        "sys.meta_path.insert(0, Missing())\n"
        "from atlas_profiler import process_dataset\n"
        "process_dataset(pandas.DataFrame({'a': [1, 2]}), geo_classifier=False)\n"
        "try:\n"
        "    process_dataset(pandas.DataFrame({'a': [1, 2]}))\n"
        "except ImportError as e:\n"
        "    print('ERROR:', e)\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert "pip install torch transformers" in out.stdout and "geo_classifier=False" in out.stdout
