"""Inputs process_dataset accepts, and profiles that serialize as strict JSON."""
import io
import json
import math
import pathlib

import numpy
import pandas
import pytest

from profiler.core import process_dataset


def strict_json(metadata):
    return json.dumps(metadata, allow_nan=False, default=str)


def test_pathlib_path(tmp_path):
    path = tmp_path / "values.csv"
    path.write_text("a,b\n1,x\n2,y\n3,z\n")
    m = process_dataset(pathlib.Path(path), geo_classifier=False)
    assert m["nb_rows"] == 3 and [c["name"] for c in m["columns"]] == ["a", "b"]


def test_delimiter_for_file_objects_and_paths(tmp_path):
    m = process_dataset(io.StringIO("a\tb\n1\t2\n3\t4\n"), geo_classifier=False, delimiter="\t")
    assert [c["name"] for c in m["columns"]] == ["a", "b"]
    path = tmp_path / "values.txt"
    path.write_text("a;b\n1;2\n3;4\n")
    m = process_dataset(str(path), geo_classifier=False, delimiter=";")
    assert [c["name"] for c in m["columns"]] == ["a", "b"]
    tsv = tmp_path / "values.tsv"   # .tsv paths default to tabs
    tsv.write_text("a\tb\n1\t2\n")
    assert process_dataset(str(tsv), geo_classifier=False)["nb_columns"] == 2


def test_non_string_and_duplicate_labels():
    frame = pandas.DataFrame([[1.0, "a", 2.0], [3.0, "b", 4.0]], columns=[0, 1, 0])
    m = process_dataset(frame, geo_classifier=False, indexes=False)
    assert [c["name"] for c in m["columns"]] == ["0", "1", "0.1"]
    strict_json(m)


def test_date_annotation_without_parseable_values():
    frame = pandas.DataFrame({"d": ["x", "y", "z"]})
    m = process_dataset(frame, geo_classifier=False, indexes=False,
                        metadata={"columns": [{"name": "d", "semantic_types": ["http://schema.org/DateTime"]}]})
    assert m.get("temporal_coverage", []) == []
    strict_json(m)


@pytest.mark.parametrize("source", ["path", "file", "frame"])
def test_nan_and_inf_never_reach_the_profile(tmp_path, source):
    values = ["1", "2.5", "nan", "inf", "-inf", "3", "NaN", "4"] * 10
    if source == "path":
        path = tmp_path / "x.csv"
        path.write_text("x\n" + "\n".join(values) + "\n")
        data = str(path)
    elif source == "file":
        data = io.StringIO("x\n" + "\n".join(values) + "\n")
    else:
        data = pandas.DataFrame({"x": [float(v) for v in values]})
    m = process_dataset(data, geo_classifier=False, plots=True, indexes=False)
    text = strict_json(m)   # raises on NaN or infinity
    x = m["columns"][0]
    for key in ("mean", "stddev", "min", "max"):
        if key in x:
            assert math.isfinite(x[key])
    assert "Infinity" not in text


def test_profiles_of_typical_data_are_strict_json(tmp_path):
    rng = numpy.random.default_rng(0)
    n = 5_000
    frame = pandas.DataFrame({
        "latitude": rng.uniform(40.6, 40.9, n), "longitude": rng.uniform(-74.1, -73.7, n),
        "when": pandas.date_range("2013-01-01", periods=n, freq="7min").strftime("%Y-%m-%d %H:%M:%S"),
        "kind": rng.choice(["a", "b", "c"], n), "count": rng.integers(0, 5, n),
    })
    path = tmp_path / "t.csv"
    frame.to_csv(path, index=False)
    for data in (str(path), frame):
        strict_json(process_dataset(data, geo_classifier=False, plots=True, full_statistics=True, indexes=False))
