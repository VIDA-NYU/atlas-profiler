"""The spatial classifier's diagnostics are written only where asked for, and logs carry no data values."""
import logging
import os

import pandas

from profiler.core import process_dataset


class FakeClassifier:
    """Stands in for HybridGeoClassifier: calls latitude/longitude by column name, no model needed."""

    def predict_batch(self, inputs, threshold=0.5):
        out = []
        for name, _values in inputs:
            label = "latitude" if "lat" in name else "longitude" if "lon" in name else "non_spatial"
            out.append({"label": label, "confidence": 0.99, "source": "ml"})
        return out


def _frame():
    return pandas.DataFrame({"lat": [40.7512, 40.7623, 40.7734] * 20, "lon": [-73.9811, -73.9922, -73.9933] * 20})


def test_default_call_writes_no_file_and_logs_no_values(tmp_path, monkeypatch, caplog):
    monkeypatch.chdir(tmp_path)
    with caplog.at_level(logging.INFO, logger="profiler"):
        m = process_dataset(_frame(), geo_classifier=FakeClassifier())
    assert m["columns"][0]["geo_classifier"]["label"] == "latitude"
    assert list(tmp_path.iterdir()) == []                # nothing written, not even output/
    assert "40.75" not in caplog.text and "-73.98" not in caplog.text
    assert "label=latitude" in caplog.text               # routine log: column, label, confidence, source


def test_diagnostics_path_writes_predictions(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "diag" / "geo.csv"
    process_dataset(_frame(), geo_classifier=FakeClassifier(), diagnostics_path=path)
    process_dataset(_frame(), geo_classifier=FakeClassifier(), diagnostics_path=path)
    rows = pandas.read_csv(path)
    assert list(rows.columns) == ["name", "values", "label", "confidence", "source"]
    assert len(rows) == 4 and set(rows["label"]) == {"latitude", "longitude"}   # appended, one row per column
    assert sorted(os.listdir(tmp_path)) == ["diag"]
