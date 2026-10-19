"""Tests for the fixes made in Auctus's vendored copy (branch auctus-fixes)."""

import pandas

from profiler import core
from profiler.core import _plausible_place_names, _resolve_admin_areas


def test_rejects_values_that_cannot_be_place_names():
    assert not _plausible_place_names(pandas.Series(["0", "1"] * 50))            # 0/1 flag
    assert not _plausible_place_names(pandas.Series(["Yes", "No"] * 50))         # yes/no flag
    assert not _plausible_place_names(pandas.Series([str(i) for i in range(1, 6)] * 20))  # small codes
    assert not _plausible_place_names(pandas.Series(["7", "10.5", "5.7", "6.2", "13"]))   # rates


def test_keeps_real_place_names():
    assert _plausible_place_names(pandas.Series(["Kenya", "Uganda", "Tanzania", "Rwanda"]))
    assert _plausible_place_names(pandas.Series(["Brooklyn", "Queens", "Bronx", "Manhattan", ""]))


class FakeArea:
    def __init__(self, name):
        self.name = name


class FakeGeo:
    """Records the names it was asked to resolve."""

    def __init__(self):
        self.asked = None

    def resolve_names_all(self, names):
        self.asked = list(names)
        return [[FakeArea(n)] for n in names]


def test_admin_areas_resolved_once_per_distinct_name(monkeypatch):
    monkeypatch.setattr(core, "disambiguate_admin_areas", lambda resolved: (0, [r[0] for r in resolved]))
    geo = FakeGeo()
    level, areas = _resolve_admin_areas(pandas.Series(["Kenya", "Uganda", "Kenya", "Rwanda", ""]), geo)
    assert geo.asked == ["Kenya", "Rwanda", "Uganda"]
    assert level == 0 and [a.name for a in areas] == ["Kenya", "Rwanda", "Uganda"]


def test_admin_areas_need_three_distinct_names():
    assert _resolve_admin_areas(pandas.Series(["Kenya", "Uganda"]), FakeGeo()) is None


def test_numbers_are_never_geocoded():
    geo = FakeGeo()
    assert _resolve_admin_areas(pandas.Series([str(i) for i in range(1, 200)]), geo) is None
    assert geo.asked is None  # not even looked up


def test_most_names_must_resolve(monkeypatch):
    monkeypatch.setattr(core, "disambiguate_admin_areas", lambda resolved: (0, resolved))

    class HalfGeo(FakeGeo):
        def resolve_names_all(self, names):
            return [[FakeArea(n)] if i % 2 == 0 else [] for i, n in enumerate(names)]

    answers = pandas.Series(["cirugia", "limpieza", "ir al dentista", "NS", "radiacion", "aseo bucal"])
    assert _resolve_admin_areas(answers, HalfGeo()) is None  # 3 of 6 is not enough


def test_geographic_code_labels_need_real_codes():
    from profiler.core import _plausible_codes
    assert not _plausible_codes("borough_code", ["A", "B", "C", "D", "E"] * 10, "Q2-19")
    assert not _plausible_codes("borough_code", [str(i) for i in range(1, 300)], "Identifier")
    assert _plausible_codes("borough_code", ["MN", "BX", "BK"] * 10, "boro")
    assert not _plausible_codes("borough_code", ["1", "2", "3", "4", "5"] * 10, "LevEduc")
    assert _plausible_codes("borough_code", ["1", "2", "3", "4", "5"] * 10, "BoroCode")
    assert not _plausible_codes("state_code", ["1", "2"] * 10, "protect_stis")
    assert _plausible_codes("state_code", ["36", "34", "9"] * 10, "state_fips")
    assert _plausible_codes("state_code", ["NY", "NJ", "CT"] * 10, "st")
    assert not _plausible_codes("zip5", ["1", "2", "3"] * 10, "facilityid")
    assert _plausible_codes("zip5", ["10003", "11201", "10458"] * 10, "zip")


# --- type fixes -------------------------------------------------------------
from profiler import types
from profiler.profile_types import identify_types, to_number


def _types(values, name="col"):
    structural, semantic, meta = identify_types([str(v) for v in values], name, None)
    return structural, set(semantic)


def test_written_missing_values_dont_make_numbers_text():
    structural, _ = _types([str(i * 1.5) for i in range(90)] + ["NULL", "NA", "n/a", "NaN"] * 3)
    assert structural == types.FLOAT


def test_formatted_numbers_are_numbers():
    assert _types([f"${i:,}.50" for i in range(1000, 1100)])[0] == types.FLOAT
    assert _types([f"{i}.5%" for i in range(100)])[0] == types.FLOAT
    assert _types([f"{i:,}" for i in range(10_000, 10_100)])[0] == types.FLOAT
    assert to_number("$1,234.50") == 1234.5 and to_number("12%") == 12.0
    assert to_number("1,5") is None  # a decimal comma is not a thousands separator


def test_identifiers_detected():
    assert types.ID in _types([f"R{i:06d}X" for i in range(100)])[1]
    assert types.ID in _types([f"{i:08x}-{i:04x}-{i:04x}" for i in range(100)])[1]
    assert types.ID in _types([f"S{i}" for i in range(100)], name="sample_id")[1]
    assert types.ID not in _types([f"Station number {i}" for i in range(100)])[1]  # has spaces


def test_coded_categories_are_also_categorical():
    structural, semantic = _types([i % 11 for i in range(200)])  # a 0-10 scale
    assert structural == types.INTEGER and types.CATEGORICAL in semantic
    assert types.CATEGORICAL not in _types(list(range(200)))[1]  # a real count
