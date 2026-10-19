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
