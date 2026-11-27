"""Written-out missing values, formatted numbers, text identifiers, separator sniffing, and geo-classifier
labels checked against the data"""

import pandas

from atlas_profiler import process_dataset
from profiler import core, types
from profiler.core import _plausible_codes, _plausible_place_names, _resolve_admin_areas, _short_names, process_column
from profiler.profile_types import identify_types, to_number


def _types(values, name="col"):
    structural, semantic, _ = identify_types([str(v) for v in values], name, None)
    return structural, set(semantic)


# --- values -------------------------------------------------------------------


def test_written_missing_values_dont_make_numbers_text():
    structural, _, meta = identify_types([str(i * 1.5) for i in range(90)] + ["NULL", "NA", "n/a", "NaN"] * 3,
                                         "col", None)
    assert structural == types.FLOAT
    assert abs(meta["missing_values_ratio"] - 12 / 102) < 1e-9


def test_formatted_numbers_are_numbers():
    assert _types([f"${i:,}.50" for i in range(1000, 1100)])[0] == types.FLOAT
    assert _types([f"{i}.5%" for i in range(100)])[0] == types.FLOAT
    assert _types([f"{i:,}" for i in range(10_000, 10_100)])[0] == types.FLOAT
    assert to_number("$1,234.50") == 1234.5 and to_number("12%") == 12.0
    assert to_number("1,5") is None  # a decimal comma is not a thousands separator


def test_text_identifiers():
    assert types.ID in _types([f"R{i:06d}X" for i in range(100)])[1]
    assert types.ID in _types([f"{i:08x}-{i:04x}-{i:04x}" for i in range(100)])[1]
    assert types.ID in _types([f"S{i}" for i in range(100)], name="sample_id")[1]
    assert types.ID not in _types([f"Station number {i}" for i in range(100)])[1]  # has spaces
    assert types.ID not in _types([f"212-555-{1000 + i}" for i in range(300)], name="phone")[1]


def test_path_mode_reads_formatted_numbers_and_missing_values(tmp_path):
    path = tmp_path / "prices.csv"
    rows = [f'"${1000 + i:,}.50",{i * 1.5}' for i in range(50)] + ["NULL,n/a"] * 5
    path.write_text("price,rate\n" + "\n".join(rows) + "\n")
    columns = {c["name"]: c for c in process_dataset(str(path), geo_classifier=False)["columns"]}
    assert columns["price"]["structural_type"] == types.FLOAT
    assert columns["price"]["min"] == 1000.5 and columns["price"]["max"] == 1049.5
    assert columns["rate"]["structural_type"] == types.FLOAT
    assert columns["rate"]["missing_values_ratio"] > 0.05


def test_separator_is_sniffed_in_path_mode(tmp_path):
    for sep in (";", "\t", "|"):
        path = tmp_path / "table.csv"  # the extension says comma
        path.write_text("\n".join(sep.join(map(str, (i, f"x{i}", i / 3))) for i in range(30)))
        meta = process_dataset(str(path), geo_classifier=False)
        assert meta["nb_columns"] == 3, sep
    path = tmp_path / "commas.tsv"
    path.write_text("a,b\n1,2\n3,4\n")
    assert process_dataset(str(path), geo_classifier=False, delimiter=",")["nb_columns"] == 2


# --- geo-classifier labels ----------------------------------------------------------


class FakeArea:
    def __init__(self, name):
        self.name = name


class FakeGeo:
    """Records the names it was asked to resolve"""

    def __init__(self):
        self.asked = None

    def resolve_names_all(self, names):
        self.asked = list(names)
        return [[FakeArea(n)] for n in names]


def test_rejects_values_that_cannot_be_place_names():
    assert not _plausible_place_names(pandas.Series(["0", "1"] * 50))  # 0/1 flag
    assert not _plausible_place_names(pandas.Series(["Yes", "No"] * 50))
    assert not _plausible_place_names(pandas.Series([str(i) for i in range(1, 6)] * 20))  # a scale
    assert not _plausible_place_names(pandas.Series(["7", "10.5", "5.7", "6.2", "13"]))  # rates
    assert _plausible_place_names(pandas.Series(["Kenya", "Uganda", "Tanzania", "Rwanda"]))
    assert _plausible_place_names(pandas.Series(["Brooklyn", "Queens", "Bronx", "Manhattan", ""]))


def test_admin_areas_resolved_once_per_distinct_name(monkeypatch):
    monkeypatch.setattr(core, "disambiguate_admin_areas", lambda resolved: (0, [r[0] for r in resolved]))
    geo = FakeGeo()
    level, areas = _resolve_admin_areas(pandas.Series(["Kenya", "Uganda", "Kenya", "Rwanda", ""]), geo)
    assert geo.asked == ["Kenya", "Rwanda", "Uganda"]
    assert level == 0 and [a.name for a in areas] == ["Kenya", "Rwanda", "Uganda"]
    assert _resolve_admin_areas(pandas.Series(["Kenya", "Uganda"]), FakeGeo()) is None  # 3 needed


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


def _column(values, label, geo=None):
    meta = {"name": "col"}
    prediction = {"label": label, "confidence": 0.9}
    process_column([str(v) for v in values], meta, plots=False, datamart_geo_data=geo, geo_prediction=prediction)
    return meta


def test_place_labels_must_fit_the_values(monkeypatch):
    flags = _column([0, 1] * 50, "country")
    assert types.ADMIN not in flags["semantic_types"] and "geo_classifier" not in flags
    scale = _column([i % 6 for i in range(100)], "state")
    assert scale["structural_type"] == types.INTEGER and types.ADMIN not in scale["semantic_types"]


def test_geographic_code_labels_need_real_codes():
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
    scale = _column([i % 6 for i in range(100)], "borough_code")
    assert types.ADMIN not in scale["semantic_types"]


def test_city_labels_need_names():
    assert _short_names(["Chicago", "Houston", "New York", "San Luis Obispo"] * 10)
    assert not _short_names(["cirugia, quimio, radia", "lo mismo que para cancer del cuerpo entero"] * 10)


def test_classifier_labelled_places_are_geocoded(monkeypatch):
    monkeypatch.setattr(core, "disambiguate_admin_areas", lambda resolved: (0, [r[0] for r in resolved]))
    meta = {"name": "Country"}
    resolved = process_column(["Kenya", "Uganda", "Rwanda"] * 10, meta, plots=False, datamart_geo_data=FakeGeo(),
                              geo_prediction={"label": "country", "confidence": 0.9})
    assert types.ADMIN in meta["semantic_types"] and meta["admin_area_level"] == 0
    assert sorted(a.name for a in resolved["admin_areas"]) == ["Kenya", "Rwanda", "Uganda"]
