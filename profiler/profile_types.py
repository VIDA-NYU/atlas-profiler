import collections
from datetime import datetime
import dateutil.tz
import re
import regex

from . import types
from .spatial import LATITUDE, LONGITUDE, disambiguate_admin_areas
from .temporal import parse_date


_re_int = re.compile(
    r"^[+-]?[0-9]+"
    r"(?:\.0*)?"  # 4.0 and 7.000 are integers
    r"$"
)
_re_float = re.compile(
    r"^[+-]?"
    r"(?:"
    r"(?:[0-9]+\.[0-9]*)|"
    r"(?:\.[0-9]+)"
    r")"
    r"(?:[Ee][+-]?[0-9]+)?$"
)
_re_url = re.compile(
    r"^(?:(?:http|https|ftp)://|www\.)" r"""[a-zA-Z0-9$@.+,;!*~'()\[\]:/?&=#%_-]+$"""
)
_re_file = re.compile(
    "(?:^file://)"  # URI format
    "|(?:^[CD]:\\\\)"  # Windows
    "|(?:^/(?:"
    "Applications|Library|System|Users|Volumes"  # MacOS
    "|bin|boot|dev|etc|home|lib|opt|proc|root|run|sbin|srv|usr|var|tmp"  # UNIX
    ")/)"
)
_re_wkt_point = re.compile(
    r"^POINT ?\(" r"-?[0-9]{1,3}\.[0-9]{1,15}" r" " r"-?[0-9]{1,3}\.[0-9]{1,15}" r"\)$"
)
_re_wkt_polygon = re.compile(
    r"^POLYGON ?\(" r"\([0-9 .]+\)" r"(" r", ?" r"\([0-9 .]+\)" r")*" r"\)$"
)
_re_geo_combined = regex.compile(
    r"^([\p{Lu}\p{Po}0-9 ])+ \("
    r"-?[0-9]{1,3}\.[0-9]{1,15}"
    r", ?"
    r"-?[0-9]{1,3}\.[0-9]{1,15}"
    r"\)$"
)
_re_other_point = re.compile(
    r"^POINT ?\("
    r"-?[0-9]{1,3}\.[0-9]{1,15}"
    r", ?"
    r"-?[0-9]{1,3}\.[0-9]{1,15}"
    r"\)$"
)
_re_latlong_point = re.compile(
    r"^\(" r"-?[0-9]{1,3}\.[0-9]{1,15}" r", ?" r"-?[0-9]{1,3}\.[0-9]{1,15}" r"\)$"
)
_re_whitespace = re.compile(r"\s+")
# Numbers written with a currency sign, thousands separators or a percent sign.
# Thousands groups must be exactly 3 digits, so a decimal comma ("1,5") never matches.
_re_formatted_number = re.compile(
    r"^[+-]?[$€£]?\s?(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]+)?\s?%?$"
)
_re_uuid = re.compile(r"^[0-9a-fA-F]{8}-?[0-9a-fA-F]{4}-?[0-9a-fA-F]{4}-?[0-9a-fA-F]{4}-?[0-9a-fA-F]{12}$|^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}$")
_re_code = re.compile(r"^(?=.*[0-9])[A-Za-z0-9_.:/#-]{3,40}$")  # no spaces, has a digit
_re_id_name = re.compile(r"(^|[^a-z])(id|ids|identifier|uuid|guid|key|code|accession)([^a-z]|$)", re.IGNORECASE)

# Written-out missing values (pandas' default na_values): counted as empty, so
# "NULL" in some rows doesn't turn a numeric column into text.
NA_TOKENS = frozenset({"#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", "-NaN", "-nan", "1.#IND",
                       "1.#QNAN", "<NA>", "N/A", "NA", "NULL", "NaN", "None", "n/a", "nan", "null"})

# Integer columns with few small values (survey scales, Likert items) are also categorical.
MAX_CODED_CATEGORIES = 12
# Short labels repeated this many times on average are categories even when there are
# more of them than MAX_CATEGORICAL_RATIO allows (e.g. 140 tags over 760 rows).
MIN_AVG_REPEATS = 3
MAX_REPEATED_LABELS = 500
MAX_LABEL_CHARS = 60
# Names of counts and summed scores: small integers there are quantities, not codes.
_COUNT_NAME = re.compile(r"count|total|score|(^|[^a-z])(num|nb|n|number|qty)([^a-z]|$)", re.IGNORECASE)
# Column names that make 2/3-letter codes ISO country codes (not US states or currencies).
_COUNTRY_NAME = re.compile(r"countr|nation|iso|ctry|cntry", re.IGNORECASE)


def iso_country_names(values, name):
    """Country names for a column of ISO 3166 alpha-2/alpha-3 codes, or None.

    Only when the column name says country and >= 80% of the values are codes:
    "CA" is also California and "CAD" a currency.
    """
    if not _COUNTRY_NAME.search(name or ""):
        return None
    try:
        import pycountry
    except ImportError:
        return None
    out, hits = [], 0
    for value in values:
        code = str(value).strip().upper()
        country = None
        if re.fullmatch(r"[A-Z]{2}", code):
            country = pycountry.countries.get(alpha_2=code)
        elif re.fullmatch(r"[A-Z]{3}", code):
            country = pycountry.countries.get(alpha_3=code)
        if country is not None:
            hits += 1
            out.append(getattr(country, "common_name", None) or country.name)
        else:
            out.append(str(value))
    return out if values and hits >= 0.8 * len(values) else None
_INTEGER_ID_TOKENS = {"id", "ids", "identifier", "idx"}


def _id_name_tokens(name):
    """Whether a column name has an ID token: snake/kebab/space parts and camelCase
    humps (uniqueID, facility_id, ID), or is exactly "index"."""
    name = name or ""
    if name.strip().lower() == "index":
        return True
    parts = re.findall(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+", name)
    return any(part.lower() in _INTEGER_ID_TOKENS for part in parts)


MIN_ROWS_FOR_IDS = 20
MIN_ID_UNIQUENESS = 0.95


def to_number(value):
    """float for plain or formatted numbers ($1,234.50, 12%), else None."""
    text = str(value).strip()
    try:
        return float(text)
    except ValueError:
        pass
    if _re_formatted_number.match(text):
        try:
            return float(text.replace("$", "").replace("€", "").replace("£", "").replace(",", "").replace("%", "").strip())
        except ValueError:
            return None
    return None


# Tolerable ratio of unclean data
MAX_UNCLEAN = 0.02  # 2%


# Free text is multiple words
# At least 50% of it needs to have at least 4 words
TEXT_WORDS = 4
TEXT_WORDS_THRESHOLD = 0.5  # 50%


# Maximum number of different values for categorical columns
MAX_CATEGORICAL_RATIO = 0.10  # 10%


def regular_exp_count(array):
    """Count instances matching the structure of each data type, using regexes."""
    re_count = collections.Counter()

    for elem in array:
        if not elem or elem.strip() in NA_TOKENS:
            re_count["empty"] += 1
        elif _re_int.match(elem):
            re_count["int"] += 1
        elif _re_float.match(elem):
            re_count["float"] += 1
        elif _re_formatted_number.match(elem):
            re_count["float"] += 1
        elif _re_url.match(elem):
            re_count["url"] += 1
        elif _re_file.match(elem):
            re_count["file"] += 1
        elif _re_wkt_point.match(elem):
            re_count["point"] += 1
        elif _re_geo_combined.match(elem):
            re_count["geo_combined"] += 1
        elif _re_other_point.match(elem):
            re_count["other_point"] += 1
        elif _re_latlong_point.match(elem):
            re_count["latlong_point"] += 1
        elif _re_wkt_polygon.match(elem):
            re_count["polygon"] += 1
        elif len(_re_whitespace.findall(elem)) >= TEXT_WORDS - 1:
            re_count["text"] += 1
        if elem.lower() in ("0", "1", "true", "false", "y", "n", "yes", "no"):
            re_count["bool"] += 1

    return re_count


def unclean_values_ratio(c_type, re_count, num_total):
    """Count how many values don't match a given type.

    This takes into account that a valid int is also a valid float, etc.
    """
    ratio = 0
    if c_type == types.INTEGER:
        ratio = (num_total - re_count["empty"] - re_count["int"]) / num_total
    if c_type == types.FLOAT:
        ratio = (
            num_total - re_count["empty"] - re_count["int"] - re_count["float"]
        ) / num_total
    if c_type == types.GEO_POINT:
        ratio = (
            num_total
            - re_count["empty"]
            - re_count["point"]
            - re_count["geo_combined"]
            - re_count["other_point"]
            - re_count["latlong_point"]
        ) / num_total
    if c_type == types.GEO_POLYGON:
        ratio = (num_total - re_count["empty"] - re_count["polygon"]) / num_total
    if c_type == types.BOOLEAN:
        ratio = (num_total - re_count["empty"] - re_count["bool"]) / num_total
    return ratio


def parse_dates(array):
    """Parse the valid dates in an array of strings."""
    parsed_dates = []
    for elem in array:
        elem = parse_date(elem)
        if elem is not None:
            parsed_dates.append(elem)
    return parsed_dates


def identify_types(array, name, datamart_geo_data, manual=None):
    """Identify the structural type and semantic types of an array.

    :param array: The list, series, or array to inspect
    :param name: The name of this column. This is taken into account for some
        heuristics like latitude, longitude, year number.
    :param datamart_geo_data: A datamart_geo.GeoData instance to use to resolve
        named administrative territorial entities.
    :param manual: Manual information provided by the user that will be
        reconciled with the observed data.
    :return: A tuple ``(structural_type, semantic_types_dict, column_meta)``
        where `structural_type` is the detected structural type (e.g. storage
        format), `semantic_types_dict` is a dict mapping semantic types (e.g.
        meaning) to parsed values for further processing, and `column_meta`
        contains additional information about the column (not related to type).
    """
    num_total = len(array)
    column_meta = {}

    # This function let you check/count how many instances match a structure of particular data type
    re_count = regular_exp_count(array)

    # Identify structural type and compute unclean values ratio
    threshold = max(1, (1.0 - MAX_UNCLEAN) * (num_total - re_count["empty"]))
    if manual:
        structural_type = manual["structural_type"]
        column_meta["unclean_values_ratio"] = unclean_values_ratio(
            structural_type, re_count, num_total
        )
    else:
        if re_count["empty"] == num_total:
            structural_type = types.MISSING_DATA
        elif re_count["int"] >= threshold:
            structural_type = types.INTEGER
        elif re_count["int"] + re_count["float"] >= threshold:
            structural_type = types.FLOAT
        elif re_count["point"] >= threshold or re_count["other_point"] >= threshold:
            structural_type = types.GEO_POINT
            column_meta["point_format"] = "long,lat"
        elif (
            re_count["latlong_point"] >= threshold
            or re_count["geo_combined"] >= threshold
        ):
            structural_type = types.GEO_POINT
            column_meta["point_format"] = "lat,long"
        elif re_count["polygon"] >= threshold:
            structural_type = types.GEO_POLYGON
        else:
            structural_type = types.TEXT
        if structural_type != types.MISSING_DATA and structural_type != types.TEXT:
            column_meta["unclean_values_ratio"] = unclean_values_ratio(
                structural_type, re_count, num_total
            )

    # compute missing values ratio
    if structural_type != types.MISSING_DATA and re_count["empty"] > 0:
        column_meta["missing_values_ratio"] = re_count["empty"] / num_total

    distinct_values = set(e for e in array if e and e.strip() not in NA_TOKENS)
    num_present = num_total - re_count["empty"]

    semantic_types_dict = {}
    if manual:
        semantic_types = manual["semantic_types"]
        semantic_types_dict = {el: None for el in semantic_types}

        for el in semantic_types:
            if el == types.BOOLEAN:
                column_meta["unclean_values_ratio"] = unclean_values_ratio(
                    types.BOOLEAN, re_count, num_total
                )
            if el == types.DATE_TIME:
                dates = parse_dates(array)
                semantic_types_dict[types.DATE_TIME] = dates
            if el == types.ADMIN:
                if datamart_geo_data is not None and len(distinct_values) >= 3:
                    admin_areas = datamart_geo_data.resolve_names_all(array)
                    admin_areas = [r for r in admin_areas if r]
                    if admin_areas:
                        admin_areas = disambiguate_admin_areas(admin_areas)
                        if admin_areas is not None:
                            semantic_types_dict[types.ADMIN] = admin_areas
            if el == types.CATEGORICAL or el == types.INTEGER:
                # Count distinct values
                column_meta["num_distinct_values"] = len(distinct_values)
                if el == types.CATEGORICAL:
                    semantic_types_dict[types.CATEGORICAL] = distinct_values
    else:
        num_bool = re_count["bool"]
        num_text = re_count["text"]
        num_url = re_count["url"]
        num_file = re_count["file"]
        num_empty = re_count["empty"]

        # Identify booleans
        if num_bool >= threshold:
            semantic_types_dict[types.BOOLEAN] = None
            column_meta["unclean_values_ratio"] = unclean_values_ratio(
                types.BOOLEAN, re_count, num_total
            )

        if structural_type == types.TEXT:
            categorical = False

            # Identifiers: (almost) every value unique, and code-like or an id-like name
            present = [e.strip() for e in array if e and e.strip() not in NA_TOKENS]
            if (
                num_present >= MIN_ROWS_FOR_IDS
                and len(distinct_values) >= MIN_ID_UNIQUENESS * num_present
                and (
                    _re_id_name.search(name)
                    or sum(1 for e in present if _re_uuid.match(e) or _re_code.match(e)) >= 0.9 * len(present)
                )
            ):
                semantic_types_dict[types.ID] = None

            # URLs
            if num_url >= threshold:
                semantic_types_dict[types.URL] = None

            # File paths
            if num_file >= threshold:
                semantic_types_dict[types.FILE_PATH] = None

            # Administrative areas
            if datamart_geo_data is not None and len(distinct_values) >= 3:
                admin_areas = datamart_geo_data.resolve_names_all(
                    iso_country_names(list(distinct_values), name) or distinct_values)
                admin_areas = [r for r in admin_areas if r]
                if len(admin_areas) > 0.7 * len(distinct_values):

                    admin_areas = disambiguate_admin_areas(admin_areas)
                    if admin_areas is not None:
                        semantic_types_dict[types.ADMIN] = admin_areas
                        categorical = True

            # Different threshold there, we don't need all text to be many words
            text_threshold = max(
                1,
                (1.0 - TEXT_WORDS_THRESHOLD) * (num_total - re_count["empty"]),
            )
            # Labels: few distinct values (multi-word ones too, e.g. "Strongly agree"),
            # or short values each repeated several times (tags, categories).
            max_categorical = MAX_CATEGORICAL_RATIO * (len(array) - num_empty)
            n_distinct = len(distinct_values)
            labels = n_distinct <= max_categorical or (
                0 < n_distinct <= MAX_REPEATED_LABELS
                and num_present >= MIN_AVG_REPEATS * n_distinct
                and sorted(len(str(v)) for v in distinct_values)[n_distinct // 2] <= MAX_LABEL_CHARS
            )
            if types.ID in semantic_types_dict:
                column_meta["num_distinct_values"] = len(distinct_values)
            elif not categorical and not labels and num_text >= text_threshold:
                # Free text
                semantic_types_dict[types.TEXT] = None
            else:
                # Count distinct values
                column_meta["num_distinct_values"] = len(distinct_values)
                if (
                    categorical
                    or labels
                    or types.BOOLEAN in semantic_types_dict
                ):
                    semantic_types_dict[types.CATEGORICAL] = distinct_values
        elif structural_type == types.INTEGER:
            # Identify ids
            # TODO: is this enough?
            # TODO: what about false positives?
            # Whole name tokens only: prefix/suffix matching marked identify_issues,
            # paid, valid, covid_cases and composite "..._index" scores as IDs.
            if _id_name_tokens(name):
                semantic_types_dict[types.ID] = None

            # Count distinct values
            column_meta["num_distinct_values"] = len(distinct_values)

            # Coded categories: a handful of small integers (0-10 scales, 1-5 codes,
            # REDCap status codes {0, 2}), but not 0/1 flags (boolean) and not
            # counts or scores named as such.
            small = [to_number(v) for v in distinct_values]
            if (
                num_present >= MIN_ROWS_FOR_IDS
                and 1 <= len(distinct_values) <= MAX_CODED_CATEGORIES
                and all(v is not None and 0 <= v <= MAX_CODED_CATEGORIES for v in small)
                and (len(distinct_values) > 2 or not set(small) <= {0, 1})
                and not _COUNT_NAME.search(name or "")
                and types.ID not in semantic_types_dict
            ):
                semantic_types_dict[types.CATEGORICAL] = distinct_values

            # Identify years
            if name.strip().lower() == "year":
                dates = []
                for year in array:
                    try:
                        dates.append(
                            datetime(
                                int(year),
                                1,
                                1,
                                tzinfo=dateutil.tz.UTC,
                            )
                        )
                    except ValueError:
                        pass
                if len(dates) >= threshold:
                    structural_type = types.TEXT
                    semantic_types_dict[types.DATE_TIME] = dates

        # Identify lat/long
        if structural_type == types.FLOAT:
            num_lat = num_long = 0
            for elem in array:
                try:
                    elem = float(elem)
                except ValueError:
                    pass
                else:
                    if -180.0 <= float(elem) <= 180.0:
                        num_long += 1
                        if -90.0 <= float(elem) <= 90.0:
                            num_lat += 1

            if num_lat >= threshold and any(n in name.lower() for n in LATITUDE):
                semantic_types_dict[types.LATITUDE] = None
            if num_long >= threshold and any(n in name.lower() for n in LONGITUDE):
                semantic_types_dict[types.LONGITUDE] = None

        # Identify dates
        parsed_dates = parse_dates(array)

        if len(parsed_dates) >= threshold:
            semantic_types_dict[types.DATE_TIME] = parsed_dates
            if structural_type == types.INTEGER:
                # 'YYYYMMDD' format means values can be parsed as integers, but
                # that's not what they are
                structural_type = types.TEXT

    return structural_type, semantic_types_dict, column_meta


SPATIAL_STRUCTURAL_TYPES = {
    types.GEO_POINT,
    types.GEO_POLYGON,
}
SPATIAL_SEMANTIC_TYPES = {
    types.LATITUDE,
    types.LONGITUDE,
    types.ADDRESS,
    types.ADMIN,
}


def determine_dataset_type(column_structural_type, column_semantic_types):
    """Determines dataset types from columns' structural and semantic types."""
    if column_structural_type in SPATIAL_STRUCTURAL_TYPES:
        return types.DATASET_SPATIAL
    elif any(t in SPATIAL_SEMANTIC_TYPES for t in column_semantic_types):
        return types.DATASET_SPATIAL
    elif types.DATE_TIME in column_semantic_types:
        return types.DATASET_TEMPORAL
    elif types.CATEGORICAL in column_semantic_types:
        return types.DATASET_CATEGORICAL
    elif column_structural_type in (types.INTEGER, types.FLOAT):
        return types.DATASET_NUMERICAL
    else:
        return None
