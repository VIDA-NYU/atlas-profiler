import collections
import logging
import dateutil.tz
import numpy
import os
import pandas
from pandas.errors import EmptyDataError
import string
import time
import re
import warnings

from .numerical import mean_stddev, get_numerical_ranges
from .profile_types import NA_TOKENS, identify_types, determine_dataset_type, iso_country_names, to_number
from .spatial import (
    LatLongColumn,
    Geohasher,
    nominatim_resolve_all,
    pair_latlong_columns,
    get_spatial_ranges,
    SPATIAL_RANGE_DELTA_LAT,
    SPATIAL_RANGE_DELTA_LONG,
    parse_wkt_column,
    HybridGeoClassifier,
    disambiguate_admin_areas,
)
from .temporal import get_temporal_resolution, parse_date
from .warning_tools import ignore_warnings
from .spatial_index import KDTREE_LEAVES, dense_area, density_grid, geohash_counts, kdtree_leaves
from .histograms import numerical_histogram, temporal_histogram, temporal_precision, temporal_sampling
from . import types


logger = logging.getLogger(__name__)


# Mapping from GeoClassifier labels to (structural_type, [semantic_types])
# Only spatial types are included - "non_spatial" falls through to regular workflow
GEO_CLASSIFIER_SPATIAL_MAP = {
    # Lat/Long coordinates
    "latitude": (types.FLOAT, [types.LATITUDE]),
    "longitude": (types.FLOAT, [types.LONGITUDE]),
    # Projected coordinates (e.g., State Plane, UTM)
    "x_coord": (types.FLOAT, []),
    "y_coord": (types.FLOAT, []),
    # WKT geometry types
    "point": (types.GEO_POINT, []),
    "polygon": (types.GEO_POLYGON, []),
    "multi-polygon": (types.GEO_POLYGON, []),
    "line": (types.GEO_POLYGON, []),
    "multi-line": (types.GEO_POLYGON, []),
    # Address-related
    "zip5": (types.TEXT, [types.ADDRESS]),
    "zip9": (types.TEXT, [types.ADDRESS]),
    "address": (types.TEXT, [types.ADDRESS]),
    # Administrative areas
    "borough": (
        types.TEXT,
        [types.ADMIN],
    ),  # Named boroughs (e.g., "Brooklyn", "Queens")
    "borough_code": (types.TEXT, [types.ADMIN]),  # Borough codes (numeric/alphanumeric)
    "city": (types.TEXT, [types.ADMIN]),
    "state": (types.TEXT, [types.ADMIN]),
    "state_code": (types.TEXT, [types.ADMIN]),
    "country": (types.TEXT, [types.ADMIN]),
    # NYC-specific identifiers (spatial context)
    "bbl": (types.INTEGER, [types.ID]),
    "bin": (types.INTEGER, [types.ID]),
}


RANDOM_SEED = 89

MAX_SIZE = 5000000  # 5 MB, the byte budget when load_max_size is given as MAX_SIZE
# Adaptive sample size (the default): profile about this many values (rows x columns), with at least
# MIN_PROFILED_ROWS rows, and at most MAX_SAMPLE_BYTES of in-memory data or all the rows.
TARGET_PROFILED_VALUES = 2_000_000
MIN_PROFILED_ROWS = 10_000
MAX_SAMPLE_BYTES = 256_000_000
SAMPLE_ROWS = 20

MAX_UNCLEAN_ADDRESSES = 0.20  # 20%


MAX_SKIPPED_ROWS = 6
"""Maximum number of rows to discard at the top of the file"""

HEADER_CONSISTENT_ROWS = 4
"""Stop throwing out lines when that many in a row have same number of columns
"""

MAX_GEOHASHES = 100
CSV_SAMPLE_ROWS = 1000
CSV_STATS_BATCH = 64  # columns aggregated per pass over the file, for the full-file statistics
CSV_BATCH_ROWS = 50000
# Distinct counts of CSV columns are approximate (HyperLogLog); below this they are recounted exactly.
EXACT_DISTINCT_LIMIT = 10000
# The spatial envelope of a CSV's lat/long pair spans these quantiles, not the raw min and max.
ENVELOPE_QUANTILES = (0.001, 0.999)


_re_word_split = re.compile(r"\W+")


def truncate_string(s, limit=140):
    """Truncate a string, replacing characters over the limit with "..."."""
    if len(s) <= limit:
        return s
    else:
        # Try to find a space
        space = s.rfind(" ", limit - 20, limit - 3)
        if space == -1:
            return s[: limit - 3] + "..."
        else:
            return s[:space] + "..."


def sample_value_to_string(value):
    if isinstance(value, (float, numpy.floating)):
        return f"{float(value):.6f}".rstrip("0").rstrip(".")
    return str(value).strip()


DELIMITERS = set(string.punctuation) | set(string.whitespace)
UPPER = set(string.ascii_uppercase)
LOWER = set(string.ascii_lowercase)


def expand_attribute_name(name):
    """Expand an attribute names to keywords derived from it."""
    name = name.replace("_", " ").replace("-", " ")

    word = []
    for c in name:
        if c in DELIMITERS:
            if word:
                yield "".join(word)
                word = []
            continue

        if word:
            if (word[-1] in string.digits) != (c in string.digits) or (
                word[-1] in LOWER and c in UPPER
            ):
                yield "".join(word)
                word = []

        word.append(c)

    yield "".join(word)


# Simple no-op context manager for compatibility
class NoOpContext:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _scan_csv(pl, path, sep, nrows=None):
    """Scan a CSV with every column as text, missing fields read as empty strings (not nulls).

    polars 1.x does the latter with ``missing_utf8_is_empty_string``, which polars 2 removed; there,
    nulls are filled with empty strings after the scan, which gives the same frame.
    """
    options = dict(separator=sep, infer_schema=False, low_memory=True, n_rows=nrows)
    if _SCAN_CSV_HAS_MISSING_AS_EMPTY:
        return pl.scan_csv(path, missing_utf8_is_empty_string=True, **options)
    return pl.scan_csv(path, **options).with_columns(pl.all().fill_null(""))


def _scan_csv_has_missing_as_empty():
    try:
        import inspect

        import polars as pl
    except ImportError:
        return False
    return "missing_utf8_is_empty_string" in inspect.signature(pl.scan_csv).parameters


_SCAN_CSV_HAS_MISSING_AS_EMPTY = _scan_csv_has_missing_as_empty()


def _collect_streaming(lazy_frame):
    return lazy_frame.collect(engine="streaming")


def _polars_to_pandas(frame):
    return pandas.DataFrame(frame.to_dict(as_series=False))


def _present(pl, column):
    """A column's stripped values, with empty and written-out missing values (NULL, n/a) as null"""
    value = pl.col(column).str.strip_chars()
    missing = value.is_null() | (value == "") | value.is_in(pl.Series(sorted(NA_TOKENS)).implode())
    return pl.when(missing).then(None).otherwise(value)


def _numbers(pl, value):
    """Values as floats: plain numbers, and formatted ones ($1,234.50, 12%). Thousands commas are only
    stripped from values with 3-digit groups, so a decimal comma isn't misread."""
    formatted = value.str.contains(r"^[+-]?[$€£]?\s?[0-9]{1,3}(,[0-9]{3})+(\.[0-9]+)?\s?%?$")
    cleaned = pl.when(formatted).then(value.str.replace_all(",", "")).otherwise(value)
    return pl.coalesce(
        value.cast(pl.Float64, strict=False),
        cleaned.str.replace_all(r"^([+-]?)[$€£]\s?|\s?%$", "$1").cast(pl.Float64, strict=False),
    )


def _csv_columns_stats(pl, path, sep, columns):
    """Full-file statistics of every column, CSV_STATS_BATCH columns per pass over the file

    Each column's cleaned values and numbers are computed once and then aggregated (an expression
    reused in several aggregations is evaluated once per aggregation).
    """
    out = {}
    for first in range(0, len(columns), CSV_STATS_BATCH):
        batch = columns[first : first + CSV_STATS_BATCH]
        values = _scan_csv(pl, path, sep).select(
            [_present(pl, column).alias(f"v{i}") for i, column in enumerate(batch)]
        )
        numbers = []
        for i in range(len(batch)):
            number = _numbers(pl, pl.col(f"v{i}"))
            # The same numbers as the sample (process_column): finite and inside float32 range
            numbers.append(pl.when(number.is_finite() & (number.abs() < 3.4e38)).then(number).alias(f"n{i}"))
        aggregations = []
        for i in range(len(batch)):
            value, number = pl.col(f"v{i}"), pl.col(f"n{i}")
            aggregations += [
                value.drop_nulls().approx_n_unique().alias(f"{i}:num_distinct_values"),
                number.count().alias(f"{i}:num_numeric_values"),
                number.min().alias(f"{i}:min"),
                number.max().alias(f"{i}:max"),
                number.mean().alias(f"{i}:mean"),
                number.std(ddof=0).alias(f"{i}:stddev"),
                value.drop_nulls().head(CSV_SAMPLE_ROWS).implode().alias(f"{i}:sample_values"),
            ]
        row = _collect_streaming(values.with_columns(numbers).select(aggregations)).row(0, named=True)
        for i, column in enumerate(batch):
            stats = {key.split(":", 1)[1]: v for key, v in row.items() if key.split(":", 1)[0] == str(i)}
            out[column] = _finish_column_stats(stats)
    return out


def _finish_column_stats(stats):
    sample_values = []
    seen = set()
    for item in stats["sample_values"] or []:
        item = sample_value_to_string(item)
        if item and item not in seen:
            seen.add(item)
            sample_values.append(item)
            if len(sample_values) == 3:
                break
    stats["num_distinct_values"] = int(stats["num_distinct_values"] or 0)
    stats["num_distinct_values_is_approximate"] = True
    stats["sample_values"] = sample_values
    return stats


def profiled_rows(nb_rows, nb_columns, avg_row_size, load_max_size=None):
    """How many rows to profile.

    With ``load_max_size`` (bytes), as many rows as fit in it. Otherwise the size adapts to the table:
    about TARGET_PROFILED_VALUES values (rows x columns), at least MIN_PROFILED_ROWS rows, at most
    MAX_SAMPLE_BYTES of in-memory data. Never more than the table has.
    """
    avg_row_size = max(float(avg_row_size), 1.0)
    if load_max_size is not None:
        wanted = int(load_max_size / avg_row_size)
    else:
        wanted = max(MIN_PROFILED_ROWS, TARGET_PROFILED_VALUES // max(int(nb_columns), 1))
        wanted = min(wanted, int(MAX_SAMPLE_BYTES / avg_row_size))
    return max(1, min(int(nb_rows), wanted))


def _exact_small_distinct_counts(pl, path, sep, full_data_stats):
    """Replace approximate distinct counts by exact ones where they are small (one more pass)."""
    small = [
        column for column, stats in full_data_stats.items()
        if stats["num_distinct_values"] <= EXACT_DISTINCT_LIMIT
    ]
    if not small:
        return
    exprs = []
    for column in small:
        value = _present(pl, column)
        exprs.append(value.drop_nulls().n_unique().alias(column))
    exact = _collect_streaming(_scan_csv(pl, path, sep).select(exprs)).row(0, named=True)
    for column in small:
        full_data_stats[column]["num_distinct_values"] = int(exact[column])
        full_data_stats[column]["num_distinct_values_is_approximate"] = False


def _sniff_separator(path):
    """Delimiter from the first 64 KiB (, ; tab |); by extension if that fails"""
    import csv

    default = "\t" if path.lower().endswith((".tsv", ".tab")) else ","
    with open(path, encoding="utf-8", errors="replace", newline="") as fh:
        head = fh.read(64 * 1024)
    try:
        return csv.Sniffer().sniff(head, delimiters=",;\t|").delimiter
    except csv.Error:
        return default


def _load_csv_path(data, load_max_size, delimiter=None):
    try:
        import polars as pl
    except ImportError as exc:
        raise ImportError(
            "Polars is required to profile CSV/TSV paths. Install project dependencies first."
        ) from exc

    path = str(data)
    if not os.path.exists(path):
        raise ValueError("data file does not exist")

    metadata = {"size": os.path.getsize(path)}
    metadata["_csv_path"] = path
    logger.info("File size: %r bytes", metadata["size"])

    sep = delimiter or _sniff_separator(path)
    metadata["_csv_sep"] = sep
    scan = _scan_csv(pl, path, sep)
    try:
        column_names = scan.collect_schema().names()
    except pl.exceptions.NoDataError as exc:  # an empty file: same result as an empty file object
        raise EmptyDataError("empty CSV") from exc
    metadata["nb_rows"] = int(
        _collect_streaming(scan.select(pl.len().alias("rows"))).item()
    )

    full_data_stats = _csv_columns_stats(pl, path, sep, column_names)
    _exact_small_distinct_counts(pl, path, sep, full_data_stats)

    nrows = None
    if metadata["nb_rows"]:
        # In-memory size of a row, from the first rows, to keep the sample under the memory cap
        first = _polars_to_pandas(_collect_streaming(_scan_csv(pl, path, sep, CSV_SAMPLE_ROWS)))
        avg_row_size = first.memory_usage(deep=True).sum() / max(len(first), 1)
        wanted = profiled_rows(metadata["nb_rows"], len(column_names), avg_row_size, load_max_size)
        if wanted < metadata["nb_rows"]:
            nrows = wanted
            logger.info("Profiling a random sample of %d of %d rows", nrows, metadata["nb_rows"])

    if nrows is None:
        data = _collect_streaming(_scan_csv(pl, path, sep))
    else:
        data = _sample_csv_rows(pl, path, sep, metadata["nb_rows"], nrows)
    # Files written with ", " between fields carry a space before each value, which would make every
    # number look like text to the type detection (the full-file statistics strip values too).
    data = _polars_to_pandas(data.with_columns(pl.all().str.strip_chars()))

    return data, metadata, column_names, full_data_stats


def _sample_csv_rows(pl, path, sep, nb_rows, nrows):
    """A uniform random sample of ``nrows`` rows of the file, in file order.

    Taking the first rows instead would profile whatever the file happens to start with (one day of a
    month of trips, one borough of a city), so the row numbers are drawn at random (seeded, so a profile
    is reproducible) and the rows are read while streaming through the file.
    """
    rand = numpy.random.RandomState(RANDOM_SEED)
    picked = numpy.sort(rand.choice(nb_rows, size=min(nrows, nb_rows), replace=False))
    scan = _scan_csv(pl, path, sep).with_row_index("__atlas_row__").with_columns(
        pl.col("__atlas_row__").cast(pl.UInt64)
    )
    wanted = pl.LazyFrame({"__atlas_row__": pl.Series(picked.astype("uint64"), dtype=pl.UInt64)})
    rows = _collect_streaming(scan.join(wanted, on="__atlas_row__", how="semi"))
    return rows.sort("__atlas_row__").drop("__atlas_row__")


def _csv_latlong_coverage(path, sep, col_lat, col_long, spatial_leaves=KDTREE_LEAVES):
    import polars as pl

    # Strip before casting: " 40.7" (after a ", " separator) would otherwise be dropped as not a number.
    lat = pl.col(col_lat.name).str.strip_chars().cast(pl.Float64, strict=False).alias("_lat")
    long = pl.col(col_long.name).str.strip_chars().cast(pl.Float64, strict=False).alias("_long")
    scan = _scan_csv(pl, path, sep).select(lat, long).filter(
        pl.col("_lat").is_between(-90.0, 90.0, closed="none")
        & pl.col("_long").is_between(-180.0, 180.0, closed="none")
    )

    # (0, 0) is a common placeholder for missing coordinates: count it apart, keep it out of the coverage.
    zero = (pl.col("_lat") == 0.0) & (pl.col("_long") == 0.0)
    zeros = int(_collect_streaming(scan.select(zero.sum())).item() or 0)
    scan = scan.filter(~zero)

    # A few stray geocodes far away would stretch a min/max box over a continent; use percentiles.
    lo, hi = ENVELOPE_QUANTILES
    bounds = _collect_streaming(scan.select(
        pl.len().alias("count"),
        pl.col("_lat").quantile(lo).alias("min_lat"), pl.col("_lat").quantile(hi).alias("max_lat"),
        pl.col("_long").quantile(lo).alias("min_long"), pl.col("_long").quantile(hi).alias("max_long"),
    )).row(0, named=True)
    count = int(bounds["count"] or 0)
    if not count:
        return None
    min_lat, max_lat = float(bounds["min_lat"]), float(bounds["max_lat"])
    min_long, max_long = float(bounds["min_long"]), float(bounds["max_long"])

    # Geohash cells only from the points inside the envelope. The tree keeps at most MAX_GEOHASHES cells
    # per level, and a few thousand stray points scattered over the world would use that up at every
    # fine level, leaving cells hundreds of kilometres wide for the data itself.
    inside = (
        pl.col("_lat").is_between(min_lat, max_lat) & pl.col("_long").is_between(min_long, max_long)
    )
    values = _collect_streaming(scan.filter(inside).select("_lat", "_long")).to_numpy()
    in_cells = len(values)
    hashes = geohash_counts(values[:, 0], values[:, 1], MAX_GEOHASHES)
    leaves = kdtree_leaves(values[:, 1], values[:, 0], (min_long, min_lat, max_long, max_lat), spatial_leaves)
    grid = density_grid(values[:, 1], values[:, 0], dense_area(leaves)) if leaves else None
    del values

    if min_long == max_long:
        min_long -= SPATIAL_RANGE_DELTA_LONG
        max_long += SPATIAL_RANGE_DELTA_LONG
    if min_lat == max_lat:
        min_lat -= SPATIAL_RANGE_DELTA_LAT
        max_lat += SPATIAL_RANGE_DELTA_LAT

    return {
        "type": "latlong",
        "column_names": [col_lat.name, col_long.name],
        "column_indexes": [col_lat.index, col_long.index],
        "geohashes4": hashes,
        "kdtree": _kdtree_json(leaves, spatial_leaves),
        **({"grid": grid} if grid else {}),
        "ranges": [
            {
                "range": {
                    "type": "envelope",
                    "coordinates": [[min_long, max_lat], [max_long, min_lat]],
                }
            }
        ],
        "number": count,
        "number_outside_envelope": count - in_cells,
        **({"number_zero_coordinates": zeros} if zeros else {}),
    }


def _kdtree_json(leaves, spatial_leaves):
    return {
        "split": "median",
        "max_leaves": int(spatial_leaves),
        "leaves": [
            {"bounds": [round(x0, 7), round(y0, 7), round(x1, 7), round(y1, 7)], "number": n}
            for x0, y0, x1, y1, n in leaves
        ],
    }


TEMPORAL_CHECK_VALUES = 200


def _as_utc(dt):
    """Timezone-aware UTC datetime; naive values are taken as UTC, as ``parse_date`` does."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=dateutil.tz.UTC)
    return dt.astimezone(dateutil.tz.UTC)


def _csv_temporal_minmax_polars(pl, path, sep, column):
    """Min and max of a date/time column parsed natively by polars, or ``None`` when that is not safe.

    Polars infers one format for the column. The result is used only when it parsed every non-empty value
    and gives the same instants as ``parse_date`` on the first values, which rules out a day/month mix-up
    (01/02 read as February 1 instead of January 2) and partial dates; otherwise the caller falls back to
    the exact, slower parse.
    """
    value = pl.col(column).str.strip_chars()
    value = pl.when(value == "").then(None).otherwise(value)
    parsed = value.str.to_datetime(strict=False, time_unit="us")
    try:
        row = _collect_streaming(
            _scan_csv(pl, path, sep).select(
                value.count().alias("values"),
                parsed.count().alias("parsed"),
                parsed.min().alias("min"),
                parsed.max().alias("max"),
            )
        ).row(0, named=True)
        check = _collect_streaming(
            _scan_csv(pl, path, sep, TEMPORAL_CHECK_VALUES * 4)
            .select(value.alias("raw"), parsed.alias("parsed"))
            .drop_nulls("raw")
            .head(TEMPORAL_CHECK_VALUES)
        )
    except Exception:  # an unexpected format or polars version: use the exact path
        logger.info("Fast temporal parse failed for %r, using the exact parser", column)
        return None

    if not row["values"] or row["parsed"] != row["values"] or row["min"] is None:
        return None
    for raw, fast in check.iter_rows():
        exact = parse_date(str(raw).strip())
        if exact is None or fast is None or _as_utc(fast) != exact:
            return None
    return _as_utc(row["min"]), _as_utc(row["max"])


def _csv_temporal_timestamps(path, sep, column):
    """Distinct POSIX timestamps of a CSV column, or ``None`` when polars cannot parse it safely."""
    import polars as pl

    if _csv_temporal_minmax_polars(pl, path, sep, column) is None:
        return None
    value = pl.col(column).str.strip_chars()
    value = pl.when(value == "").then(None).otherwise(value)
    parsed = value.str.to_datetime(strict=False, time_unit="us")
    # Epoch microseconds are absolute instants, so time zones need no handling here.
    frame = _collect_streaming(
        _scan_csv(pl, path, sep).select(parsed.dt.epoch("us").alias("t")).drop_nulls().unique()
    )
    return frame["t"].to_numpy().astype("float64") / 1e6


def _csv_temporal_minmax(path, sep, column):
    import polars as pl

    fast = _csv_temporal_minmax_polars(pl, path, sep, column)
    if fast is not None:
        return fast

    scan = _scan_csv(pl, path, sep).select(pl.col(column))
    min_dt = max_dt = None
    count = 0

    for batch in scan.collect_batches(chunk_size=CSV_BATCH_ROWS, engine="streaming"):
        for value in batch[column].to_list():
            dt = parse_date(str(value).strip())
            if dt is None:
                continue
            count += 1
            if min_dt is None or dt < min_dt:
                min_dt = dt
            if max_dt is None or dt > max_dt:
                max_dt = dt

    if not count:
        return None
    return min_dt, max_dt


def _drop_internal_metadata(metadata):
    metadata.pop("_csv_path", None)
    metadata.pop("_csv_sep", None)


def load_data(data, load_max_size=None, indexes=True, delimiter=None):
    """Load data from file path, file object, or DataFrame.

    Returns sampled data for profiling along with full-data statistics.
    """
    metadata = {}
    full_data_stats = {}

    # Step 1: Convert file path/file object to DataFrame
    if isinstance(data, os.PathLike):
        data = os.fspath(data)
    if isinstance(data, bytes):
        data = os.fsdecode(data)
    if isinstance(data, str):
        data, metadata, column_names, full_data_stats = _load_csv_path(
            data, load_max_size, delimiter
        )

    elif hasattr(data, "read"):
        # File object
        data.seek(0, 2)
        metadata["size"] = data.tell()
        data.seek(0, 0)
        data = pandas.read_csv(data, dtype=str, na_filter=False, sep=delimiter or ",")

    elif not isinstance(data, pandas.DataFrame):
        raise TypeError(
            "data should be a filename, a file object, or a pandas.DataFrame"
        )

    # Step 2: Process DataFrame (common path for all input types)
    if indexes and (
        data.index.dtype != numpy.int64
        or not pandas.Index(numpy.arange(len(data))).equals(data.index)
    ):
        data = data.reset_index()

    metadata.setdefault("nb_rows", len(data))
    logger.info("DataFrame: %d rows, %d columns", data.shape[0], data.shape[1])

    if not full_data_stats:
        # Compute stats on data before sampling (cheap operations)
        # Also extract 3 non-null sample values for geo classifier
        for col in data.columns:
            # Count distinct values
            full_data_stats[col] = {"num_distinct_values": data[col].nunique()}

            # Extract 3 unique non-null sample values from full dataset
            sample_values = []
            seen = set()
            col_series = data[col]
            for v in col_series:
                v_str = sample_value_to_string(v)
                if v_str and v_str not in ("", "nan", "None") and v_str not in seen:
                    seen.add(v_str)
                    sample_values.append(v_str)
                    if len(sample_values) >= 3:
                        break
            full_data_stats[col]["sample_values"] = sample_values

    # Sample if the DataFrame is larger than the sample size
    avg_row_size = data.memory_usage(deep=True).sum() / max(len(data), 1)
    max_rows = profiled_rows(len(data), data.shape[1], avg_row_size, load_max_size)
    if len(data) > max_rows:
        logger.info("Sampling %d of %d rows for profiling", max_rows, len(data))
        data = data.sample(n=max_rows, random_state=RANDOM_SEED)

    metadata["nb_profiled_rows"] = len(data)

    # Convert to string dtype (workaround for pandas nan-as-'nan' bug)
    data = data.astype(object).fillna("").astype(str)

    return data, metadata, data.columns, full_data_stats


# Geo-classifier labels that mean "this column holds place names". Code-like labels (zip5,
# borough_code, ...) legitimately hold numbers and aren't checked here.
PLACE_NAME_LABELS = {"borough", "city", "state", "country"}
# Labels whose places are all in datamart-geo, so the gazetteer must confirm them
GAZETTEER_LABELS = {"country", "state", "borough"}
MIN_RESOLVED_SHARE = 0.7  # as in identify_types()
_YES_NO = {"yes", "no", "y", "n", "true", "false", "t", "f"}
_NUMBER_RE = re.compile(r"[-+]?\d+(\.\d+)?")


def _place_values(array):
    return [str(v).strip() for v in array if str(v).strip() not in ("", "nan", "None")]


def _plausible_place_names(array):
    """False when values can't be place names: mostly numbers, yes/no, or <= 2 distinct.

    The geo-classifier labels such columns as places (0/1 flags, survey codes, rates), which then
    turn into wrong admin areas and map boxes.
    """
    distinct = set(_place_values(array))
    if len(distinct) <= 2:
        return False
    lowered = {v.lower() for v in distinct}
    if len(lowered & _YES_NO) == len(lowered):
        return False
    numeric = sum(1 for v in distinct if _NUMBER_RE.fullmatch(v))
    return numeric <= len(distinct) / 2


def _resolve_admin_areas(array, datamart_geo_data, name=None):
    """(level, areas) for a place-name column, or None if it doesn't look like one.

    As in identify_types(): numbers are dropped (a column of row numbers once resolved to provinces
    of Iran), at least 3 distinct names are needed, and more than MIN_RESOLVED_SHARE must resolve.
    """
    distinct = sorted(set(_place_values(array)))
    distinct = iso_country_names(distinct, name) or distinct
    distinct = [v for v in distinct if not _NUMBER_RE.fullmatch(v)]
    if len(distinct) < 3:
        return None
    resolved = [r for r in datamart_geo_data.resolve_names_all(distinct) if r]
    if len(resolved) <= MIN_RESOLVED_SHARE * len(distinct):
        return None
    return disambiguate_admin_areas(resolved)


def _short_names(array, max_words=4):
    """Values that read like names: mostly at most ``max_words`` words, no commas"""
    values = _place_values(array)
    if not values:
        return False
    namelike = sum(1 for v in values if "," not in v and len(v.split()) <= max_words)
    return namelike >= 0.9 * len(values)


CODE_LABELS = {"zip5", "zip9", "borough_code", "state_code"}
_US_STATE_CODES = set(
    "AL AK AZ AR CA CO CT DE DC FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ NM NY NC ND "
    "OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY PR GU VI AS MP".split()
)
_BOROUGH_CODES = {"1", "2", "3", "4", "5", "MN", "BX", "BK", "QN", "SI"}


def _plausible_codes(label, array, name):
    """Whether a column labelled as a geographic code holds such codes

    The geo-classifier labels survey answers (A-E, 1-5 scales) and row numbers as borough or state
    codes.
    """
    values = _place_values(array)
    if not values:
        return False
    distinct = set(values)
    if label in ("zip5", "zip9"):
        return sum(1 for v in values if re.fullmatch(r"\d{5}(-?\d{4})?", v)) >= 0.8 * len(values)
    if label == "borough_code":
        upper = {v.upper() for v in distinct}
        if not upper <= _BOROUGH_CODES or len(distinct) < 2:
            return False
        # 1-5 is also every survey scale: numeric codes need a borough-like column name
        return not upper <= {"1", "2", "3", "4", "5"} or bool(re.search(r"boro", name or "", re.IGNORECASE))
    if label == "state_code":
        if {v.upper() for v in distinct} <= _US_STATE_CODES:
            return len(distinct) >= 2
        # FIPS numbers are only believable when the column name says so
        return bool(re.search(r"state|fips", name or "", re.IGNORECASE)) and all(
            v.isdigit() and 1 <= int(v) <= 78 for v in distinct
        )
    return True


def process_column(
    array,
    column_meta,
    *,
    manual=None,
    plots=True,
    coverage=True,
    datamart_geo_data=None,
    nominatim=None,
    geo_prediction=None,  # Pre-computed from batch prediction
):
    structural_type = None
    semantic_types_dict = {}
    additional_meta = {}
    used_geo_prediction = False

    # Use pre-computed geo_prediction if available and no manual override
    if geo_prediction is not None and manual is None:
        label = geo_prediction["label"]
        admin_areas = None
        place_names = label in PLACE_NAME_LABELS and _plausible_place_names(array)
        if place_names and datamart_geo_data is not None:
            admin_areas = _resolve_admin_areas(array, datamart_geo_data, column_meta.get("name"))
        implausible = (
            (label in PLACE_NAME_LABELS and not place_names)
            # countries, states and boroughs are in datamart-geo, so it must confirm them; many cities
            # aren't, so city labels only need short name-like values
            or (label in GAZETTEER_LABELS and datamart_geo_data is not None and admin_areas is None)
            or (label == "city" and not _short_names(array))
            or (label in CODE_LABELS and not _plausible_codes(label, array, column_meta.get("name")))
        )
        if implausible:
            logger.info(
                "Ignoring geo_classifier label %s for column %r: its values don't fit it",
                label,
                column_meta.get("name"),
            )
        elif label in GEO_CLASSIFIER_SPATIAL_MAP:
            # Geo classifier identified a spatial type
            used_geo_prediction = True
            structural_type, semantic_list = GEO_CLASSIFIER_SPATIAL_MAP[label]
            semantic_types_dict = {st: None for st in semantic_list}
            if types.ADMIN in semantic_types_dict and admin_areas is not None:
                # identify_types() is skipped on this path, and with it the datamart-geo
                # resolution that gives admin areas their coverage
                semantic_types_dict[types.ADMIN] = admin_areas
            additional_meta["geo_classifier"] = {
                "label": label,
                "confidence": geo_prediction["confidence"],
                "source": geo_prediction.get("source", "ml"),
            }

    # If geo prediction wasn't used, use regular identify_types
    if not used_geo_prediction:
        structural_type, semantic_types_dict, additional_meta = identify_types(
            array, column_meta["name"], datamart_geo_data, manual
        )

    # Log column type with source information
    if used_geo_prediction:
        geo_info = additional_meta.get("geo_classifier", {})
        label = geo_info.get("label", "unknown")
        confidence = geo_info.get("confidence", 0.0)
        source = geo_info.get("source", "ml")

        # Data values stay out of the logs (they can be collected or shipped elsewhere)
        column_name = column_meta.get("name", "unknown")
        logger.info(
            "Column type %s [%s] (from geo_classifier: column=%r, label=%s, confidence=%.4f, source=%s)",
            structural_type,
            ", ".join(semantic_types_dict),
            column_name,
            label,
            confidence,
            source,
        )
    else:
        logger.info(
            "Column type %s [%s]",
            structural_type,
            ", ".join(semantic_types_dict),
        )

    # Set structural type
    column_meta["structural_type"] = structural_type
    # Add semantic types to the ones already present
    sem_types = column_meta.setdefault("semantic_types", [])
    for sem_type in semantic_types_dict:
        if sem_type not in sem_types:
            sem_types.append(sem_type)
    # Insert additional metadata
    column_meta.update(additional_meta)

    # Resolved values are returned so they can be used again to compute spatial
    # coverage
    resolved = {}

    # Compute ranges for numerical data
    if structural_type in (types.INTEGER, types.FLOAT) and (coverage or plots):
        # Get numerical values needed for either ranges or plot
        numerical_values = []
        for e in array:
            e = to_number(e)  # also $1,234.50 and 12%
            if e is not None and -3.4e38 < e < 3.4e38:  # Overflows in ES
                numerical_values.append(e)

        # Compute ranges from numerical values
        if coverage:
            column_meta["mean"], column_meta["stddev"] = mean_stddev(numerical_values)

            ranges = get_numerical_ranges(numerical_values)
            if ranges:
                column_meta["coverage"] = ranges

        # Compute histogram from numerical values
        if plots:
            histogram = numerical_histogram(numerical_values)
            if histogram is not None:
                column_meta["plot"] = histogram

    if types.DATE_TIME in semantic_types_dict:
        datetimes = semantic_types_dict[types.DATE_TIME]
        resolved["datetimes"] = datetimes
        timestamps = numpy.empty(
            len(datetimes),
            dtype="float64",
        )
        for j, dt in enumerate(datetimes):
            timestamps[j] = dt.timestamp()
        resolved["timestamps"] = timestamps

        # Compute histogram from temporal values
        if plots and "plot" not in column_meta:
            histogram = temporal_histogram(timestamps)
            if histogram is not None:
                column_meta["plot"] = histogram

    # Compute histogram from categorical values
    if plots and types.CATEGORICAL in semantic_types_dict:
        counter = collections.Counter()
        for value in array:
            if not value:
                continue
            counter[value] += 1
        counts = counter.most_common(5)
        counts = sorted(counts)
        column_meta["plot"] = {
            "type": "histogram_categorical",
            "data": [
                {
                    "bin": value,
                    "count": count,
                }
                for value, count in counts
            ],
        }

    # Compute histogram from textual values
    if plots and types.TEXT in semantic_types_dict and "plot" not in column_meta:
        counter = collections.Counter()
        for value in array:
            for word in _re_word_split.split(value):
                word = word.lower()
                if word:
                    counter[word] += 1
        counts = counter.most_common(5)
        column_meta["plot"] = {
            "type": "histogram_text",
            "data": [
                {
                    "bin": value,
                    "count": count,
                }
                for value, count in counts
            ],
        }

    # Resolve addresses into coordinates
    if (
        nominatim is not None
        and structural_type == types.TEXT
        and types.TEXT in semantic_types_dict
        and types.ADMIN not in semantic_types_dict
    ):
        locations, non_empty = nominatim_resolve_all(
            nominatim,
            array,
        )
        if non_empty > 0:
            unclean_ratio = 1.0 - len(locations) / non_empty
            if unclean_ratio <= MAX_UNCLEAN_ADDRESSES:
                resolved["addresses"] = locations
                if types.ADDRESS not in column_meta["semantic_types"]:
                    column_meta["semantic_types"].append(types.ADDRESS)

    # Set level of administrative areas
    if types.ADMIN in semantic_types_dict:
        admin_value = semantic_types_dict[types.ADMIN]
        if admin_value is not None:
            # disambiguate_admin_areas returns (level, areas) tuple or just areas
            if isinstance(admin_value, tuple) and len(admin_value) == 2:
                level, areas = admin_value
            else:
                # Fallback: admin_value is just the areas list
                level, areas = None, admin_value
            if level is not None:
                column_meta["admin_area_level"] = level
            resolved["admin_areas"] = areas

    return resolved


def process_dataset(
    data,
    geo_classifier=True,
    geo_classifier_threshold=0.5,
    include_sample=False,
    coverage=True,
    plots=False,
    indexes=True,
    load_max_size=None,
    metadata=None,
    nominatim=None,
    datamart_geo_data=None,
    spatial_leaves=KDTREE_LEAVES,
    full_statistics=False,
    diagnostics_path=None,
    delimiter=None,
    **kwargs,
):
    """Compute all metafeatures from a dataset.

    :param data: path to dataset, or file object, or DataFrame
    :param geo_classifier: ``True`` to enable geo_classifier
    :param geo_classifier_threshold: Confidence threshold for geo_classifier
        predictions (default: 0.85). Predictions below this threshold will be
        treated as "non_spatial".
    :param include_sample: Set to True to include a few random rows to the
        result. Useful to present to a user.
    :param coverage: Whether to compute data ranges
    :param plots: Whether to compute plots
    :param indexes: Whether to include indexes. If True (the default), the
        input is a DataFrame, and it has index(es) different from the default
        range, they will appear in the result with the columns.
    :param load_max_size (bytes): Size of the in-memory data to profile; bigger data is randomly
        sampled to fit. By default the sample size adapts to the table: about 2 million values
        (rows x columns, see ``profiled_rows``), at least 10,000 rows, at most 256 MB in memory.
        This is different from the sample data included in the result.
    :param full_statistics: With ``plots=True``, compute the histograms over every row instead of the
        profiled sample: numbers and times (when they parse safely) with polars for a CSV/TSV path, or
        pandas for a DataFrame, and exact most-frequent values for categories. Column types, value
        ranges and word counts of free text still come from the sample. Each histogram says where it
        comes from in its own ``computed_on``.
    :param delimiter: Field separator of a CSV/TSV path or file object. By default a comma, or a tab
        for paths ending in ``.tsv`` or ``.tab``.
    :param diagnostics_path: Path of a CSV file to which the spatial classifier's predictions are
        appended, with the sampled values it saw for each spatial column. Off by default: profiling
        writes no file unless asked, and the logs never include data values.
    :param spatial_leaves: Number of leaves of the k-d tree that summarizes each lat/long pair
        (``kdtree`` in the spatial coverage): the points are split at the median, alternating
        longitude and latitude, so the cells follow the density of the data. Default 4096.
    :param metadata: The metadata provided by the discovery plugin (might be
        very limited).
    :param nominatim: URL of the Nominatim server
    :param datamart_geo_data: ``True`` or a datamart_geo.GeoData instance to use to
        resolve named administrative territorial entities
    :return: JSON structure (dict)
    """
    # Track runtime for each pipeline step
    pipeline_start = time.perf_counter()
    step_times = {}

    if "sample_size" in kwargs:
        warnings.warn(
            "Argument 'sample_size' is deprecated, use 'load_max_size'",
            DeprecationWarning,
        )
        load_max_size = kwargs.pop("sample_size")
    if kwargs:
        raise TypeError(
            "process_dataset() got unexpected keyword argument %r" % next(iter(kwargs))
        )

    if datamart_geo_data is True:
        from datamart_geo import GeoData

        datamart_geo_data = GeoData.from_local_cache()

    if geo_classifier is True:
        try:
            from .geo_model import GeoClassifier  # imports PyTorch, so only when needed
        except ImportError as exc:
            raise ImportError(
                "The spatial classifier (geo_classifier=True, the default) needs PyTorch and "
                "transformers: pip install torch transformers. To profile without it, pass "
                "geo_classifier=False."
            ) from exc

        geo_classifier = HybridGeoClassifier(GeoClassifier())

    if metadata is None:
        metadata = {}

    # =========================================================================
    # STEP 1: Load data
    # =========================================================================
    step_start = time.perf_counter()
    logger.info("[STEP 1/6] Loading data...")

    if isinstance(data, pandas.DataFrame):
        data = _string_labels(data)
    full_frame = data if (full_statistics and isinstance(data, pandas.DataFrame)) else None
    try:
        data, file_metadata, column_names, full_data_stats = load_data(
            data,
            load_max_size=load_max_size,
            indexes=indexes,
            delimiter=delimiter,
        )
    except EmptyDataError:
        logger.warning("Dataframe is empty!")
        metadata["nb_rows"] = 0
        metadata["nb_profiled_rows"] = 0
        metadata["columns"] = []
        metadata["types"] = []
        return metadata

    step_times["1_load_data"] = time.perf_counter() - step_start
    logger.info(
        "[STEP 1/6] Data loaded in %.3fs (%d rows, %d columns)",
        step_times["1_load_data"],
        data.shape[0],
        data.shape[1],
    )

    metadata.update(file_metadata)
    csv_path = metadata.get("_csv_path")
    csv_sep = metadata.get("_csv_sep")
    metadata["nb_profiled_rows"] = data.shape[0]
    metadata["nb_columns"] = data.shape[1]

    if "columns" in metadata:
        columns = metadata["columns"]
        logger.info("Using provided columns info")
        if len(columns) != len(data.columns):
            raise ValueError("Column metadata doesn't match number of columns")
        for column_meta, name in zip(columns, column_names):
            if "name" in column_meta and column_meta["name"] != name:
                raise ValueError("Column names don't match")
            column_meta["name"] = name
    else:
        logger.info("Setting column names from header")
        columns = [{"name": name} for name in column_names]
        metadata["columns"] = columns

    if data.shape[0] == 0:
        logger.info("0 rows, returning early")
        metadata["types"] = []
        _drop_internal_metadata(metadata)
        return metadata

    # Get manual updates from the user
    manual_columns = {}
    if "manual_annotations" in metadata:
        if "columns" in metadata["manual_annotations"]:
            manual_columns = {
                col["name"]: col for col in metadata["manual_annotations"]["columns"]
            }

    # Cache some values that have been resolved for type identification but are
    # useful for spatial coverage computation: admin areas and addresses
    # Having to resolve them once to see if they're valid and a second time to
    # build coverage information would be too slow
    resolved_columns = {}

    # =========================================================================
    # STEP 2: Batch ML prediction for ALL columns (single forward pass!)
    # =========================================================================
    step_start = time.perf_counter()
    logger.info("[STEP 2/6] Geo classifier batch prediction...")

    geo_predictions = {}  # column_idx -> prediction dict

    if geo_classifier:
        # Collect samples from all columns (no manual override)
        # Use pre-extracted sample values from full dataset (before sampling)
        columns_for_batch = []
        for column_idx, column_meta in enumerate(columns):
            name = column_meta["name"]
            if name in manual_columns:
                continue  # Skip columns with manual annotations

            # Use pre-extracted sample values from full dataset
            if name in full_data_stats and "sample_values" in full_data_stats[name]:
                sample_values = full_data_stats[name]["sample_values"]
                if sample_values:
                    columns_for_batch.append((column_idx, name, sample_values))

        # BATCH PREDICTION - single forward pass for ALL columns!
        if columns_for_batch:
            batch_inputs = [(name, vals) for _, name, vals in columns_for_batch]
            batch_results = geo_classifier.predict_batch(
                batch_inputs, threshold=geo_classifier_threshold
            )

            for (column_idx, name, sample_values), prediction in zip(
                columns_for_batch, batch_results
            ):
                # Store prediction with sample values used
                geo_predictions[column_idx] = prediction
                geo_predictions[column_idx]["sample_values"] = sample_values

    step_times["2_geo_batch_predict"] = time.perf_counter() - step_start

    # Classifier diagnostics (the sampled values of each spatial column) only where asked for
    if diagnostics_path and geo_predictions:
        _save_classifier_diagnostics(diagnostics_path, columns, geo_predictions)

    logger.info(
        "[STEP 2/6] Geo batch prediction completed in %.3fs (%d columns)",
        step_times["2_geo_batch_predict"],
        len(geo_predictions),
    )

    # =========================================================================
    # STEP 3: Process columns (apply predictions + compute features)
    # =========================================================================
    step_start = time.perf_counter()
    logger.info("[STEP 3/6] Processing %d columns...", len(columns))

    for col_idx, column_meta in enumerate(columns):
        resolved = process_column(
            data[column_meta["name"]],
            column_meta,
            manual=manual_columns.get(column_meta["name"]),
            plots=plots,
            coverage=coverage,
            datamart_geo_data=datamart_geo_data,
            nominatim=nominatim,
            geo_prediction=geo_predictions.get(col_idx),
        )
        resolved_columns[col_idx] = resolved

        # Override with full-file stats computed before sampling.
        col_name = column_meta["name"]
        if col_name in full_data_stats:
            stats = full_data_stats[col_name]
            if "num_distinct_values" in column_meta:
                column_meta["num_distinct_values"] = stats["num_distinct_values"]
                if "num_distinct_values_is_approximate" in stats:
                    column_meta["num_distinct_values_is_approximate"] = bool(
                        stats["num_distinct_values_is_approximate"]
                    )
            if (
                column_meta["structural_type"] in (types.INTEGER, types.FLOAT)
                and stats.get("num_numeric_values")
            ):
                column_meta["min"] = stats["min"]
                column_meta["max"] = stats["max"]
                column_meta["mean"] = stats["mean"]
                column_meta["stddev"] = stats["stddev"]

    step_times["3_process_columns"] = time.perf_counter() - step_start
    logger.info(
        "[STEP 3/6] Column processing completed in %.3fs",
        step_times["3_process_columns"],
    )

    if full_statistics and plots:
        step_start = time.perf_counter()
        for column_meta in columns:
            if "plot" in column_meta:
                _histogram_over_all_rows(column_meta, csv_path, csv_sep, full_frame)
        step_times["3b_full_statistics"] = time.perf_counter() - step_start
        logger.info("Histograms over all rows computed in %.3fs", step_times["3b_full_statistics"])

    # =========================================================================
    # STEP 4: Pair lat/long columns and determine dataset types
    # =========================================================================
    step_start = time.perf_counter()
    logger.info("[STEP 4/6] Pairing lat/long columns and determining dataset types...")

    # Pair lat & long columns
    columns_lat = [
        LatLongColumn(
            index=col_idx,
            name=col["name"],
            annot_pair=manual_columns.get(col["name"], {}).get("latlong_pair"),
        )
        for col_idx, col in enumerate(columns)
        if types.LATITUDE in col["semantic_types"]
    ]
    columns_long = [
        LatLongColumn(
            index=col_idx,
            name=col["name"],
            annot_pair=manual_columns.get(col["name"], {}).get("latlong_pair"),
        )
        for col_idx, col in enumerate(columns)
        if types.LONGITUDE in col["semantic_types"]
    ]
    latlong_pairs, (missed_lat, missed_long) = pair_latlong_columns(
        columns_lat, columns_long
    )

    # Log missed columns
    if missed_lat:
        logger.warning("Unmatched latitude columns: %r", missed_lat)
    if missed_long:
        logger.warning("Unmatched longitude columns: %r", missed_long)

    # Remove semantic type from unpaired columns
    for col in columns:
        if col["name"] in missed_lat:
            col["semantic_types"].remove(types.LATITUDE)
        if col["name"] in missed_long:
            col["semantic_types"].remove(types.LONGITUDE)

    # Identify the overall dataset types (numerical, categorical, spatial, or temporal)
    dataset_types = collections.Counter()
    for column_meta in columns:
        dataset_type = determine_dataset_type(
            column_meta["structural_type"],
            column_meta["semantic_types"],
        )
        if dataset_type:
            dataset_types[dataset_type] += 1
    for key, d_type in [
        ("nb_spatial_columns", types.DATASET_SPATIAL),
        ("nb_temporal_columns", types.DATASET_TEMPORAL),
        ("nb_categorical_columns", types.DATASET_CATEGORICAL),
        ("nb_numerical_columns", types.DATASET_NUMERICAL),
    ]:
        if dataset_types[d_type]:
            metadata[key] = dataset_types[d_type]
    metadata["types"] = sorted(set(dataset_types))

    step_times["4_latlong_pairing"] = time.perf_counter() - step_start
    logger.info(
        "[STEP 4/6] Lat/long pairing completed in %.3fs (%d pairs found)",
        step_times["4_latlong_pairing"],
        len(latlong_pairs),
    )

    # =========================================================================
    # STEP 5: Compute spatial coverage
    # =========================================================================
    step_start = time.perf_counter()

    if coverage:
        logger.info("[STEP 5/6] Computing spatial coverage...")
        spatial_coverage = []
        with NoOpContext():
            # Compute sketches from lat/long pairs
            for col_lat, col_long in latlong_pairs:
                if csv_path is not None:
                    cov = _csv_latlong_coverage(csv_path, csv_sep, col_lat, col_long, spatial_leaves)
                    if cov is not None:
                        spatial_coverage.append(cov)
                    continue

                lat_values = data.iloc[:, col_lat.index]
                lat_values = pandas.to_numeric(lat_values, errors="coerce")
                long_values = data.iloc[:, col_long.index]
                long_values = pandas.to_numeric(long_values, errors="coerce")
                mask = (
                    ~numpy.isnan(lat_values)
                    & ~numpy.isnan(long_values)
                    & (-90.0 < lat_values)
                    & (lat_values < 90.0)
                    & (-180.0 < long_values)
                    & (long_values < 180.0)
                )
                # (0, 0) is a common placeholder for missing coordinates
                zero_mask = mask & (lat_values == 0.0) & (long_values == 0.0)
                zeros = int(zero_mask.sum())
                mask = mask & ~zero_mask

                if mask.any():
                    lat_values = lat_values[mask]
                    long_values = long_values[mask]
                    values = numpy.array([lat_values, long_values]).T
                    logger.info(
                        "Computing spatial sketch lat=%r long=%r (%d rows)",
                        col_lat.name,
                        col_long.name,
                        len(values),
                    )
                    # Ranges
                    spatial_ranges = get_spatial_ranges(values)
                    # Geohashes, from the points inside the 0.1-99.9% envelope (see _csv_latlong_coverage)
                    q_lo, q_hi = ENVELOPE_QUANTILES
                    lat_lo, lat_hi = numpy.quantile(values[:, 0], (q_lo, q_hi))
                    long_lo, long_hi = numpy.quantile(values[:, 1], (q_lo, q_hi))
                    in_box = (
                        (values[:, 0] >= lat_lo) & (values[:, 0] <= lat_hi)
                        & (values[:, 1] >= long_lo) & (values[:, 1] <= long_hi)
                    )
                    inner = values[in_box]
                    hashes = geohash_counts(inner[:, 0], inner[:, 1], MAX_GEOHASHES)
                    leaves = kdtree_leaves(
                        inner[:, 1], inner[:, 0], (long_lo, lat_lo, long_hi, lat_hi), spatial_leaves
                    )
                    grid = density_grid(inner[:, 1], inner[:, 0], dense_area(leaves)) if leaves else None
                    outside_envelope = int(len(values) - in_box.sum())

                    spatial_coverage.append(
                        {
                            "type": "latlong",
                            "column_names": [col_lat.name, col_long.name],
                            "column_indexes": [
                                col_lat.index,
                                col_long.index,
                            ],
                            "geohashes4": hashes,
                            "kdtree": _kdtree_json(leaves, spatial_leaves),
                            **({"grid": grid} if grid else {}),
                            "ranges": spatial_ranges,
                            "number": len(values),
                            "number_outside_envelope": outside_envelope,
                            **({"number_zero_coordinates": zeros} if zeros else {}),
                        }
                    )

            # Compute sketches from WKT points
            for i, col in enumerate(columns):
                if col["structural_type"] != types.GEO_POINT:
                    continue
                latlong = col.get("point_format") == "lat,long"
                name = col["name"]
                values = parse_wkt_column(
                    data.iloc[:, i],
                    latlong=latlong,
                )
                total = numpy.sum(data.iloc[:, i].apply(lambda x: bool(x)))
                if len(values) < 0.5 * total:
                    logger.warning(
                        "Most data points did not parse correctly as "
                        "point (%s) col=%d %r",
                        "lat,long" if latlong else "long,lat",
                        i,
                        col,
                    )
                if values:
                    logger.info(
                        "Computing spatial sketches point=%r (%d rows)",
                        name,
                        len(values),
                    )
                    # Ranges
                    spatial_ranges = get_spatial_ranges(values)
                    # Geohashes
                    builder = Geohasher(number=MAX_GEOHASHES)
                    builder.add_points(values)
                    hashes = builder.get_hashes_json()

                    spatial_coverage.append(
                        {
                            "type": "point_latlong" if latlong else "point",
                            "column_names": [name],
                            "column_indexes": [i],
                            "geohashes4": hashes,
                            "ranges": spatial_ranges,
                            "number": len(values),
                        }
                    )

            for idx, resolved in resolved_columns.items():
                # Compute sketches from addresses
                if "addresses" in resolved:
                    locations = resolved["addresses"]

                    name = columns[idx]["name"]
                    logger.info(
                        "Computing spatial sketches address=%r (%d rows)",
                        name,
                        len(locations),
                    )
                    # Ranges
                    spatial_ranges = get_spatial_ranges(locations)
                    # Geohashes
                    builder = Geohasher(number=MAX_GEOHASHES)
                    builder.add_points(locations)
                    hashes = builder.get_hashes_json()

                    spatial_coverage.append(
                        {
                            "type": "address",
                            "column_names": [name],
                            "column_indexes": [idx],
                            "geohashes4": hashes,
                            "ranges": spatial_ranges,
                            "number": len(locations),
                        }
                    )

                # Compute sketches from administrative areas
                if "admin_areas" in resolved:
                    areas = resolved["admin_areas"]

                    name = columns[idx]["name"]
                    logger.info(
                        "Computing spatial sketches admin_areas=%r (%d rows)",
                        name,
                        len(areas),
                    )
                    cov = {
                        "type": "admin",
                        "column_names": [name],
                        "column_indexes": [idx],
                    }

                    # Merge into a single range
                    merged = None
                    for area in areas:
                        if area is None:
                            continue
                        new = area.bounds
                        if new:
                            if merged is None:
                                merged = new
                            else:
                                merged = (
                                    min(merged[0], new[0]),
                                    max(merged[1], new[1]),
                                    min(merged[2], new[2]),
                                    max(merged[3], new[3]),
                                )
                    if (
                        merged is not None
                        and merged[1] - merged[0] > 0.01
                        and merged[3] - merged[2] > 0.01
                    ):
                        logger.info("Computed bounding box")
                        cov["ranges"] = [
                            {
                                "range": {
                                    "type": "envelope",
                                    "coordinates": [
                                        [merged[0], merged[3]],
                                        [merged[1], merged[2]],
                                    ],
                                },
                            },
                        ]
                    else:
                        logger.info("Couldn't build a bounding box")

                    # Compute geohashes
                    builder = Geohasher(number=MAX_GEOHASHES)
                    for area in areas:
                        if area is None or not area.bounds:
                            continue
                        builder.add_aab(area.bounds)
                    hashes = builder.get_hashes_json()
                    if hashes:
                        cov["geohashes4"] = hashes

                    # Count
                    cov["number"] = builder.total

                    if "ranges" in cov or "geohashes4" in cov:
                        spatial_coverage.append(cov)

        if spatial_coverage:
            metadata["spatial_coverage"] = spatial_coverage

        step_times["5_spatial_coverage"] = time.perf_counter() - step_start
        logger.info(
            "[STEP 5/6] Spatial coverage completed in %.3fs (%d coverage entries)",
            step_times["5_spatial_coverage"],
            len(spatial_coverage),
        )

        # =====================================================================
        # STEP 6: Compute temporal coverage
        # =====================================================================
        step_start = time.perf_counter()
        logger.info("[STEP 6/6] Computing temporal coverage...")
        temporal_coverage = []

        with NoOpContext():
            # Datetime columns
            for idx, col in enumerate(columns):
                if types.DATE_TIME not in col["semantic_types"]:
                    continue
                datetimes = resolved_columns[idx].get("datetimes")
                timestamps = resolved_columns[idx].get("timestamps")
                if not datetimes:  # e.g. annotated as a date, but no value parses
                    continue
                logger.info(
                    "Computing temporal ranges datetime=%r (%d rows)",
                    col["name"],
                    len(datetimes),
                )

                full_minmax = (
                    _csv_temporal_minmax(csv_path, csv_sep, col["name"])
                    if csv_path is not None
                    else None
                )
                if full_minmax is None:
                    ranges = get_numerical_ranges(timestamps)
                else:
                    min_dt, max_dt = full_minmax
                    ranges = [
                        {
                            "range": {
                                "gte": min_dt.timestamp(),
                                "lte": max_dt.timestamp(),
                            }
                        }
                    ]
                if not ranges:
                    continue

                # Get temporal resolution
                resolution = get_temporal_resolution(datetimes)

                entry = {
                    "type": "datetime",
                    "column_names": [col["name"]],
                    "column_indexes": [idx],
                    "column_types": [types.DATE_TIME],
                    "ranges": ranges,
                    "temporal_resolution": resolution,
                    "temporal_precision": temporal_precision(timestamps),
                }
                # The spacing of timestamps is only meaningful on all of them: a sample of rows spaces
                # them further apart than the data does.
                if csv_path is not None:
                    all_timestamps = _csv_temporal_timestamps(csv_path, csv_sep, col["name"])
                elif metadata.get("nb_profiled_rows") == metadata.get("nb_rows"):
                    all_timestamps = timestamps
                else:
                    all_timestamps = None
                if all_timestamps is not None:
                    sampling = temporal_sampling(all_timestamps)
                    if sampling is not None:
                        entry["sampling"] = sampling
                temporal_coverage.append(entry)

            # TODO: Times split over multiple columns

        if temporal_coverage:
            metadata["temporal_coverage"] = temporal_coverage

        step_times["6_temporal_coverage"] = time.perf_counter() - step_start
        logger.info(
            "[STEP 6/6] Temporal coverage completed in %.3fs (%d coverage entries)",
            step_times["6_temporal_coverage"],
            len(temporal_coverage),
        )
    else:
        step_times["5_spatial_coverage"] = 0
        step_times["6_temporal_coverage"] = 0
        logger.info("[STEP 5-6/6] Coverage computation skipped (coverage=False)")

    # Attribute names
    # Names are stripped (CSV headers often carry a space after each comma) and keywords deduplicated.
    attribute_keywords = []
    for col in columns:
        name = str(col["name"]).strip()
        for kw in (name, *expand_attribute_name(name)):
            if kw and kw not in attribute_keywords:
                attribute_keywords.append(kw)
    metadata["attribute_keywords"] = attribute_keywords

    # Sample data
    if include_sample:
        step_start = time.perf_counter()
        rand = numpy.random.RandomState(RANDOM_SEED)
        choose_rows = rand.choice(
            len(data),
            min(SAMPLE_ROWS, len(data)),
            replace=False,
        )
        choose_rows.sort()  # Keep it in order
        sample = data.iloc[choose_rows]
        sample = sample.map(truncate_string)  # Truncate long values
        metadata["sample"] = sample.to_csv(index=False, lineterminator="\r\n")
        step_times["sample_extraction"] = time.perf_counter() - step_start

    # =========================================================================
    # PIPELINE SUMMARY
    # =========================================================================
    total_time = time.perf_counter() - pipeline_start

    # Build summary string
    summary_parts = [f"\n{'='*60}"]
    summary_parts.append("PROFILING PIPELINE SUMMARY")
    summary_parts.append(f"{'='*60}")
    summary_parts.append(f"Dataset: {data.shape[0]} rows × {data.shape[1]} columns")
    summary_parts.append(f"{'-'*60}")

    for step_name, step_time in step_times.items():
        pct = (step_time / total_time * 100) if total_time > 0 else 0
        bar_len = int(pct / 2)  # 50 chars = 100%
        bar = "█" * bar_len + "░" * (50 - bar_len)
        summary_parts.append(f"{step_name:25s} {step_time:7.3f}s ({pct:5.1f}%) {bar}")

    summary_parts.append(f"{'-'*60}")
    summary_parts.append(f"{'TOTAL':25s} {total_time:7.3f}s")
    summary_parts.append(f"{'='*60}\n")

    logger.info("\n".join(summary_parts))

    # Also store timing in metadata for analysis
    metadata["_profiling_times"] = {
        "steps": step_times,
        "total": total_time,
    }

    metadata["computed_on"] = _computed_on(metadata, load_max_size)
    _drop_internal_metadata(metadata)
    return metadata


def _string_labels(frame):
    """The DataFrame with string column labels, duplicates renamed ``name``, ``name.1``, ... (as
    ``pandas.read_csv`` does). The profile refers to columns by these names."""
    labels, seen = [], {}
    for label in (str(c) for c in frame.columns):
        name, k = label, seen.get(label, 0)
        while name in seen:
            k += 1
            name = f"{label}.{k}"
        seen[label] = k
        seen[name] = 0
        labels.append(name)
    if labels == list(frame.columns):
        return frame
    if len(set(labels)) != len(labels):  # pragma: no cover - guarded by the loop above
        raise ValueError("Could not make column labels unique")
    renamed = [f"{a!r} -> {b!r}" for a, b in zip(frame.columns, labels) if a != b]
    logger.warning("Column labels converted to unique strings: %s", ", ".join(renamed[:10]))
    return frame.set_axis(labels, axis=1)


def _save_classifier_diagnostics(path, columns, geo_predictions):
    """Append the spatial classifier's predictions, with the sampled values it saw, to a CSV file."""
    rows = []
    for col_idx, prediction in geo_predictions.items():
        label = prediction.get("label", "non_spatial")
        if label == "non_spatial":
            continue
        rows.append({
            "name": columns[col_idx]["name"],
            "values": prediction.get("sample_values"),
            "label": label,
            "confidence": prediction.get("confidence"),
            "source": prediction.get("source"),
        })
    if not rows:
        return
    path = os.fspath(path)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    exists = os.path.exists(path)
    pandas.DataFrame(rows).to_csv(path, index=False, mode="a" if exists else "w", header=not exists)
    logger.info("Saved %d geo classifier results to %s", len(rows), path)


def _all_numbers(name, csv_path, csv_sep, frame):
    """Every numeric value of a column (as floats), or ``None`` when the column is not available."""
    if csv_path is not None:
        import polars as pl

        value = pl.col(name).str.strip_chars()
        number = pl.when(value == "").then(None).otherwise(value).cast(pl.Float64, strict=False)
        values = _collect_streaming(_scan_csv(pl, csv_path, csv_sep).select(number.alias("v")).drop_nulls())
        values = values["v"].to_numpy()
    elif frame is not None and name in frame.columns:
        values = pandas.to_numeric(frame[name], errors="coerce").dropna().to_numpy(dtype="float64")
    else:
        return None
    return values[numpy.abs(values) < 3.4e38]  # as the sample histogram does


def _all_timestamps(name, csv_path, csv_sep, frame):
    """POSIX seconds of every value of a date column, or ``None`` when they cannot be parsed safely.

    Parsing in bulk is only used when it reads every value and agrees with ``parse_date`` on the first
    values (see ``_csv_temporal_minmax_polars``); otherwise the histogram stays on the sample.
    """
    if csv_path is not None:
        import polars as pl

        if _csv_temporal_minmax_polars(pl, csv_path, csv_sep, name) is None:
            return None
        value = pl.col(name).str.strip_chars()
        parsed = pl.when(value == "").then(None).otherwise(value).str.to_datetime(strict=False, time_unit="us")
        frame_t = _collect_streaming(
            _scan_csv(pl, csv_path, csv_sep).select(parsed.dt.epoch("us").alias("t")).drop_nulls()
        )
        return frame_t["t"].to_numpy().astype("float64") / 1e6
    if frame is None or name not in frame.columns:
        return None
    text = frame[name].astype(str).str.strip()
    text = text[(text != "") & (text != "nan") & (text != "None")]
    if not len(text):
        return None
    with ignore_warnings(UserWarning):
        parsed = pandas.to_datetime(text, errors="coerce", utc=True)
    if parsed.isna().any():
        return None
    for raw, fast in zip(text.iloc[:TEMPORAL_CHECK_VALUES], parsed.iloc[:TEMPORAL_CHECK_VALUES]):
        if parse_date(raw) != fast.to_pydatetime():
            return None
    return parsed.astype("int64").to_numpy() / 1e9


def _all_top_values(name, csv_path, csv_sep, frame, top=5):
    """The ``top`` most frequent non-empty values of a column over every row, as a categorical plot."""
    if csv_path is not None:
        import polars as pl

        value = pl.col(name).str.strip_chars()
        counts = _collect_streaming(
            _scan_csv(pl, csv_path, csv_sep).select(value.alias("v")).filter(pl.col("v") != "")
            .group_by("v").len()
        )
        pairs = list(zip(counts["v"].to_list(), counts["len"].to_list()))
    elif frame is not None and name in frame.columns:
        text = frame[name].astype(str).str.strip()
        text = text[(text != "") & (text != "nan") & (text != "None")]
        pairs = list(text.value_counts().items())
    else:
        return None
    pairs = sorted(sorted(pairs, key=lambda p: -p[1])[:top])  # as the sample histogram: top values, by value
    return {
        "type": "histogram_categorical",
        "data": [{"bin": value, "count": int(count)} for value, count in pairs],
    }


def _histogram_over_all_rows(column_meta, csv_path, csv_sep, frame):
    """Replace a column's sample histogram by one over every row, where that can be done safely."""
    name = column_meta["name"]
    kind = column_meta["plot"].get("type")
    histogram = None
    if kind == "histogram_numerical":
        values = _all_numbers(name, csv_path, csv_sep, frame)
        histogram = numerical_histogram(values) if values is not None else None
    elif kind == "histogram_temporal":
        timestamps = _all_timestamps(name, csv_path, csv_sep, frame)
        histogram = temporal_histogram(timestamps) if timestamps is not None else None
    elif kind == "histogram_categorical":
        histogram = _all_top_values(name, csv_path, csv_sep, frame)
    if histogram is not None:
        histogram["computed_on"] = "all_rows"
        column_meta["plot"] = histogram
    else:
        column_meta["plot"]["computed_on"] = "sample"


def _computed_on(metadata, load_max_size=None):
    """Which rows each part of the profile comes from, and how the profiled rows were chosen.

    Every value is ``all_rows`` or ``sample`` (``mixed`` when parts of one kind differ). Column types and
    value ranges always come from the profiled rows; histograms too, unless ``full_statistics``
    recomputed them over every row. Distinct counts come from every row (approximate for large CSV
    columns, see ``num_distinct_values_is_approximate``). A CSV/TSV path also gets numeric statistics,
    lat/long coverage, and the extent and spacing of dates from every row; WKT points, addresses and
    administrative areas, and the resolution and precision of dates, come from the profiled rows.

    Each spatial and temporal coverage entry also gets its own ``computed_on``.
    """
    sampled = metadata.get("nb_profiled_rows", 0) < metadata.get("nb_rows", 0)
    profiled = "sample" if sampled else "all_rows"
    from_file = "all_rows" if metadata.get("_csv_path") else profiled

    def combine(values, default):
        values = set(values)
        if not values:
            return default
        return values.pop() if len(values) == 1 else "mixed"

    histograms = profiled
    sources = {c["plot"]["computed_on"] for c in metadata.get("columns", []) if "computed_on" in c.get("plot", {})}
    if sampled and sources:  # full_statistics: each histogram says where it comes from
        histograms = combine(sources, profiled)

    for entry in metadata.get("spatial_coverage", []):
        entry["computed_on"] = from_file if entry.get("type") == "latlong" else profiled
    for entry in metadata.get("temporal_coverage", []):
        entry["computed_on"] = {
            "ranges": from_file,
            **({"sampling": "all_rows"} if "sampling" in entry else {}),  # only ever over every row
            "temporal_resolution": profiled,
            "temporal_precision": profiled,
        }

    return {
        "sampling": {
            "method": "random" if sampled else "none",
            "seed": RANDOM_SEED if sampled else None,
            "size_rule": "load_max_size" if load_max_size is not None else "adaptive",
            "rows": metadata.get("nb_profiled_rows", 0),
            "of": metadata.get("nb_rows", 0),
        },
        "column_types": profiled,
        "value_ranges": profiled,
        "histograms": histograms,
        "distinct_counts": "all_rows",
        "numeric_statistics": from_file,
        "spatial_coverage": combine((e["computed_on"] for e in metadata.get("spatial_coverage", [])), from_file),
        "temporal_coverage": from_file,
        "temporal_resolution": profiled,
    }
