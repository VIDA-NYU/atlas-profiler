"""A report of a profile (the metadata ``process_dataset`` returns, or its saved JSON), drawn with Autark.

Everything comes from the profile itself; the data is not read again. The page is organized as a report:
an overview with warnings, a table of the columns, the distribution of each column, maps of the spatial
coverage, the temporal coverage and the profiling time. The charts and maps are built in Python with
pyautark (``import autark``) and drawn in the browser by Autark; the rest is plain HTML.

Usage::

    from atlas_profiler import ProfileReport

    ProfileReport.from_json("profile.json").save("report.html")

or from the command line::

    python -m profiler.report profile.json -o report.html

Histograms come from the profile's ``plot`` entries, so profile with ``plots=True`` to get them. The maps
need a browser with WebGPU. pyautark is not on PyPI yet; install it from its repository.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

__all__ = ["ProfileReport", "render_html", "save_html", "decode_geohash"]

# Esri's light-gray canvas: a quiet basemap that does not compete with the data.
_BASEMAP = {
    "urlTemplate": "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
    "attribution": "Esri, HERE, Garmin, © OpenStreetMap contributors",
    "maxZoom": 16,
}
_HIST_COLUMNS = 3          # histograms per row
_HIST_WIDTH = 360          # pixels
_HIST_HEIGHT = 210
_MAP_HEIGHT = 380


def _autark():
    try:
        import autark
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "The profile report draws with Autark through pyautark (`import autark`), which is not "
            "installed. Install it from https://github.com/urban-toolkit/autark (python/ folder)."
        ) from exc
    return autark


# ---------------------------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------------------------

def _esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _short_type(uri: str) -> str:
    """``http://schema.org/Float`` -> ``Float``."""
    return str(uri).rstrip("/").rsplit("/", 1)[-1]


def _fmt_num(value: Any) -> str:
    if value is None:
        return "–"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return f"{value:,}"
    try:
        x = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(x) or math.isinf(x):
        return str(x)
    if x == int(x) and abs(x) < 1e15:
        return f"{int(x):,}"
    if abs(x) >= 1e4:
        return f"{x:,.0f}"
    return f"{x:.4g}"


def _fmt_bytes(n: Any) -> str:
    try:
        n = float(n)
    except (TypeError, ValueError):
        return "–"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return "–"


def _fmt_seconds(s: float) -> str:
    if s < 0.001:
        return "<1 ms"
    if s < 1:
        return f"{s * 1000:.0f} ms"
    if s < 60:
        return f"{s:.1f} s"
    return f"{int(s // 60)} min {s % 60:.0f} s"


def _fmt_ts(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def _fmt_span(seconds: float) -> str:
    days = seconds / 86400
    if days >= 2:
        return f"{days:,.1f} days"
    return f"{seconds / 3600:,.1f} hours"


def _display_name(name: Any) -> str:
    """Column name for HTML, quoted when it has leading or trailing whitespace so that is visible."""
    name = str(name)
    if name != name.strip():
        return f'<span class="ws" title="Name has leading or trailing whitespace">"{_esc(name)}"</span>'
    return _esc(name)


def _anchor(name: Any) -> str:
    return "col-" + "".join(ch if ch.isalnum() else "-" for ch in str(name).strip().lower())


def _ranges(coverage: Iterable[dict] | None) -> list[tuple[float, float]]:
    out = []
    for entry in coverage or []:
        rng = entry.get("range", {}) if isinstance(entry, dict) else {}
        if "gte" in rng and "lte" in rng:
            out.append((float(rng["gte"]), float(rng["lte"])))
    return out


def _slug(text: str, used: set[str]) -> str:
    """A table name for Autark: letters, digits and underscores, unique within the report."""
    base = "".join(ch if ch.isalnum() else "_" for ch in str(text).strip().lower()).strip("_") or "t"
    if base[0].isdigit():
        base = f"t_{base}"
    name, i = base, 2
    while name in used:
        name, i = f"{base}_{i}", i + 1
    used.add(name)
    return name


# ---------------------------------------------------------------------------------------------
# Geohash decoding (base-4 quadtree hashes used for ``geohashes4``)
# ---------------------------------------------------------------------------------------------

_GEOHASH_CHARS = "0123456789bcdefghjkmnpqrstuvwxyz"
_GEOHASH_VALUES = {c: i for i, c in enumerate(_GEOHASH_CHARS)}


def decode_geohash(hash_: str, base: int = 4) -> tuple[float, float, float, float]:
    """Turn a hash into its cell, as ``(min_lat, max_lat, min_lon, max_lon)``.

    Same scheme as ``profiler.spatial.decode_hash``, reimplemented so the report does not import the
    spatial module (and with it PyTorch).
    """
    base_bits = base.bit_length() - 1
    if 2 ** base_bits != base:
        raise ValueError("Base is not a power of 2")
    lat = [-90.0, 90.0]
    lon = [-180.0, 180.0]
    next_lon = True
    for char in hash_:
        value = _GEOHASH_VALUES[char]
        for i in reversed(range(base_bits)):
            bit = (value >> i) & 1
            rng = lon if next_lon else lat
            mid = (rng[0] + rng[1]) / 2.0
            if bit:
                rng[0] = mid
            else:
                rng[1] = mid
            next_lon = not next_lon
    return lat[0], lat[1], lon[0], lon[1]


def _envelope(entry: dict) -> tuple[float, float, float, float] | None:
    """Bounding box of a spatial coverage entry as ``(min_lon, max_lon, min_lat, max_lat)``."""
    lons, lats = [], []
    for rng in entry.get("ranges", []):
        coords = (rng.get("range") or {}).get("coordinates") or []
        for pt in coords:
            if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                lons.append(float(pt[0]))
                lats.append(float(pt[1]))
    if not lons:
        return None
    return min(lons), max(lons), min(lats), max(lats)


def _coverage_cells(entry: dict, share: float = 0.95, margin: float = 0.25):
    """Geohash cells as GeoJSON polygons around where most of the points are, and that area's box.

    The frame is the bounding box of the densest cells (points per area) that together hold ``share``
    of the points; cells overlapping the frame grown by ``margin`` on each side are kept for context.
    Far-away cells (bad geocodes, (0, 0) placeholders) are left out: Autark centres the map on its data,
    so they would pull the view away from the data. Returns the features, the number of cells and points
    left out, and the frame as ``(min_lon, max_lon, min_lat, max_lat)``.
    """
    if entry.get("grid"):
        return _grid_features(entry["grid"])
    cells = _cells(entry)
    if not cells:
        return [], 0, 0, None

    total = sum(c[4] for c in cells) or 1
    dense, acc = [], 0
    for c in sorted(cells, key=_density, reverse=True):
        dense.append(c)
        acc += c[4]
        if acc >= share * total:
            break
    x0, x1 = min(c[0] for c in dense), max(c[1] for c in dense)
    y0, y1 = min(c[2] for c in dense), max(c[3] for c in dense)
    frame = (x0, x1, y0, y1)
    mx, my = (x1 - x0) * margin, (y1 - y0) * margin
    kept = [c for c in cells if c[1] > x0 - mx and c[0] < x1 + mx and c[3] > y0 - my and c[2] < y1 + my]
    features = [
        {
            "type": "Feature",
            "properties": {
                "cell": cid, "points": n,
                "log_density": round(math.log10(max(_density(c), 1e-9)), 4),
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[lon0, lat0], [lon1, lat0], [lon1, lat1], [lon0, lat1], [lon0, lat0]]],
            },
        }
        for c in kept
        for lon0, lon1, lat0, lat1, n, cid in [c]
    ]
    return features, len(cells) - len(kept), total - sum(c[4] for c in kept), frame


def _grid_features(grid: dict):
    """The profile's density grid as GeoJSON squares, with exact counts; framed on the grid's area."""
    x0, y0, x1, y1 = grid["bounds"]
    dx, dy = grid["cell_degrees"]
    area = float(grid["cell_km"]) ** 2
    features = []
    for col, row, count in grid["cells"]:
        gx, gy = x0 + col * dx, y0 + row * dy
        features.append({
            "type": "Feature",
            "properties": {"cell": f"grid {col},{row}", "points": int(count),
                           "log_density": round(math.log10(count / area), 4)},
            "geometry": {"type": "Polygon", "coordinates": [[
                [gx, gy], [gx + dx, gy], [gx + dx, gy + dy], [gx, gy + dy], [gx, gy],
            ]]},
        })
    return features, 0, int(grid.get("outside", 0)), (x0, x1, y0, y1)


def _cells(entry: dict) -> list[tuple]:
    """Cells of a spatial coverage entry as ``(min_lon, max_lon, min_lat, max_lat, points, id)``.

    The leaves of the profile's k-d tree when it has one (they follow the density of the data), else
    the base-4 geohash cells of older profiles.
    """
    leaves = (entry.get("kdtree") or {}).get("leaves")
    if leaves:
        return [
            (float(b[0]), float(b[2]), float(b[1]), float(b[3]), int(leaf.get("number", 0)), f"leaf {i}")
            for i, leaf in enumerate(leaves)
            for b in [leaf["bounds"]]
        ]
    cells = []
    for c in entry.get("geohashes4") or []:
        try:
            lat0, lat1, lon0, lon1 = decode_geohash(c["hash"], base=4)
        except (KeyError, ValueError):
            continue
        cells.append((lon0, lon1, lat0, lat1, int(c.get("number", 0)), c["hash"]))
    return cells


def _density(c: tuple) -> float:
    """Points per square kilometre of a cell."""
    mid_lat = math.radians((c[2] + c[3]) / 2)
    km2 = (c[1] - c[0]) * 111.32 * max(math.cos(mid_lat), 0.01) * (c[3] - c[2]) * 110.57
    return c[4] / max(km2, 1e-6)


def _fit_zoom(frame: tuple[float, float, float, float], width_px: int, height_px: int) -> float:
    """The autk-grammar ``Camera.zoom`` that shows ``frame`` (min_lon, max_lon, min_lat, max_lat).

    autk-grammar puts the camera ``10000 / 2 ** (zoom - 10)`` metres above the centre of the map's data,
    with a 45-degree vertical field of view; Autark draws in Mercator metres.
    """
    min_lon, max_lon, min_lat, max_lat = frame
    r = 6378137.0
    merc_y = lambda lat: r * math.log(math.tan(math.pi / 4 + math.radians(max(min(lat, 85.0), -85.0)) / 2))  # noqa: E731
    span_x = r * math.radians(max_lon - min_lon)
    span_y = merc_y(max_lat) - merc_y(min_lat)
    visible_per_metre = 2 * math.tan(math.radians(22.5))  # visible height per metre of camera distance
    distance = 1.15 * max(span_y, span_x * height_px / width_px) / visible_per_metre
    return round(10 - math.log2(max(distance, 1.0) / 10000), 3)


# ---------------------------------------------------------------------------------------------
# Histograms from the profile's ``plot`` entries
# ---------------------------------------------------------------------------------------------

def _temporal_label(iso: str, unit: str | None, span_days: float) -> str:
    dt = datetime.fromisoformat(iso)
    if unit == "hour" or (unit is None and span_days < 2):
        return dt.strftime("%m-%d %H:00")
    if unit == "day" or (unit is None and span_days < 366):
        return dt.strftime("%m-%d")
    if unit == "month":
        return dt.strftime("%Y-%m")
    if unit == "year":
        return dt.strftime("%Y")
    return dt.strftime("%Y-%m")


MAX_BARS = 40       # a 360-pixel chart can label about this many bars
MAX_LABEL_CHARS = 14


def _merge_bins(plot: dict, max_bars: int = MAX_BARS) -> tuple[dict, int]:
    """Merge consecutive bins of a numeric or time histogram so it has at most ``max_bars`` bars.

    Bins are contiguous, so merged counts stay exact. Returns the plot and how many bins each bar holds.
    """
    data = plot.get("data") or []
    if plot.get("type") not in ("histogram_numerical", "histogram_temporal") or plot.get("discrete") or len(data) <= max_bars:
        return plot, 1
    k = math.ceil(len(data) / max_bars)
    start_key, end_key = ("bin_start", "bin_end") if plot["type"] == "histogram_numerical" else ("date_start", "date_end")
    merged = [
        {start_key: group[0][start_key], end_key: group[-1][end_key], "count": sum(int(d.get("count", 0)) for d in group)}
        for group in (data[i:i + k] for i in range(0, len(data), k))
    ]
    return {**plot, "data": merged}, k


def _short(label: str, limit: int = MAX_LABEL_CHARS) -> str:
    return label if len(label) <= limit else label[: limit - 1].rstrip() + "…"


def _bin_labels(plot: dict) -> list[str]:
    kind = plot.get("type", "")
    data = plot.get("data") or []
    if kind == "histogram_numerical":
        if plot.get("discrete"):  # one bar per value: the bin is centred on it
            labels = [_fmt_num((d["bin_start"] + d["bin_end"]) / 2) for d in data]
        else:
            labels = [f"{_fmt_num(d.get('bin_start'))}–{_fmt_num(d.get('bin_end'))}" for d in data]
    elif kind == "histogram_temporal":
        try:
            span = datetime.fromisoformat(str(data[-1]["date_end"])) - datetime.fromisoformat(str(data[0]["date_start"]))
            labels = [_temporal_label(str(d["date_start"]), plot.get("unit"), span.days) for d in data]
        except (ValueError, KeyError, IndexError):
            labels = [str(d.get("date_start", ""))[:16] for d in data]
    else:
        labels = [_short(str(d.get("bin", "")).strip()) for d in data]
    # The chart needs one distinct label per bar; number duplicates (from rounding) to keep them apart.
    seen: dict[str, int] = {}
    out = []
    for label in labels:
        seen[label] = seen.get(label, 0) + 1
        out.append(label if seen[label] == 1 else f"{label} ({seen[label]})")
    return out


def _histogram(column: dict) -> tuple[list[dict], str] | None:
    """Bars for a column's histogram and a caption, or ``None`` without one.

    Profiles from the robust histograms carry ``below``/``above`` counts of values outside the plotted
    range; they become a first and a last bar. Older profiles have 10 equal-width bins over min to max,
    and the caption says when a few extreme values flatten them.
    """
    plot = column.get("plot")
    if not isinstance(plot, dict) or not plot.get("data"):
        return None
    plot, merged = _merge_bins(plot)
    data = plot["data"]
    counts = [int(d.get("count", 0)) for d in data]
    labels = _bin_labels(plot)
    records = [{"bin": label, "count": n} for label, n in zip(labels, counts)]
    kind = plot.get("type", "")
    below, above = int(plot.get("below") or 0), int(plot.get("above") or 0)
    if below:
        first = data[0].get("bin_start", data[0].get("date_start"))
        edge = _fmt_num(first) if kind == "histogram_numerical" else str(first)[:10]
        records.insert(0, {"bin": f"< {edge}", "count": below})
    if above:
        last = data[-1].get("bin_end", data[-1].get("date_end"))
        edge = _fmt_num(last) if kind == "histogram_numerical" else str(last)[:10]
        records.append({"bin": f"> {edge}", "count": above})

    caption = ""
    total = sum(counts) or 1
    top = max(range(len(counts)), key=lambda i: counts[i])
    if "below" in plot:
        if below or above:
            caption = (f"The first and last bars count the values outside the plotted range "
                       f"({below:,} below, {above:,} above).")
    elif kind in ("histogram_numerical", "histogram_temporal") and counts[top] / total >= 0.9:
        caption = (
            f"{counts[top] / total:.0%} of the values fall in one bin ({labels[top]}): the 10 equal-width bins "
            f"span min {_fmt_num(column.get('min'))} to max {_fmt_num(column.get('max'))}, so a few extreme values flatten the rest."
        )
    if kind in ("histogram_categorical", "histogram_text"):
        caption = "Most frequent values" + (" (words)" if kind == "histogram_text" else "") + "."
        if any(len(str(d.get("bin", "")).strip()) > MAX_LABEL_CHARS for d in data):
            caption += f" Labels are cut to {MAX_LABEL_CHARS} characters."
    elif plot.get("discrete"):
        caption = "One bar per value."
    if kind == "histogram_temporal" and (plot.get("step") or 1) > 1:
        caption = (f"Bars of {plot['step']} {plot.get('unit', '')}s, labelled by their start. " + caption).strip()
    if merged > 1:
        caption = (f"{merged} bins per bar, merged for display. " + caption).strip()
    return records, caption


# ---------------------------------------------------------------------------------------------
# Autark views
# ---------------------------------------------------------------------------------------------

def _plot_source(ak: Any, name: str, records: list[dict]) -> Any:
    """An inline table for plots.

    Autark's plots read tables through its spatial database, which only takes tables with a geometry,
    so each record gets placeholder point coordinates (``_lat``/``_lon``). They are spread a little so
    the table's workspace has an extent, and no map draws them.
    """
    rows = [{**r, "_lat": 0.0, "_lon": i * 0.001} for i, r in enumerate(records)]
    return ak.JSON(name, values=rows, geometry=ak.latlng("_lat", "_lon"), workspace=name)


def _distributions_spec(metadata: dict) -> tuple[Any, list[tuple[str, str]], list[str]] | None:
    """One small bar chart per column with a histogram, in a grid; captions; columns without one."""
    ak = _autark()
    used: set[str] = set()
    data, views, captions, missing = [], [], [], []
    for c in metadata.get("columns", []):
        name = str(c.get("name", "")).strip()
        hist = _histogram(c)
        if hist is None:
            missing.append(name)
            continue
        records, caption = hist
        table = _slug(f"hist {name}", used)
        src = _plot_source(ak, table, records)
        data.append(src)
        views.append(ak.Barchart(
            src, x="bin", y="count", name=f"bar_{table}", title=name,
            width=_HIST_WIDTH, height=_HIST_HEIGHT, workspace=table,
            margins={"top": 28, "right": 28, "bottom": 70, "left": 48},
        ))
        if caption:
            captions.append((name, caption))
    if not views:
        return None
    spec = ak.Spec(data=data, views=views, layout=ak.Layout(type="grid", columns=_HIST_COLUMNS, gap=12))
    return spec, captions, missing


def _maps_spec(metadata: dict) -> tuple[Any, list[str]] | None:
    """Maps of the spatial coverage, side by side, all at the same initial zoom; one caption per map."""
    ak = _autark()
    used: set[str] = set()
    entries = [e for e in metadata.get("spatial_coverage") or [] if e.get("geohashes4") or e.get("grid")]
    if not entries:
        return None
    width = int(1100 / min(len(entries), 2))
    built = []
    for entry in entries:
        features, out_cells, out_points, frame = _coverage_cells(entry)
        if features:
            built.append((entry, features, out_cells, out_points, frame))
    if not built:
        return None
    # One zoom for every map, the one that shows the widest of their frames, so they compare at a glance.
    zoom = min(_fit_zoom(frame, width, _MAP_HEIGHT) for *_, frame in built)

    data, views, captions = [], [], []
    for entry, features, out_cells, out_points, frame in built:
        cols = " / ".join(str(n).strip() for n in entry.get("column_names", []))
        table = _slug(f"coverage {cols}", used)
        src = ak.GeoJSON(table, values={"type": "FeatureCollection", "features": features},
                         coordinate_format="EPSG:4326", workspace=table)
        data.append(src)
        views.append(ak.Map(
            name=f"map_{table}",
            layers=[ak.Layer(src, id=f"{cols} (log10 points/km²)").encode(color=ak.field("log_density", ak.Scale(scheme="blues"))).style(opacity=0.75)],
            basemap=ak.Basemap(_BASEMAP, title="Basemap"),
            camera=ak.Camera(zoom=zoom),
            workspace=table,
        ))
        total = sum(c[4] for c in _cells(entry)) or 1
        grid = entry.get("grid")
        if grid:
            caption = (f"<b>{_esc(cols)}</b>: points counted exactly on a grid of {grid['columns']} × {grid['rows']} "
                       f"cells of {1000 * grid['cell_km']:,.0f} m, over the area holding most of the points "
                       "(chosen with the profile's k-d tree). Darker is denser (legend: log10 points per km²).")
            if grid.get("outside"):
                caption += f" {grid['outside']:,} points fall outside the grid."
        else:
            caption = (f"<b>{_esc(cols)}</b>: {len(_cells(entry)):,} geohash cells. Darker is denser (legend: log10 "
                       "points per km²). The view is centred on the densest cells holding 95% of the points.")
        if out_cells and not grid:
            share = out_points / total
            caption += f" {out_cells} far cell(s) with {out_points:,} points ({'<0.1%' if 0 < share < 0.001 else f'{share:.1%}'}) are not shown."
        zeros = entry.get("number_zero_coordinates")
        if zeros:
            caption += f" {zeros:,} points at (0, 0), a placeholder for missing coordinates, are left out."
        outside = entry.get("number_outside_envelope")
        if outside:
            caption += f" {outside:,} points outside the 0.1–99.9% range of coordinates are not in the cells."
        captions.append(caption)
    spec = ak.Spec(data=data, views=views, layout=ak.Layout(type="horizontal" if len(views) > 1 else "vertical", gap=12))
    return spec, captions


# Autark sizes a map's drawing buffer when the map starts and afterwards only on a window resize. In
# pyautark's layout the map starts before its frame has its final size, so it would keep a buffer of the
# wrong shape (drawn squashed). Report a window resize whenever a canvas changes size.
_RESIZE_SCRIPT = """<script>
(() => {
  let pending = 0;
  const notify = () => { cancelAnimationFrame(pending); pending = requestAnimationFrame(() => window.dispatchEvent(new Event("resize"))); };
  const observer = new ResizeObserver(notify);
  const watch = () => document.querySelectorAll("canvas").forEach((c) => { if (!c.dataset.aprWatched) { c.dataset.aprWatched = "1"; observer.observe(c); notify(); } });
  new MutationObserver(watch).observe(document.body, { childList: true, subtree: true });
  watch();
})();
</script>"""


def _autark_frame(spec: Any, height_px: int, label: str, borderless_maps: bool = False) -> str:
    """An Autark view as a self-contained page in an iframe (pyautark's standalone page).

    ``borderless_maps`` turns off polygon borders on map layers (UrbanSpec ``stroke: false``); thousands
    of small cells would otherwise be drawn as solid outlines. pyautark's ``Layer.style`` cannot say
    this yet, so it is set on the UrbanSpec dict, which ``to_embedded_html`` also accepts.
    """
    ak = _autark()
    if borderless_maps:
        spec = spec.to_dict()
        maps = spec.get("map") or []
        for map_spec in [maps] if isinstance(maps, dict) else maps:  # one map is an object, several a list
            for layer in map_spec.get("layerRefs") or []:
                layer["stroke"] = False
    page = ak.to_embedded_html(spec, height=f"{height_px}px", title=label)
    page = page.replace("</body>", _RESIZE_SCRIPT + "</body>", 1)
    return (
        f'<iframe class="autark" title="{_esc(label)}" style="height:{height_px + 8}px" '
        f'srcdoc="{html.escape(page, quote=True)}"></iframe>'
    )


# ---------------------------------------------------------------------------------------------
# Report sections (HTML)
# ---------------------------------------------------------------------------------------------

def _warnings(meta: dict) -> list[str]:
    notes = []
    rows, profiled = meta.get("nb_rows"), meta.get("nb_profiled_rows")
    if rows and profiled and profiled < rows:
        computed = meta.get("computed_on")
        if isinstance(computed, dict):
            sample = [k.replace("_", " ") for k, v in computed.items() if v == "sample"]
            every = [k.replace("_", " ") for k, v in computed.items() if v == "all_rows"]
            notes.append(
                f"Profiled on a random sample of {profiled:,} of {rows:,} rows: {', '.join(sample)} come from the sample"
                + (f"; {', '.join(every)} from every row." if every else ".")
            )
        else:
            notes.append(
                f"Column types, statistics and histograms come from a sample of {profiled:,} of {rows:,} rows. "
                "Spatial and temporal coverage scan every row when a CSV/TSV path is profiled, "
                "and only the sample otherwise."
            )
    spaced = [c["name"] for c in meta.get("columns", []) if str(c.get("name", "")) != str(c.get("name", "")).strip()]
    if spaced:
        notes.append(
            f"{len(spaced)} column name(s) have leading or trailing whitespace, for example "
            f"{_display_name(spaced[0])}; joins and lookups by name may miss them."
        )
    for c in meta.get("columns", []):
        ratio = c.get("missing_values_ratio")
        if ratio is not None and ratio >= 0.5:
            notes.append(f"Column {_display_name(c['name'])} is {ratio:.1%} missing.")
    for entry in meta.get("spatial_coverage", []):
        box = _envelope(entry)
        if box and box[0] <= 0 <= box[1] and box[2] <= 0 <= box[3] and "number_zero_coordinates" not in entry:
            cols = ", ".join(_display_name(n) for n in entry.get("column_names", []))
            notes.append(
                f"The bounding box of {cols} contains (0, 0), often a placeholder for missing coordinates; "
                "the box then says little about where the data is."
            )
    return notes


def _overview(meta: dict) -> str:
    rows = meta.get("nb_rows")
    profiled = meta.get("nb_profiled_rows")
    total_time = (meta.get("_profiling_times") or {}).get("total")
    cards = [
        ("Rows", _fmt_num(rows)),
        ("Profiled rows", _fmt_num(profiled) + (
            f' <span class="sub">({profiled / rows:.1%})</span>' if rows and profiled and profiled < rows else ""
        )),
        ("Columns", _fmt_num(meta.get("nb_columns"))),
        ("Size", _fmt_bytes(meta["size"]) if "size" in meta else "–"),
        ("Profiling time", _fmt_seconds(total_time) if total_time is not None else "–"),
    ]
    counts = [
        (label, meta.get(key)) for label, key in (
            ("numerical", "nb_numerical_columns"), ("categorical", "nb_categorical_columns"),
            ("spatial", "nb_spatial_columns"), ("temporal", "nb_temporal_columns"),
        ) if meta.get(key)
    ]
    chips = "".join(f'<span class="chip">{_esc(label)} · {n}</span>' for label, n in counts)
    cards_html = "".join(f'<div class="card"><div class="k">{k}</div><div class="v">{v}</div></div>' for k, v in cards)
    out = f'<section id="overview"><div class="chips">{chips}</div><div class="cards">{cards_html}</div>'
    warnings = _warnings(meta)
    if warnings:
        items = "".join(f"<li>{w}</li>" for w in warnings)
        out += f'<details class="warn" open><summary>{len(warnings)} warning(s)</summary><ul>{items}</ul></details>'
    return out + "</section>"


def _columns_table(meta: dict, with_hist: set[str]) -> str:
    rows = []
    for i, c in enumerate(meta.get("columns", []), start=1):
        name = str(c.get("name", "")).strip()
        semantic = [_short_type(t) for t in c.get("semantic_types", [])]
        tags = "".join(f'<span class="tag">{_esc(t)}</span>' for t in semantic)
        geo = c.get("geo_classifier")
        if isinstance(geo, dict) and geo.get("label"):
            conf = geo.get("confidence")
            tags += f'<span class="tag ml" title="Spatial classifier">ML {_esc(geo["label"])}{f" {conf:.0%}" if isinstance(conf, (int, float)) else ""}</span>'
        missing = c.get("missing_values_ratio")
        distinct = c.get("num_distinct_values")
        approx = "≈" if c.get("num_distinct_values_is_approximate") else ""
        ranges = "; ".join(f"{_fmt_num(a)} – {_fmt_num(b)}" for a, b in _ranges(c.get("coverage")))
        link = f'<a href="#{_anchor(name)}">{_display_name(c.get("name", ""))}</a>' if name in with_hist else _display_name(c.get("name", ""))
        rows.append(
            "<tr>"
            f'<td class="num muted">{i}</td><td class="name">{link}</td>'
            f'<td><span class="type">{_esc(_short_type(c.get("structural_type", "")))}</span>{tags}</td>'
            f'<td class="num">{f"{missing:.1%}" if missing is not None else ""}</td>'
            f'<td class="num">{approx}{_fmt_num(distinct) if distinct is not None else ""}</td>'
            f'<td class="num">{_fmt_num(c["mean"]) if "mean" in c else ""}</td>'
            f'<td class="num">{_fmt_num(c["stddev"]) if "stddev" in c else ""}</td>'
            f'<td class="num">{_fmt_num(c["min"]) if "min" in c else ""}</td>'
            f'<td class="num">{_fmt_num(c["max"]) if "max" in c else ""}</td>'
            f'<td class="ranges">{_esc(ranges)}</td>'
            "</tr>"
        )
    head = (
        "<tr><th>#</th><th>Column</th><th>Types</th><th>Missing</th><th>Distinct</th>"
        "<th>Mean</th><th>Std dev</th><th>Min</th><th>Max</th><th>Value ranges</th></tr>"
    )
    note = '<p class="sub">Value ranges are the clusters of values the profiler found (outer 5% trimmed). Click a name to see its distribution.</p>'
    return f'<section id="columns"><h2>Columns</h2><div class="scroll"><table class="cols">{head}{"".join(rows)}</table></div>{note}</section>'


def _distributions(meta: dict) -> tuple[str, set[str]]:
    built = _distributions_spec(meta)
    if built is None:
        return ('<section id="distributions"><h2>Distributions</h2><p class="sub">The profile has no histograms; '
                "profile with <code>plots=True</code> to include them.</p></section>"), set()
    spec, captions, missing = built
    rows = math.ceil(len(spec.views) / _HIST_COLUMNS)
    height = rows * (_HIST_HEIGHT + 12) + 12
    names = [v.title for v in spec.views]
    anchors = "".join(f'<span id="{_anchor(n)}"></span>' for n in names)
    notes = "".join(f"<li><b>{_esc(n)}</b>: {_esc(text)}</li>" for n, text in captions)
    if missing:
        notes += f"<li>No histogram for {', '.join(_esc(n) for n in missing)} (identifier-like or empty).</li>"
    body = (
        f'<section id="distributions"><h2>Distributions</h2>{anchors}'
        f'<p class="sub">From the profile\'s histograms, computed on the profiled rows.</p>'
        f'{_autark_frame(spec, height, "Distributions")}'
        f'{f"<ul class=notes>{notes}</ul>" if notes else ""}</section>'
    )
    return body, set(names)


def _spatial(meta: dict) -> str:
    built = _maps_spec(meta)
    if built is None:
        return ""
    spec, captions = built
    notes = "".join(f"<li>{c}</li>" for c in captions)
    return (
        f'<section id="spatial"><h2>Spatial coverage</h2>{_autark_frame(spec, _MAP_HEIGHT, "Spatial coverage", borderless_maps=True)}'
        f'<ul class="notes">{notes}</ul></section>'
    )


def _fmt_interval(seconds: float | None) -> str:
    if seconds is None:
        return ""
    if seconds < 120:
        return f"{seconds:,.0f} s"
    if seconds < 7200:
        return f"{seconds / 60:,.1f} min"
    if seconds < 172800:
        return f"{seconds / 3600:,.1f} h"
    return f"{seconds / 86400:,.1f} days"


def _temporal(meta: dict) -> str:
    rows = []
    has_sampling = any("sampling" in e for e in meta.get("temporal_coverage") or [])
    for entry in meta.get("temporal_coverage") or []:
        names = ", ".join(_display_name(n) for n in entry.get("column_names", []))
        precision = entry.get("temporal_precision") or entry.get("temporal_resolution") or "–"
        sampling = entry.get("sampling") or {}
        extra = ""
        if has_sampling:
            regular = sampling.get("regular")
            extra = (
                f'<td class="num">{_fmt_interval(sampling.get("median_interval_seconds"))}</td>'
                f'<td>{"" if regular is None else ("regular" if regular else "irregular")}</td>'
                f'<td class="num">{_fmt_interval(sampling.get("largest_gap_seconds"))}</td>'
            )
        for a, b in _ranges(entry.get("ranges")):
            rows.append(
                f'<tr><td class="name">{names}</td><td><span class="tag">{_esc(precision)}</span></td>'
                f'<td class="num">{_fmt_ts(a)}</td><td class="num">{_fmt_ts(b)}</td><td class="num">{_fmt_span(b - a)}</td>{extra}</tr>'
            )
    if not rows:
        return ""
    head = "<tr><th>Column</th><th>Precision</th><th>Start (UTC)</th><th>End (UTC)</th><th>Span</th>"
    if has_sampling:
        head += "<th>Median interval</th><th>Spacing</th><th>Largest gap</th>"
    head += "</tr>"
    note = ('<p class="sub">Precision is the finest unit the timestamps use. The interval and gap are between '
            "consecutive distinct timestamps over all rows; spacing is regular when one interval accounts for 90% of them.</p>")
    return f'<section id="temporal"><h2>Temporal coverage</h2><table class="cols compact">{head}{"".join(rows)}</table>{note}</section>'


def _timings(meta: dict) -> str:
    steps = (meta.get("_profiling_times") or {}).get("steps") or {}
    if not steps:
        return ""
    total = sum(steps.values()) or 1
    rows = "".join(
        f'<tr><td>{_esc((k.split("_", 1)[-1] if k[:1].isdigit() else k).replace("_", " "))}</td>'
        f'<td class="num">{_fmt_seconds(v)}</td><td class="num muted">{v / total:.0%}</td></tr>'
        for k, v in steps.items()
    )
    return f'<section id="timing"><h2>Profiling time</h2><table class="cols compact">{rows}</table></section>'


_CSS = """
* { box-sizing:border-box; }
body { margin:0; background:#f6f7f9; color:#1f2933; font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif; }
nav { position:sticky; top:0; z-index:5; background:#fff; border-bottom:1px solid #e2e8f0; }
nav .in { max-width:1180px; margin:0 auto; padding:8px 20px; display:flex; gap:18px; align-items:baseline; flex-wrap:wrap; }
nav b { font-size:15px; margin-right:8px; }
nav a { color:#475569; text-decoration:none; font-size:13px; } nav a:hover { color:#2563eb; }
main { max-width:1180px; margin:0 auto; padding:8px 20px 40px; }
section { background:#fff; border:1px solid #e2e8f0; border-radius:10px; padding:14px 18px; margin-top:14px; scroll-margin-top:52px; }
h2 { font-size:15px; margin:0 0 10px; }
.sub, .muted { color:#64748b; font-size:12px; }
.chip, .tag, .type { display:inline-block; border-radius:10px; padding:0 7px; font-size:11.5px; margin:1px 3px 1px 0;
  border:1px solid #e2e8f0; background:#f8fafc; white-space:nowrap; }
.type { font-weight:600; } .tag.ml { border-color:#bfdbfe; color:#1d4ed8; }
.cards { display:grid; grid-template-columns:repeat(5,1fr); gap:10px; margin-top:10px; }
.card { background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; padding:8px 12px; }
.card .k { color:#64748b; font-size:12px; } .card .v { font-size:17px; font-weight:600; }
details.warn { margin-top:12px; background:#fff7ed; color:#9a3412; border-radius:8px; padding:8px 12px; }
details.warn summary { cursor:pointer; font-weight:600; } details.warn ul { margin:6px 0 2px; padding-left:20px; }
.ws { background:#ffedd5; border-radius:3px; }
.scroll { overflow-x:auto; }
table.cols { border-collapse:collapse; width:100%; font-size:13px; }
table.cols th { text-align:left; font-size:12px; color:#64748b; font-weight:600; border-bottom:1px solid #e2e8f0; padding:5px 8px; white-space:nowrap; }
table.cols td { border-bottom:1px solid #f1f5f9; padding:5px 8px; vertical-align:top; }
table.cols tr:hover td { background:#f8fafc; }
table.compact { width:auto; }
td.num { text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; }
td.name { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:12.5px; white-space:nowrap; }
td.name a { color:#1d4ed8; text-decoration:none; } td.name a:hover { text-decoration:underline; }
td.ranges { font-size:12px; color:#475569; }
iframe.autark { width:100%; border:0; display:block; }
ul.notes { margin:8px 0 0; padding-left:20px; color:#475569; font-size:12.5px; }
@media (max-width: 800px) { .cards { grid-template-columns:repeat(2,1fr); } }
"""


def render_html(metadata: dict, title: str | None = None) -> str:
    """The report as a standalone HTML page."""
    heading = title or metadata.get("name") or "Dataset profile"
    distributions, with_hist = _distributions(metadata)
    sections = [
        ("overview", "Overview", _overview(metadata)),
        ("columns", "Columns", _columns_table(metadata, with_hist)),
        ("distributions", "Distributions", distributions),
        ("spatial", "Spatial", _spatial(metadata)),
        ("temporal", "Temporal", _temporal(metadata)),
        ("timing", "Profiling time", _timings(metadata)),
    ]
    links = "".join(f'<a href="#{sid}">{label}</a>' for sid, label, body in sections if body)
    body = "".join(b for _, _, b in sections if b)
    return (
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
        f"<title>{_esc(heading)}</title><style>{_CSS}</style></head><body>"
        f'<nav><div class="in"><b>{_esc(heading)}</b>{links}</div></nav><main>{body}</main></body></html>'
    )


def save_html(metadata: dict, path: str | Path, title: str | None = None) -> Path:
    """Write the report for ``metadata`` to ``path`` and return the path."""
    path = Path(path)
    path.write_text(render_html(metadata, title=title), encoding="utf-8")
    return path


class ProfileReport:
    """A report of a profile; save it as a page, or display it in Jupyter."""

    def __init__(self, metadata: dict, title: str | None = None):
        self.metadata = metadata
        self.title = title

    @classmethod
    def from_json(cls, path: str | Path, title: str | None = None) -> "ProfileReport":
        """A report from a profile saved as JSON."""
        path = Path(path)
        return cls(json.loads(path.read_text(encoding="utf-8")), title=title or path.stem)

    def save(self, path: str | Path) -> Path:
        return save_html(self.metadata, path, title=self.title)

    def _repr_html_(self) -> str:
        page = render_html(self.metadata, title=self.title)
        return f'<iframe srcdoc="{html.escape(page, quote=True)}" style="width:100%;height:900px;border:0"></iframe>'


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Make an HTML report of an Atlas Profiler profile (JSON), drawn with Autark.")
    parser.add_argument("profile", help="Profile JSON written from process_dataset's output")
    parser.add_argument("-o", "--output", help="Output HTML path (default: next to the profile, .html)")
    parser.add_argument("--title", help="Report title")
    args = parser.parse_args(argv)
    report = ProfileReport.from_json(args.profile, title=args.title)
    out = Path(args.output) if args.output else Path(args.profile).with_suffix(".html")
    print(report.save(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
