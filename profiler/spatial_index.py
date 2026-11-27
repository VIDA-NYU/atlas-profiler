"""Spatial summaries of point sets, computed on whole arrays with numpy.

``kdtree_leaves`` partitions the points with a 2-d k-d tree split at the median, as TaxiVis indexes taxi
trips. Every leaf holds about the same number of points, so the cells are small where the data is dense
and large where it is sparse: the grid adapts to the data, instead of a fixed cell size that is too
coarse for a city or too fine for a country. Unlike TaxiVis, which cycles through its keys, each node is
split across its longer side (in kilometres), which keeps cells close to square instead of long slivers.

``geohash_counts`` computes the base-4 geohash cells of ``profiler.spatial.Geohasher`` (the
``geohashes4`` field) without a Python loop over the points.
"""

from __future__ import annotations

import math

import numpy

KDTREE_LEAVES = 4096       # leaves of the spatial k-d tree (a power of two; rounded up otherwise)
KDTREE_MIN_LEAF_POINTS = 8  # a node with fewer than twice this is not split
_GEOHASH_LEVELS = 16


def kdtree_leaves(lon, lat, bounds, max_leaves: int = KDTREE_LEAVES, min_leaf_points: int = KDTREE_MIN_LEAF_POINTS):
    """Leaves of a median-split k-d tree over the points, as ``[min_lon, min_lat, max_lon, max_lat, count]``.

    The root is ``bounds`` (``(min_lon, min_lat, max_lon, max_lat)``) and the leaves tile it: each split
    cuts its node across its longer side (in kilometres) at the median coordinate of its points, by rank
    (so repeated values split too), until ``max_leaves`` leaves or nodes too small to split.
    """
    pts = numpy.column_stack([numpy.asarray(lon, dtype="float64"), numpy.asarray(lat, dtype="float64")])
    n = len(pts)
    if not n:
        return []
    max_depth = max(0, math.ceil(math.log2(max(int(max_leaves), 1))))
    leaves = []
    stack = [(0, n, 0, tuple(float(b) for b in bounds))]
    while stack:
        start, end, depth, (x0, y0, x1, y1) = stack.pop()
        size = end - start
        if depth >= max_depth or size < 2 * min_leaf_points:
            leaves.append([x0, y0, x1, y1, int(size)])
            continue
        width_km = (x1 - x0) * 111.32 * max(math.cos(math.radians((y0 + y1) / 2)), 0.01)
        height_km = (y1 - y0) * 110.57
        axis = 0 if width_km >= height_km else 1
        segment = pts[start:end]
        k = size // 2
        order = numpy.argpartition(segment[:, axis], k)
        pts[start:end] = segment[order]
        split = float(pts[start + k, axis])
        if axis == 0:
            left, right = (x0, y0, split, y1), (split, y0, x1, y1)
        else:
            left, right = (x0, y0, x1, split), (x0, split, x1, y1)
        stack.append((start + k, end, depth + 1, right))
        stack.append((start, start + k, depth + 1, left))
    leaves.sort(key=lambda leaf: (leaf[1], leaf[0]))
    return leaves


def _spread_bits(v: numpy.ndarray) -> numpy.ndarray:
    """Spread the 16 low bits of ``v`` to the even bit positions of a 32-bit word (Morton order)."""
    v = v.astype("uint64") & 0xFFFF
    v = (v | (v << 8)) & 0x00FF00FF
    v = (v | (v << 4)) & 0x0F0F0F0F
    v = (v | (v << 2)) & 0x33333333
    v = (v | (v << 1)) & 0x55555555
    return v


def geohash_counts(lat, lon, number: int, precision: int = _GEOHASH_LEVELS) -> list[dict]:
    """Base-4 geohash cells holding the points, as ``[{"hash": ..., "number": ...}]``.

    The same result as ``Geohasher(number=number, base=4, precision=precision)``: the cells at the
    deepest level that has at most ``number`` distinct cells.
    """
    lat = numpy.asarray(lat, dtype="float64")
    lon = numpy.asarray(lon, dtype="float64")
    if not len(lat):
        return []
    scale = float(1 << _GEOHASH_LEVELS)
    # Cell index along each axis at the finest level; a value on a boundary goes to the lower cell, as
    # ``location_to_bits`` (``value > mid``) does.
    lon_i = numpy.clip(numpy.ceil((lon + 180.0) / 360.0 * scale) - 1, 0, scale - 1).astype("uint64")
    lat_i = numpy.clip(numpy.ceil((lat + 90.0) / 180.0 * scale) - 1, 0, scale - 1).astype("uint64")
    # Each base-4 digit is (longitude bit, latitude bit), longitude first.
    codes = numpy.sort((_spread_bits(lon_i) << 1) | _spread_bits(lat_i))

    level = 0
    for candidate in range(1, min(precision, _GEOHASH_LEVELS) + 1):
        prefixes = codes >> (2 * (_GEOHASH_LEVELS - candidate))
        distinct = 1 + int(numpy.count_nonzero(prefixes[1:] != prefixes[:-1]))
        if distinct > number:
            break
        level = candidate
    if level == 0:
        return [{"hash": "", "number": int(len(codes))}]

    prefixes = codes >> (2 * (_GEOHASH_LEVELS - level))
    starts = numpy.flatnonzero(numpy.concatenate([[True], prefixes[1:] != prefixes[:-1]]))
    counts = numpy.diff(numpy.append(starts, len(prefixes)))
    out = []
    for prefix, count in zip(prefixes[starts], counts):
        digits = numpy.base_repr(int(prefix), 4).rjust(level, "0")
        out.append({"hash": digits, "number": int(count)})
    return out


GRID_CELLS = 200       # cells of the density grid along the longer side of its area
GRID_SHARE = 0.95      # the grid covers the densest k-d tree leaves holding this share of the points...
GRID_MARGIN = 0.25     # ...grown by this fraction of its size on each side


def dense_area(leaves, share: float = GRID_SHARE, margin: float = GRID_MARGIN):
    """Area holding most of the points: the box around the densest leaves (points per area) that together
    hold ``share`` of the points, grown by ``margin`` on each side. ``leaves`` as from ``kdtree_leaves``.
    """
    def density(leaf):
        x0, y0, x1, y1, n = leaf
        km2 = (x1 - x0) * 111.32 * max(math.cos(math.radians((y0 + y1) / 2)), 0.01) * (y1 - y0) * 110.57
        return n / max(km2, 1e-9)

    total = sum(leaf[4] for leaf in leaves) or 1
    chosen, acc = [], 0
    for leaf in sorted(leaves, key=density, reverse=True):
        chosen.append(leaf)
        acc += leaf[4]
        if acc >= share * total:
            break
    x0, y0 = min(leaf[0] for leaf in chosen), min(leaf[1] for leaf in chosen)
    x1, y1 = max(leaf[2] for leaf in chosen), max(leaf[3] for leaf in chosen)
    mx, my = (x1 - x0) * margin, (y1 - y0) * margin
    return x0 - mx, y0 - my, x1 + mx, y1 + my


def density_grid(lon, lat, area, cells: int = GRID_CELLS) -> dict:
    """Exact point counts on a grid of square cells covering ``area`` (min_lon, min_lat, max_lon, max_lat).

    Cells are square on the ground (equal sides in kilometres at the area's middle latitude), ``cells``
    of them along the longer side. Every point is counted in the one cell that contains it; points
    outside the area are counted in ``outside``. Only non-empty cells are listed, as ``[column, row,
    count]`` with column 0 at ``min_lon`` and row 0 at ``min_lat``.
    """
    lon = numpy.asarray(lon, dtype="float64")
    lat = numpy.asarray(lat, dtype="float64")
    x0, y0, x1, y1 = (float(v) for v in area)
    kx = 111.32 * max(math.cos(math.radians((y0 + y1) / 2)), 0.01)  # km per degree of longitude
    ky = 110.57
    size_km = max((x1 - x0) * kx, (y1 - y0) * ky, 1e-6) / cells
    dx, dy = size_km / kx, size_km / ky
    nx, ny = max(1, math.ceil((x1 - x0) / dx)), max(1, math.ceil((y1 - y0) / dy))
    col = numpy.floor((lon - x0) / dx).astype("int64")
    row = numpy.floor((lat - y0) / dy).astype("int64")
    inside = (col >= 0) & (col < nx) & (row >= 0) & (row < ny)
    flat = numpy.bincount(row[inside] * nx + col[inside], minlength=nx * ny)
    nonzero = numpy.flatnonzero(flat)
    return {
        "bounds": [x0, y0, x0 + nx * dx, y0 + ny * dy],
        "columns": int(nx),
        "rows": int(ny),
        "cell_degrees": [dx, dy],
        "cell_km": size_km,
        "outside": int(len(lon) - int(inside.sum())),
        "cells": [[int(k % nx), int(k // nx), int(flat[k])] for k in nonzero],
    }
