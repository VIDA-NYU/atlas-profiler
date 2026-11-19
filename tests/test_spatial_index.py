import numpy
import pytest

from profiler.spatial import Geohasher
from profiler.spatial_index import geohash_counts, kdtree_leaves


@pytest.mark.parametrize("number", [16, 100, 1000])
def test_geohash_counts_match_geohasher(number):
    rng = numpy.random.default_rng(3)
    pts = numpy.column_stack([rng.uniform(40.6, 40.9, 5000), rng.uniform(-74.1, -73.7, 5000)])
    pts = numpy.vstack([pts, [[0.0, 0.0], [45.0, -90.0]]])  # values on cell boundaries
    g = Geohasher(number=number)
    g.add_points(pts)
    expected = sorted((d["hash"], d["number"]) for d in g.get_hashes_json())
    assert sorted((d["hash"], d["number"]) for d in geohash_counts(pts[:, 0], pts[:, 1], number)) == expected


def test_kdtree_leaves_tile_and_balance():
    rng = numpy.random.default_rng(4)
    lon, lat = rng.uniform(0, 10, 10_000), rng.uniform(0, 5, 10_000)
    leaves = kdtree_leaves(lon, lat, (0, 0, 10, 5), max_leaves=64)
    assert len(leaves) == 64
    assert sum(leaf[4] for leaf in leaves) == 10_000
    assert {leaf[4] for leaf in leaves} <= {156, 157}  # median splits: equal counts
    area = sum((x1 - x0) * (y1 - y0) for x0, y0, x1, y1, _ in leaves)
    assert area == pytest.approx(50.0)  # the leaves tile the root, without overlap
    assert all(0 <= x0 <= x1 <= 10 and 0 <= y0 <= y1 <= 5 for x0, y0, x1, y1, _ in leaves)


def test_kdtree_cells_follow_density():
    rng = numpy.random.default_rng(5)
    dense = rng.uniform(0, 1, (9000, 2))
    sparse = rng.uniform(0, 10, (1000, 2))
    pts = numpy.vstack([dense, sparse])
    leaves = kdtree_leaves(pts[:, 0], pts[:, 1], (0, 0, 10, 10), max_leaves=256)
    area = lambda leaf: (leaf[2] - leaf[0]) * (leaf[3] - leaf[1])  # noqa: E731
    inside = [area(leaf) for leaf in leaves if leaf[2] <= 1 and leaf[3] <= 1]
    outside = [area(leaf) for leaf in leaves if leaf[0] >= 1 or leaf[1] >= 1]
    assert numpy.median(inside) * 20 < numpy.median(outside)


def test_small_inputs():
    assert kdtree_leaves([], [], (0, 0, 1, 1)) == []
    assert kdtree_leaves([0.5], [0.5], (0, 0, 1, 1)) == [[0.0, 0.0, 1.0, 1.0, 1]]
    assert geohash_counts([], [], 100) == []


def test_density_grid_counts_points_exactly():
    from profiler.spatial_index import density_grid

    rng = numpy.random.default_rng(6)
    lon, lat = rng.uniform(-74.05, -73.75, 50_000), rng.uniform(40.60, 40.90, 50_000)
    g = density_grid(lon, lat, (-74.0, 40.65, -73.8, 40.85), cells=50)
    x0, y0 = g["bounds"][0], g["bounds"][1]
    dx, dy = g["cell_degrees"]
    brute = {}
    for x, y in zip(lon, lat):
        c, r = int((x - x0) // dx), int((y - y0) // dy)
        if 0 <= c < g["columns"] and 0 <= r < g["rows"]:
            brute[(c, r)] = brute.get((c, r), 0) + 1
    assert {(c, r): n for c, r, n in g["cells"]} == brute
    assert g["outside"] == 50_000 - sum(brute.values())
    # Square on the ground: equal sides in km
    km_x = dx * 111.32 * numpy.cos(numpy.radians(40.75))
    assert km_x == pytest.approx(dy * 110.57, rel=1e-6)


def test_dense_area_frames_the_dense_cluster():
    from profiler.spatial_index import dense_area

    rng = numpy.random.default_rng(7)
    pts = numpy.vstack([rng.uniform(0, 1, (9500, 2)), rng.uniform(0, 30, (500, 2))])
    leaves = kdtree_leaves(pts[:, 0], pts[:, 1], (0, 0, 30, 30), max_leaves=256)
    x0, y0, x1, y1 = dense_area(leaves, share=0.9, margin=0.0)
    assert x1 - x0 < 3 and y1 - y0 < 3
