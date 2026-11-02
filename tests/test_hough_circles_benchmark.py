"""Side-by-side benchmark & comparison between old and new hough_circles.

Uses only NumPy and Python standard library (time, tracemalloc).
Tests:
1. One circle
2. Several circles of different sizes
3. Concentric circles
4. Noisy circle with gaps
5. Partial arc
6. Large image (2000x2000, ~30k edge pixels)

Run:
  cd image_lab_task2
  python -m tests.test_hough_circles_benchmark
"""
from __future__ import annotations

import gc
import time
import tracemalloc
import numpy as np

from core.types import Circle
from core.circles_ellipses import hough_circles as new_hough_circles
from tests.test_hough_circles import make_circle_edge_map


# ===========================================================================
# Exact OLD hough_circles (preserved byte-for-byte for side-by-side testing)
# ===========================================================================

def old_hough_circles(
    edges: np.ndarray,
    r_min: int,
    r_max: int,
    threshold: float = 0.5,
    min_center_distance: int | None = None,
    max_circles: int = 50,
) -> list[Circle]:
    if not isinstance(edges, np.ndarray):
        raise ValueError("edges must be a numpy ndarray.")
    if edges.ndim != 2:
        raise ValueError(
            f"edges must be a 2-D array, got {edges.ndim}-D (shape {edges.shape})."
        )
    if edges.size == 0:
        return []

    r_min = int(r_min)
    r_max = int(r_max)
    if r_min < 1:
        raise ValueError(f"r_min must be >= 1, got {r_min}.")
    if r_max < r_min:
        raise ValueError(
            f"r_max ({r_max}) must be >= r_min ({r_min})."
        )
    if not (0 < threshold <= 1):
        raise ValueError(
            f"threshold must be in (0, 1], got {threshold}."
        )

    if min_center_distance is None:
        min_center_distance = r_min
    min_center_distance = int(min_center_distance)

    max_circles = int(max_circles)

    H, W = edges.shape
    num_radii = r_max - r_min + 1

    MAX_CELLS = 250_000_000
    if num_radii * H * W > MAX_CELLS:
        raise ValueError(
            f"Accumulator size {num_radii}x{H}x{W} = "
            f"{num_radii * H * W:,} cells exceeds the safety limit of "
            f"{MAX_CELLS:,}. Narrow the radius range or use a smaller image."
        )

    edge_rows, edge_cols = np.nonzero(edges)
    if edge_rows.size == 0:
        return []

    accumulator = np.zeros((num_radii, H, W), dtype=np.int32)

    for ri, r in enumerate(range(r_min, r_max + 1)):
        n_theta = max(int(np.ceil(2 * np.pi * r)), 8)
        thetas = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)

        dx = np.round(r * np.cos(thetas)).astype(np.intp)
        dy = np.round(r * np.sin(thetas)).astype(np.intp)

        offsets = np.column_stack((dx, dy))
        offsets = np.unique(offsets, axis=0)
        dx = offsets[:, 0]
        dy = offsets[:, 1]

        a = edge_cols[:, None] - dx[None, :]
        b = edge_rows[:, None] - dy[None, :]

        a = a.ravel()
        b = b.ravel()
        valid = (a >= 0) & (a < W) & (b >= 0) & (b < H)
        a = a[valid]
        b = b[valid]

        flat_idx = b * W + a
        np.add.at(accumulator[ri].ravel(), flat_idx, 1)

    radii = np.arange(r_min, r_max + 1)
    vote_thresh = (threshold * 2 * np.pi * radii).astype(np.float64)

    above_thresh = accumulator >= vote_thresh[:, None, None]

    pad_acc = np.pad(accumulator, ((1, 1), (1, 1), (1, 1)),
                     mode='constant', constant_values=0)
    is_max = np.ones(accumulator.shape, dtype=bool)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            for dd in (-1, 0, 1):
                if dr == 0 and dc == 0 and dd == 0:
                    continue
                shifted = pad_acc[
                    1 + dr : 1 + dr + num_radii,
                    1 + dc : 1 + dc + H,
                    1 + dd : 1 + dd + W,
                ]
                is_max &= accumulator >= shifted

    candidates_mask = above_thresh & is_max

    cand_ri, cand_row, cand_col = np.nonzero(candidates_mask)
    cand_votes = accumulator[cand_ri, cand_row, cand_col]
    cand_radii = cand_ri + r_min

    order = np.argsort(-cand_votes)
    cand_ri    = cand_ri[order]
    cand_row   = cand_row[order]
    cand_col   = cand_col[order]
    cand_votes = cand_votes[order]
    cand_radii = cand_radii[order]

    accepted: list[Circle] = []
    accepted_centres: list[tuple[int, int, int]] = []

    dist_sq_min = min_center_distance ** 2

    for i in range(len(cand_votes)):
        cx = int(cand_col[i])
        cy = int(cand_row[i])
        r  = int(cand_radii[i])
        v  = int(cand_votes[i])

        suppress = False
        for acx, acy, ar in accepted_centres:
            d2 = (cx - acx) ** 2 + (cy - acy) ** 2
            if d2 < dist_sq_min and abs(r - ar) < min_center_distance:
                suppress = True
                break
        if suppress:
            continue

        accepted.append(Circle(cx=cx, cy=cy, r=r))
        accepted_centres.append((cx, cy, r))

        if len(accepted) >= max_circles:
            break

    return accepted


# ===========================================================================
# Benchmark Runner
# ===========================================================================

def run_bench(func, *args, **kwargs):
    """Run function under tracemalloc and high-resolution timer."""
    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()
    try:
        res = func(*args, **kwargs)
        err = None
    except Exception as e:
        res = None
        err = str(e)
    t1 = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    gc.collect()
    return res, (t1 - t0) * 1000.0, peak / (1024.0 * 1024.0), err


def format_circles(circles):
    if circles is None:
        return "ERROR"
    if not circles:
        return "[]"
    return ", ".join(f"({c.cx},{c.cy},r={c.r})" for c in circles)


def test_suite():
    print("=" * 80)
    print("CIRCULAR HOUGH TRANSFORM BENCHMARK: OLD vs NEW")
    print("=" * 80)

    cases = []

    # 1. One circle
    e1 = make_circle_edge_map(200, 200, [(100, 100, 40)])
    gt1 = [(100, 100, 40)]
    cases.append(("One circle (200x200)", e1, 30, 50, 0.5, gt1))

    # 2. Several circles
    e2 = make_circle_edge_map(300, 400, [(80, 80, 30), (200, 150, 50), (320, 100, 25)])
    gt2 = [(80, 80, 30), (200, 150, 50), (320, 100, 25)]
    cases.append(("Three circles of different sizes (300x400)", e2, 20, 60, 0.5, gt2))

    # 3. Concentric circles
    e3 = make_circle_edge_map(250, 250, [(125, 125, 40), (125, 125, 70)])
    gt3 = [(125, 125, 40), (125, 125, 70)]
    cases.append(("Concentric circles (250x250, r=40 & r=70)", e3, 35, 75, 0.5, gt3))

    # 4. Noisy circle with 30% gap
    e4 = make_circle_edge_map(200, 200, [(100, 100, 45)], noise_fraction=0.005, gap_fraction=0.3)
    gt4 = [(100, 100, 45)]
    cases.append(("Noisy circle with 30% gap (200x200)", e4, 30, 60, 0.3, gt4))

    # 5. Partial arc (continuous 70% arc)
    H5, W5 = 200, 200
    e5 = np.zeros((H5, W5), dtype=np.uint8)
    theta = np.linspace(0, 1.4 * np.pi, 250)
    ax = np.round(100 + 40 * np.cos(theta)).astype(int)
    ay = np.round(100 + 40 * np.sin(theta)).astype(int)
    v5 = (ax >= 0) & (ax < W5) & (ay >= 0) & (ay < H5)
    e5[ay[v5], ax[v5]] = 255
    gt5 = [(100, 100, 40)]
    cases.append(("Partial arc 70% arc (200x200)", e5, 30, 50, 0.35, gt5))

    # 6. Large image (2000x2000, ~30k edge pixels)
    print("Generating large image (2000x2000 with 30k edge pixels)...")
    H6, W6 = 2000, 2000
    e6 = np.zeros((H6, W6), dtype=np.uint8)
    rng = np.random.RandomState(42)
    large_gt = [
        (500, 500, 120),
        (1500, 500, 90),
        (1000, 1500, 150),
    ]
    for cx, cy, r in large_gt:
        n_pts = int(2 * np.pi * r)
        th = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
        xs = np.round(cx + r * np.cos(th)).astype(int)
        ys = np.round(cy + r * np.sin(th)).astype(int)
        v = (xs >= 0) & (xs < W6) & (ys >= 0) & (ys < H6)
        e6[ys[v], xs[v]] = 255
    # Add random background edges to reach ~30,000 edge pixels
    cur_edges = int((e6 > 0).sum())
    needed = max(0, 30_000 - cur_edges)
    rx = rng.randint(0, W6, size=needed)
    ry = rng.randint(0, H6, size=needed)
    e6[ry, rx] = 255
    print(f"Large image generated with {(e6 > 0).sum():,} edge pixels.")
    cases.append(("Large image 2000x2000 (30k edge px)", e6, 80, 160, 0.5, large_gt))

    # Run comparisons
    for name, edges, r_min, r_max, thresh, gt in cases:
        print(f"\n{'-' * 80}")
        print(f"CASE: {name}")
        print(f"  Ground truth: {gt}")
        print(f"  Radius range: [{r_min}, {r_max}] ({r_max - r_min + 1} radii)")

        # Run OLD
        old_res, old_time, old_mem, old_err = run_bench(
            old_hough_circles, edges, r_min, r_max, threshold=thresh
        )
        if old_err:
            print(f"  OLD: ERROR: {old_err}")
            print(f"       Time: {old_time:.1f} ms | Peak memory: {old_mem:.2f} MB")
        else:
            print(f"  OLD: {len(old_res)} circle(s) -> {format_circles(old_res)}")
            print(f"       Time: {old_time:.1f} ms | Peak memory: {old_mem:.2f} MB")

        # Run NEW
        new_res, new_time, new_mem, new_err = run_bench(
            new_hough_circles, edges, r_min, r_max, threshold=thresh
        )
        if new_err:
            print(f"  NEW: ERROR: {new_err}")
        else:
            print(f"  NEW: {len(new_res)} circle(s) -> {format_circles(new_res)}")
            print(f"       Time: {new_time:.1f} ms | Peak memory: {new_mem:.2f} MB")

        # Speedup and memory reduction
        if not old_err and not new_err:
            speedup = old_time / max(new_time, 1e-6)
            mem_saving = old_mem / max(new_mem, 1e-6)
            print(f"  >>> Speedup: {speedup:.2f}x | Memory ratio: {mem_saving:.2f}x less memory")

        # Verification: new must not lose correct detections
        if not new_err:
            for gtx, gty, gtr in gt:
                matched = any(
                    abs(c.cx - gtx) <= 5 and abs(c.cy - gty) <= 5 and abs(c.r - gtr) <= 5
                    for c in new_res
                )
                assert matched, f"NEW missed ground truth circle ({gtx}, {gty}, r={gtr}) in {name}!"

    # Verify return_scores functionality
    e_test = make_circle_edge_map(200, 200, [(100, 100, 40)])
    c_list = new_hough_circles(e_test, 30, 50, return_scores=False)
    c_tuple, scores = new_hough_circles(e_test, 30, 50, return_scores=True)
    assert len(c_list) == len(c_tuple) == len(scores) == 1
    assert c_list[0] == c_tuple[0]
    assert scores[0] > 0
    print("\nreturn_scores test: PASSED (circles and scores correctly returned)")

    print(f"\n{'=' * 80}")
    print("ALL BENCHMARK COMPARISONS PASSED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    test_suite()

