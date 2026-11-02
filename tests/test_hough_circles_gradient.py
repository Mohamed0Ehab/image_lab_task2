"""Test script comparing hough_circles (standard) and hough_circles_gradient.

Run from the project root:
    python -m tests.test_hough_circles_gradient
"""
from __future__ import annotations

import sys, os, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.circles_ellipses import hough_circles, hough_circles_gradient, detect_circles


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_circle_edge_map(
    height: int,
    width: int,
    circles: list[tuple[int, int, int]],
    noise_fraction: float = 0.0,
    gap_fraction: float = 0.0,
) -> np.ndarray:
    """Draw perfect circles on a blank image, optionally with noise/gaps."""
    img = np.zeros((height, width), dtype=np.uint8)
    for cx, cy, r in circles:
        n_pts = max(int(2 * np.pi * r * 2), 360)
        thetas = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
        if gap_fraction > 0:
            keep = int(n_pts * (1 - gap_fraction))
            thetas = thetas[:keep]
        xs = np.round(cx + r * np.cos(thetas)).astype(int)
        ys = np.round(cy + r * np.sin(thetas)).astype(int)
        valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
        img[ys[valid], xs[valid]] = 255
    if noise_fraction > 0:
        n = int(height * width * noise_fraction)
        img[np.random.randint(0, height, n),
            np.random.randint(0, width, n)] = 255
    return img


def run_both(tag: str, edge_map: np.ndarray, r_min: int, r_max: int,
             threshold: float = 0.5):
    """Run both methods, print results and timing."""
    print(f"\n{'='*65}")
    print(f"  {tag}")
    print(f"{'='*65}")

    # Standard
    t0 = time.perf_counter()
    res_std = hough_circles(edge_map, r_min, r_max, threshold=threshold)
    dt_std = time.perf_counter() - t0

    # Gradient
    t0 = time.perf_counter()
    res_grad = hough_circles_gradient(edge_map, r_min, r_max, threshold=threshold)
    dt_grad = time.perf_counter() - t0

    print(f"\n  Standard ({dt_std*1000:.1f} ms):")
    if not res_std:
        print("    (none)")
    for c in res_std:
        print(f"    cx={c.cx:4d}  cy={c.cy:4d}  r={c.r:4.0f}")

    print(f"\n  Gradient ({dt_grad*1000:.1f} ms):")
    if not res_grad:
        print("    (none)")
    for c in res_grad:
        print(f"    cx={c.cx:4d}  cy={c.cy:4d}  r={c.r:4.0f}")
    print()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_single_circle():
    """Single circle, easy case."""
    edge_map = make_circle_edge_map(200, 200, [(100, 100, 40)])
    run_both("TEST 1: Single circle (100,100,r=40)", edge_map, 30, 60, 0.5)


def test_multiple_circles():
    """Three circles of different sizes."""
    gt = [(80, 80, 30), (200, 150, 50), (320, 100, 25)]
    edge_map = make_circle_edge_map(300, 400, gt)
    run_both("TEST 2: Three circles (r=30,50,25)", edge_map, 20, 60, 0.5)


def test_concentric_circles():
    """Two concentric circles (same center, different radii)."""
    edge_map = make_circle_edge_map(250, 250, [(125, 125, 40), (125, 125, 70)])
    run_both("TEST 3: Concentric circles (r=40, r=70)", edge_map, 30, 80, 0.5)


def test_noisy_with_gaps():
    """Circle with 30% gap and 1% salt noise."""
    edge_map = make_circle_edge_map(
        200, 200, [(100, 100, 45)],
        noise_fraction=0.01, gap_fraction=0.30,
    )
    for th in (0.3, 0.5, 0.7):
        run_both(f"TEST 4: Noisy+gap  threshold={th}", edge_map, 35, 55, th)


def test_dispatcher():
    """Verify detect_circles dispatcher routes correctly."""
    print(f"\n{'='*65}")
    print("  TEST 5: detect_circles dispatcher")
    print(f"{'='*65}")
    edge_map = make_circle_edge_map(200, 200, [(100, 100, 40)])

    r1 = detect_circles(edge_map, 30, 50, method="standard")
    r2 = detect_circles(edge_map, 30, 50, method="gradient")
    print(f"  standard -> {len(r1)} circles")
    print(f"  gradient -> {len(r2)} circles")

    try:
        detect_circles(edge_map, 30, 50, method="magic")
        print("  FAIL: expected ValueError for unknown method")
    except ValueError as e:
        print(f"  OK: unknown method raises ValueError: {e}")
    print()


def test_input_validation():
    """Gradient method should also validate inputs."""
    print(f"\n{'='*65}")
    print("  TEST 6: Input validation (gradient)")
    print(f"{'='*65}")
    edge_map = np.zeros((100, 100), dtype=np.uint8)

    cases = [
        ("3-D array",    lambda: hough_circles_gradient(np.zeros((5, 5, 3)), 5, 10)),
        ("r_min > r_max", lambda: hough_circles_gradient(edge_map, 20, 5)),
        ("r_min == 0",   lambda: hough_circles_gradient(edge_map, 0, 10)),
        ("threshold 0",  lambda: hough_circles_gradient(edge_map, 5, 10, threshold=0)),
        ("threshold 2",  lambda: hough_circles_gradient(edge_map, 5, 10, threshold=2)),
        ("gx wrong shape", lambda: hough_circles_gradient(
            edge_map, 5, 10, gx=np.zeros((50, 50)), gy=np.zeros((100, 100)))),
    ]
    for name, fn in cases:
        try:
            fn()
            print(f"  FAIL - {name}: expected ValueError")
        except ValueError as e:
            print(f"  OK   - {name}: {e}")
        except Exception as e:
            print(f"  FAIL - {name}: unexpected {type(e).__name__}: {e}")

    # Blank edge map -> []
    assert hough_circles_gradient(edge_map, 5, 10) == []
    print("  OK   - empty edge map returns []")
    print()


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(42)

    test_input_validation()
    test_single_circle()
    test_multiple_circles()
    test_concentric_circles()
    test_noisy_with_gaps()
    test_dispatcher()

    print("=" * 65)
    print("TUNING NOTES")
    print("=" * 65)
    print("""
  * threshold too LOW  (< 0.3) -> false positives from random edge clusters.
    threshold too HIGH (> 0.7) -> missed/incomplete circles.
    Sensible range: 0.4 - 0.6 (start at 0.5).
  * min_gradient: raise for noisy images to discard weak spurious edges.
  * center_threshold: raise to prune more aggressively in stage 1 when
    the image has many edges (e.g. textured backgrounds).
""")

