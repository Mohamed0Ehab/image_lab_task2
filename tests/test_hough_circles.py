"""Test script for hough_circles – synthetic edge maps.

Run from the project root:
    python -m tests.test_hough_circles
"""
from __future__ import annotations

import sys, os
import numpy as np

# Ensure the project root is on the path so `core` is importable.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.circles_ellipses import hough_circles


def make_circle_edge_map(
    height: int,
    width: int,
    circles: list[tuple[int, int, int]],
    noise_fraction: float = 0.0,
    gap_fraction: float = 0.0,
) -> np.ndarray:
    """Draw perfect circles on a blank image and optionally add noise / gaps.

    Parameters
    ----------
    circles : list of (cx, cy, r)
    noise_fraction : fraction of image pixels to set as random noise edges.
    gap_fraction   : fraction of each circle's perimeter to erase.
    """
    img = np.zeros((height, width), dtype=np.uint8)

    for cx, cy, r in circles:
        n_pts = max(int(2 * np.pi * r * 2), 360)  # oversample
        thetas = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)

        # Optionally create a gap
        if gap_fraction > 0:
            keep = int(n_pts * (1 - gap_fraction))
            thetas = thetas[:keep]

        xs = np.round(cx + r * np.cos(thetas)).astype(int)
        ys = np.round(cy + r * np.sin(thetas)).astype(int)
        valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
        img[ys[valid], xs[valid]] = 255

    # Add salt noise
    if noise_fraction > 0:
        n_noise = int(height * width * noise_fraction)
        rs = np.random.randint(0, height, n_noise)
        cs = np.random.randint(0, width, n_noise)
        img[rs, cs] = 255

    return img


def print_results(tag: str, circles):
    print(f"\n{'='*60}")
    print(f"  {tag}")
    print(f"{'='*60}")
    if not circles:
        print("  (no circles detected)")
    for c in circles:
        print(f"  cx={c.cx:4d}  cy={c.cy:4d}  r={c.r:4.0f}")
    print()


def test_single_circle():
    """One circle, easy case."""
    print("\n>>> TEST 1: Single circle (cx=100, cy=100, r=40)")
    edge_map = make_circle_edge_map(200, 200, [(100, 100, 40)])
    for th in (0.3, 0.5, 0.7):
        res = hough_circles(edge_map, 30, 60, threshold=th)
        print_results(f"threshold={th}", res)


def test_multiple_circles():
    """Several circles of different sizes."""
    print("\n>>> TEST 2: Three circles")
    circles_gt = [(80, 80, 30), (200, 150, 50), (320, 100, 25)]
    edge_map = make_circle_edge_map(300, 400, circles_gt)
    res = hough_circles(edge_map, 20, 60, threshold=0.5)
    print_results("Multiple circles  threshold=0.5", res)


def test_noisy_circle_with_gaps():
    """Circle with 30 % gap and 1 % noise."""
    print("\n>>> TEST 3: Noisy circle with 30% gap")
    edge_map = make_circle_edge_map(
        200, 200, [(100, 100, 45)],
        noise_fraction=0.01, gap_fraction=0.30,
    )
    for th in (0.3, 0.5, 0.7):
        res = hough_circles(edge_map, 35, 55, threshold=th)
        print_results(f"Noisy+gap  threshold={th}", res)


def test_input_validation():
    """Ensure bad inputs raise ValueError, not a crash."""
    print("\n>>> TEST 4: Input validation")
    edge_map = np.zeros((100, 100), dtype=np.uint8)

    tests = [
        ("3-D array",   lambda: hough_circles(np.zeros((5, 5, 3)), 5, 10)),
        ("r_min > r_max", lambda: hough_circles(edge_map, 20, 5)),
        ("r_min == 0",  lambda: hough_circles(edge_map, 0, 10)),
        ("threshold 0", lambda: hough_circles(edge_map, 5, 10, threshold=0)),
        ("threshold 2", lambda: hough_circles(edge_map, 5, 10, threshold=2)),
    ]
    for name, fn in tests:
        try:
            fn()
            print(f"  FAIL – {name}: expected ValueError")
        except ValueError as e:
            print(f"  OK   – {name}: {e}")
        except Exception as e:
            print(f"  FAIL – {name}: unexpected {type(e).__name__}: {e}")

    # Empty edge map should return []
    result = hough_circles(edge_map, 5, 10)
    assert result == [], f"Expected [] for blank image, got {result}"
    print("  OK   – empty edge map returns []")


if __name__ == "__main__":
    np.random.seed(42)
    test_input_validation()
    test_single_circle()
    test_multiple_circles()
    test_noisy_circle_with_gaps()

    print("=" * 60)
    print("TUNING NOTES")
    print("=" * 60)
    print("""
  * threshold too LOW  (< 0.3) -> false positives: random edge clusters
    accumulate enough votes to pass the bar.
  * threshold too HIGH (> 0.7) -> missed circles: incomplete edges (gaps,
    noise) cannot meet the strict vote requirement.
  * Recommended range: 0.4 - 0.6 for typical Canny edge maps.
    Start at 0.5 and adjust based on edge quality.
""")
