"""Tests for detect_ellipses() -- synthetic edge maps.

Run:
  cd image_lab_task2
  python -m tests.test_detect_ellipses
"""
from __future__ import annotations

import time
import numpy as np

from core.circles_ellipses import detect_ellipses


# -----------------------------------------------------------------------
# Helper: draw an ellipse edge map
# -----------------------------------------------------------------------

def make_ellipse_edge_map(
    H: int,
    W: int,
    ellipses: list[tuple[float, float, float, float, float]],
    noise_fraction: float = 0.0,
    gap_fraction: float = 0.0,
    seed: int = 42,
) -> np.ndarray:
    """Create a binary edge map with one or more ellipses.

    Each ellipse is (cx, cy, a, b, angle_rad).
    gap_fraction : fraction of each ellipse's perimeter pixels to erase.
    noise_fraction : fraction of total pixels to sprinkle as random noise.
    """
    rng = np.random.RandomState(seed)
    img = np.zeros((H, W), dtype=np.uint8)

    for cx, cy, a, b, angle in ellipses:
        # Sample densely around the ellipse
        n_pts = max(int(4 * np.pi * max(a, b)), 360)
        theta = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
        x_ell = a * np.cos(theta)
        y_ell = b * np.sin(theta)
        # Rotate
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        x_rot = x_ell * cos_a - y_ell * sin_a + cx
        y_rot = x_ell * sin_a + y_ell * cos_a + cy
        # Round and filter
        xi = np.round(x_rot).astype(int)
        yi = np.round(y_rot).astype(int)
        valid = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
        xi = xi[valid]
        yi = yi[valid]

        # Apply gaps
        if gap_fraction > 0:
            keep = rng.rand(len(xi)) > gap_fraction
            xi = xi[keep]
            yi = yi[keep]

        img[yi, xi] = 255

    # Add noise
    if noise_fraction > 0:
        n_noise = int(noise_fraction * H * W)
        ny = rng.randint(0, H, n_noise)
        nx = rng.randint(0, W, n_noise)
        img[ny, nx] = 255

    return img


def report(test_name, ground_truth, detected, elapsed):
    """Print a compact test report."""
    print(f"\n{'=' * 60}")
    print(f"TEST: {test_name}")
    print(f"  Ground truth : {len(ground_truth)} ellipse(s)")
    print(f"  Detected     : {len(detected)} ellipse(s)")
    print(f"  Time         : {elapsed:.3f}s")
    for i, e in enumerate(detected):
        print(f"  [{i}] cx={e.cx:.1f} cy={e.cy:.1f} "
              f"a={e.a:.1f} b={e.b:.1f} angle={np.degrees(e.angle):.1f} deg")
    for i, gt in enumerate(ground_truth):
        cx, cy, a, b, ang = gt
        print(f"  GT[{i}] cx={cx:.1f} cy={cy:.1f} "
              f"a={a:.1f} b={b:.1f} angle={np.degrees(ang):.1f} deg")


# -----------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------

def test_single_ellipse():
    """One axis-aligned ellipse."""
    gt = [(150.0, 100.0, 60.0, 30.0, 0.0)]
    edges = make_ellipse_edge_map(200, 300, gt)
    t0 = time.time()
    result = detect_ellipses(edges, threshold=0.4)
    elapsed = time.time() - t0
    report("Single axis-aligned ellipse", gt, result, elapsed)
    assert len(result) >= 1, "Should detect at least 1 ellipse"
    e = result[0]
    assert abs(e.cx - 150) < 5, f"cx off: {e.cx}"
    assert abs(e.cy - 100) < 5, f"cy off: {e.cy}"
    assert abs(e.a - 60) < 8, f"a off: {e.a}"
    assert abs(e.b - 30) < 8, f"b off: {e.b}"
    print("  PASSED")


def test_rotated_ellipse():
    """One rotated ellipse."""
    angle = np.radians(35)
    gt = [(120.0, 120.0, 70.0, 25.0, angle)]
    edges = make_ellipse_edge_map(250, 250, gt)
    t0 = time.time()
    result = detect_ellipses(edges, threshold=0.4)
    elapsed = time.time() - t0
    report("Rotated ellipse (35 deg)", gt, result, elapsed)
    assert len(result) >= 1, "Should detect at least 1 ellipse"
    e = result[0]
    assert abs(e.cx - 120) < 5, f"cx off: {e.cx}"
    assert abs(e.cy - 120) < 5, f"cy off: {e.cy}"
    assert abs(e.a - 70) < 8, f"a off: {e.a}"
    assert abs(e.b - 25) < 8, f"b off: {e.b}"
    assert abs(np.degrees(e.angle) - 35) < 5, f"angle off: {np.degrees(e.angle)}"
    print("  PASSED")


def test_multiple_ellipses():
    """Three ellipses of different sizes and orientations."""
    gt = [
        (80.0,  80.0,  40.0, 20.0, 0.0),
        (200.0, 80.0,  50.0, 30.0, np.radians(45)),
        (140.0, 200.0, 60.0, 35.0, np.radians(-20)),
    ]
    edges = make_ellipse_edge_map(300, 300, gt)
    t0 = time.time()
    result = detect_ellipses(edges, threshold=0.3)
    elapsed = time.time() - t0
    report("Multiple ellipses (3)", gt, result, elapsed)
    assert len(result) >= 3, f"Should detect 3 ellipses, got {len(result)}"
    print("  PASSED")


def test_partial_arc():
    """Continuous partial arc covering ~70% of perimeter (30% missing sector)."""
    gt = [(150.0, 100.0, 55.0, 30.0, np.radians(15))]
    H, W = 200, 300
    cx, cy, a, b, angle = gt[0]
    n_pts = 500
    theta = np.linspace(0, 1.4 * np.pi, n_pts)
    xe, ye = a * np.cos(theta), b * np.sin(theta)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    xr = xe * cos_a - ye * sin_a + cx
    yr = xe * sin_a + ye * cos_a + cy
    xi = np.round(xr).astype(int)
    yi = np.round(yr).astype(int)
    valid = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
    edges = np.zeros((H, W), dtype=np.uint8)
    edges[yi[valid], xi[valid]] = 255

    t0 = time.time()
    result = detect_ellipses(edges, threshold=0.35)
    elapsed = time.time() - t0
    report("Partial arc (70% arc / 30% missing)", gt, result, elapsed)
    assert len(result) >= 1, "Should detect the ellipse from partial arc"
    e = result[0]
    assert abs(e.cx - 150) < 5, f"cx off: {e.cx}"
    assert abs(e.cy - 100) < 5, f"cy off: {e.cy}"
    assert abs(e.a - 55) < 8, f"a off: {e.a}"
    assert abs(e.b - 30) < 8, f"b off: {e.b}"
    print("  PASSED")


def test_noisy():
    """Ellipse + 1% random noise pixels."""
    gt = [(100.0, 100.0, 50.0, 25.0, 0.0)]
    edges = make_ellipse_edge_map(200, 200, gt, noise_fraction=0.01)
    t0 = time.time()
    result = detect_ellipses(edges, threshold=0.4)
    elapsed = time.time() - t0
    report("Noisy (1% random pixels)", gt, result, elapsed)
    assert len(result) >= 1, "Should detect the ellipse in noise"
    print("  PASSED")


def test_near_circle():
    """Nearly circular ellipse (a ~= b)."""
    gt = [(100.0, 100.0, 40.0, 38.0, 0.0)]
    edges = make_ellipse_edge_map(200, 200, gt)
    t0 = time.time()
    result = detect_ellipses(edges, threshold=0.4)
    elapsed = time.time() - t0
    report("Near-circle (a=40, b=38)", gt, result, elapsed)
    assert len(result) >= 1, "Should detect the near-circular ellipse"
    print("  PASSED")


def test_overlapping_ransac():
    """Two overlapping/intersecting ellipses resolved via RANSAC."""
    gt = [
        (100.0, 100.0, 50.0, 25.0, 0.0),
        (110.0, 100.0, 45.0, 20.0, np.radians(60)),
    ]
    edges = make_ellipse_edge_map(220, 220, gt)
    t0 = time.time()
    result = detect_ellipses(edges, threshold=0.3, ransac_iters=300)
    elapsed = time.time() - t0
    report("Overlapping ellipses with RANSAC", gt, result, elapsed)
    assert len(result) >= 2, f"Expected 2 ellipses, got {len(result)}"
    print("  PASSED")


def test_validation():
    """Input validation."""
    ok = np.zeros((100, 100), dtype=np.uint8)
    # Empty edge map -> empty list
    assert detect_ellipses(ok) == [], "Empty edge map should return []"

    # Bad dimensions
    try:
        detect_ellipses(np.zeros((5,), dtype=np.uint8))
        assert False, "Should have raised ValueError for 1-D input"
    except ValueError:
        pass

    # Bad threshold
    try:
        detect_ellipses(ok, threshold=0)
        assert False, "threshold=0 should raise"
    except ValueError:
        pass
    try:
        detect_ellipses(ok, threshold=1.5)
        assert False, "threshold=1.5 should raise"
    except ValueError:
        pass

    print("\n  Validation tests PASSED")


def test_gui_kwargs():
    """Verify the legacy GUI keyword arguments work."""
    gt = [(100.0, 100.0, 50.0, 30.0, 0.0)]
    edges = make_ellipse_edge_map(200, 200, gt)
    # The GUI passes min_axis, max_axis, vote_threshold
    result = detect_ellipses(edges, min_axis=10, max_axis=80, vote_threshold=0.3)
    assert len(result) >= 1, "GUI kwargs should work"
    print("\n  GUI kwargs test PASSED")


def test_threshold_sweep():
    """Show effect of different thresholds on a single ellipse."""
    gt = [(100.0, 100.0, 50.0, 25.0, 0.0)]
    edges = make_ellipse_edge_map(200, 200, gt)
    print(f"\n{'=' * 60}")
    print("THRESHOLD SWEEP (single ellipse, a=50, b=25)")
    for thr in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        result = detect_ellipses(edges, threshold=thr)
        status = "detected" if len(result) >= 1 else "MISSED"
        print(f"  threshold={thr:.1f} -> {len(result)} ellipse(s)  [{status}]")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

if __name__ == "__main__":
    test_validation()
    test_single_ellipse()
    test_rotated_ellipse()
    test_multiple_ellipses()
    test_partial_arc()
    test_noisy()
    test_near_circle()
    test_overlapping_ransac()
    test_gui_kwargs()
    test_threshold_sweep()

    print(f"\n{'=' * 60}")
    print("ALL TESTS PASSED")

    # -------------------------------------------------------------------
    # Tuning notes
    # -------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("TUNING NOTES")
    print("  - threshold in (0, 1]: fraction of the Ramanujan perimeter")
    print("    that must be covered by edge pixels within dist_tolerance.")
    print("  - Too low (< 0.3): false positives from noise or partial arcs")
    print("    that do not form real ellipses.")
    print("  - Too high (> 0.7): misses ellipses with gaps or partial occlusion.")
    print("  - Recommended range: 0.3 to 0.5 for typical Canny edges.")
    print("  - link_distance=2 bridges 1-pixel gaps from Canny thinning.")
    print("  - dist_tolerance=2.0 is a good default; raise for low-res images.")
    print("  - ransac_iters=200-300 helps split touching/overlapping ellipses.")
