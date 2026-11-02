"""Hough circle detection and ellipse detection module.

This module provides three feature detectors for circular and elliptical shapes:
1. Standard Circular Hough Transform (3-D accumulator sliding window, no gradients).
   Processes radii using a 3-slice sliding window (O(3*H*W) memory), evaluates local
   maxima directly on suprathreshold cells, and performs non-maximum suppression (NMS).
2. Gradient-based Circular Hough Transform (2-stage, 2-D accumulator).
   Exploits edge gradient vectors to collapse voting into an O(H*W) accumulator,
   followed by 1-D radial histogramming at candidate centers.
3. Ellipse Detection (connected components + least-squares direct fit).
   Groups 8-connected edge pixels via Chebyshev flood fill, fits conics using the
   Halir-Flusser / Fitzgibbon direct least-squares method under the exact ellipse
   constraint (4AC - B^2 = 1), verifies support against Ramanujan perimeter, and
   resolves overlapping/touching shapes via RANSAC.
"""
from __future__ import annotations

import numpy as np

from core.canny_lines import (
    correlate_1d,
    gaussian_blur,
    sobel_gradients,
)
from core.types import Circle, Ellipse


# ---------------------------------------------------------------------------
# Named Module Constants
# ---------------------------------------------------------------------------
_NMS_DIST_FACTOR: float = 0.5
_NMS_RADIUS_DIFF_FACTOR: float = 0.25
_NMS_RADIUS_DIFF_MIN: int = 2
_VOTE_CHUNK_POINTS: int = 2_000_000
_TOP_CANDIDATE_FACTOR: int = 20
_MIN_THETA_SAMPLES: int = 8
_OFFSET_K_FACTOR: int = 4

# Module-level cache for radius circle offsets: r -> (dx, dy)
_RADIUS_OFFSETS_CACHE: dict[int, tuple[np.ndarray, np.ndarray]] = {}


# ---------------------------------------------------------------------------
# Shared Helpers (prefixed _common_)
# ---------------------------------------------------------------------------

def _common_get_circle_offsets(r: int) -> tuple[np.ndarray, np.ndarray]:
    """Return precomputed unique (dx, dy) integer circle offsets for radius r.

    Uses 1-D integer encoding (dx * K + dy) for fast unique extraction
    and caches results per radius.
    """
    if r in _RADIUS_OFFSETS_CACHE:
        return _RADIUS_OFFSETS_CACHE[r]

    n_theta = max(int(np.ceil(2 * np.pi * r)), _MIN_THETA_SAMPLES)
    thetas = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    dx = np.round(r * np.cos(thetas)).astype(np.intp)
    dy = np.round(r * np.sin(thetas)).astype(np.intp)

    K = _OFFSET_K_FACTOR * r + 1
    encoded = dx * K + dy
    _, u_idx = np.unique(encoded, return_index=True)
    res = (dx[u_idx], dy[u_idx])
    _RADIUS_OFFSETS_CACHE[r] = res
    return res


def _common_accumulate_radius_slice(
    edge_cols: np.ndarray,
    edge_rows: np.ndarray,
    dx: np.ndarray,
    dy: np.ndarray,
    H: int,
    W: int,
) -> np.ndarray:
    """Compute 2-D vote accumulator slice (int32) for a single radius.

    Chunks edge pixels to bound peak memory to ~_VOTE_CHUNK_POINTS candidate
    points per chunk, accumulating with np.bincount.
    """
    n_edges = len(edge_cols)
    if n_edges == 0:
        return np.zeros((H, W), dtype=np.int32)

    n_offsets = len(dx)
    chunk_size = max(1, _VOTE_CHUNK_POINTS // max(n_offsets, 1))
    slice_votes = np.zeros((H, W), dtype=np.int32)

    for start in range(0, n_edges, chunk_size):
        end = min(start + chunk_size, n_edges)
        c_cols = edge_cols[start:end]
        c_rows = edge_rows[start:end]

        a = (c_cols[:, None] - dx[None, :]).ravel()
        b = (c_rows[:, None] - dy[None, :]).ravel()

        valid = (a >= 0) & (a < W) & (b >= 0) & (b < H)
        if not np.any(valid):
            continue

        flat_idx = b[valid] * W + a[valid]
        counts = np.bincount(flat_idx, minlength=H * W)[: H * W]
        slice_votes += counts.reshape(H, W).astype(np.int32)

    return slice_votes


def _common_neighborhood_max(arr: np.ndarray, radius: int = 1) -> np.ndarray:
    """Max over a (2r+1)x(2r+1) window, computed separably (2 passes, O(r) shifts)."""
    out = np.asarray(arr)
    for axis in (0, 1):
        n = out.shape[axis]
        pad = [(0, 0), (0, 0)]
        pad[axis] = (radius, radius)
        padded = np.pad(out, pad, mode="edge")
        sl = [slice(None), slice(None)]
        sl[axis] = slice(0, n)
        res = padded[tuple(sl)].copy()
        for k in range(1, 2 * radius + 1):
            sl[axis] = slice(k, k + n)
            np.maximum(res, padded[tuple(sl)], out=res)
        out = res
    return out


def _common_eval_slice_local_max(
    curr_slice: np.ndarray,
    prev_slice: np.ndarray | None,
    next_slice: np.ndarray | None,
    r: int,
    threshold: float,
    H: int,
    W: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Suprathreshold cells of curr_slice that are 26-neighbourhood maxima.

    Thresholds first, then tests neighbours only at surviving cells. Indices
    are clamped at the border; a clamped neighbour is always inside the true
    neighbourhood, so the result is identical to zero padding.
    """
    rows, cols = np.nonzero(curr_slice >= threshold * 2.0 * np.pi * r)
    if rows.size == 0:
        return None

    v = curr_slice[rows, cols]
    keep = np.ones(rows.size, dtype=bool)
    for plane in (curr_slice, prev_slice, next_slice):
        if plane is None:
            continue
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if plane is curr_slice and dy == 0 and dx == 0:
                    continue
                keep &= v >= plane[np.clip(rows + dy, 0, H - 1),
                                   np.clip(cols + dx, 0, W - 1)]
        if not keep.any():
            return None

    return cols[keep], rows[keep], np.full(int(keep.sum()), r, dtype=int), v[keep]


def _common_suppress_circles(
    cands: list[tuple[int, int, int, int | float]],
    min_center_distance_user: float | int | None,
    max_circles: int,
) -> tuple[list[Circle], list[int | float]]:
    """Greedy non-maximum suppression for circular candidates.

    Rejects a candidate when:
      dist < 0.5 * min(r, accepted_r) AND
      |r - accepted_r| < max(2, 0.25 * min(r, accepted_r)),
    and also when dist < min_center_distance_user if explicitly specified.

    Parameters
    ----------
    cands : list of (cx, cy, r, score)
        Candidate circles sorted by score descending.
    min_center_distance_user : float | int | None
        Explicit minimum distance between circle centers if requested by user.
        If None, this additional distance constraint is not applied.
    max_circles : int
        Maximum number of circles to retain.

    Returns
    -------
    tuple[list[Circle], list[int | float]]
        Accepted circles and their corresponding scores.
    """
    accepted_circles: list[Circle] = []
    accepted_scores: list[int | float] = []

    for item in cands:
        cx, cy, r, score = item
        cx_i = int(round(cx))
        cy_i = int(round(cy))
        r_i = int(round(r))

        suppress = False
        for ac in accepted_circles:
            dx = cx - ac.cx
            dy = cy - ac.cy
            dist = (dx * dx + dy * dy) ** 0.5

            min_r = min(r, ac.r)
            if (
                dist < _NMS_DIST_FACTOR * min_r
                and abs(r - ac.r) < max(_NMS_RADIUS_DIFF_MIN, _NMS_RADIUS_DIFF_FACTOR * min_r)
            ):
                suppress = True
                break

            if min_center_distance_user is not None and dist < min_center_distance_user:
                suppress = True
                break

        if not suppress:
            accepted_circles.append(Circle(cx=cx_i, cy=cy_i, r=r_i))
            accepted_scores.append(score)
            if len(accepted_circles) >= max_circles:
                break

    return accepted_circles, accepted_scores


# ---------------------------------------------------------------------------
# Circular Hough Transform
# ---------------------------------------------------------------------------

def hough_circles(
    edges: np.ndarray,
    r_min: int,
    r_max: int,
    threshold: float = 0.5,
    min_center_distance: int | None = None,
    max_circles: int = 50,
    return_scores: bool = False,
) -> list[Circle] | tuple[list[Circle], list[int]]:
    """Detect circles in an edge map via the standard Circular Hough Transform.

    Uses a 3-slice sliding 3-D accumulator (O(3*H*W) memory) and votes along
    full circles at every edge pixel – no gradient information is used.

    Parameters
    ----------
    edges : np.ndarray
        H×W binary edge map (any non-zero pixel is treated as an edge).
    r_min : int
        Minimum circle radius to search for (pixels).
    r_max : int
        Maximum circle radius to search for (pixels).
    threshold : float, optional
        Normalized threshold in (0, 1]. A candidate must collect at least
        ``threshold * 2π * r`` votes for its radius *r*. Default 0.5.
    min_center_distance : int | None, optional
        Minimum Euclidean distance between the centers of two accepted
        circles. If None, only relative overlap suppression is used.
    max_circles : int, optional
        Maximum number of circles to return. Default 50. Must be >= 1.
    return_scores : bool, optional
        If True, returns a tuple (circles, votes_list). If False, returns
        only circles. Default False.

    Returns
    -------
    list[Circle] or tuple[list[Circle], list[int]]
        Detected circles sorted by votes (descending), optionally with votes.

    Raises
    ------
    ValueError
        On invalid inputs (wrong shape, bad radius range, bad threshold …).
    """

    # ------------------------------------------------------------------
    # 1. Input validation
    # ------------------------------------------------------------------
    if not isinstance(edges, np.ndarray):
        raise ValueError("edges must be a numpy ndarray.")
    if edges.ndim != 2:
        raise ValueError(
            f"edges must be a 2-D array, got {edges.ndim}-D (shape {edges.shape})."
        )
    if edges.size == 0:
        return ([], []) if return_scores else []

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

    max_circles = int(max_circles)
    if max_circles < 1:
        raise ValueError(f"max_circles must be >= 1, got {max_circles}.")

    if min_center_distance is not None:
        min_center_distance = int(min_center_distance)
        if min_center_distance < 0:
            raise ValueError(
                f"min_center_distance must be >= 0, got {min_center_distance}."
            )

    # ------------------------------------------------------------------
    # 2. Extract edge pixel coordinates (row, col)
    # ------------------------------------------------------------------
    edge_rows, edge_cols = np.nonzero(edges)
    if edge_rows.size == 0:
        return ([], []) if return_scores else []

    H, W = edges.shape

    # ------------------------------------------------------------------
    # 3 & 4. Sliding 3-slice voting & local-maximum testing
    # ------------------------------------------------------------------
    all_cols: list[np.ndarray] = []
    all_rows: list[np.ndarray] = []
    all_radii: list[np.ndarray] = []
    all_votes: list[np.ndarray] = []

    curr_dx, curr_dy = _common_get_circle_offsets(r_min)
    curr_slice = _common_accumulate_radius_slice(
        edge_cols, edge_rows, curr_dx, curr_dy, H, W
    )
    prev_slice: np.ndarray | None = None

    for r_next in range(r_min + 1, r_max + 1):
        ndx, ndy = _common_get_circle_offsets(r_next)
        next_slice = _common_accumulate_radius_slice(
            edge_cols, edge_rows, ndx, ndy, H, W
        )

        res = _common_eval_slice_local_max(
            curr_slice, prev_slice, next_slice, r_next - 1, threshold, H, W
        )
        if res is not None:
            all_cols.append(res[0])
            all_rows.append(res[1])
            all_radii.append(res[2])
            all_votes.append(res[3])

        prev_slice = curr_slice
        curr_slice = next_slice
        next_slice = None

    # Finalize the last radius slice
    res = _common_eval_slice_local_max(
        curr_slice, prev_slice, None, r_max, threshold, H, W
    )
    if res is not None:
        all_cols.append(res[0])
        all_rows.append(res[1])
        all_radii.append(res[2])
        all_votes.append(res[3])

    if not all_cols:
        return ([], []) if return_scores else []

    cand_cols = np.concatenate(all_cols)
    cand_rows = np.concatenate(all_rows)
    cand_radii = np.concatenate(all_radii)
    cand_votes = np.concatenate(all_votes)

    # ------------------------------------------------------------------
    # 5. Top candidates selection via argpartition
    # ------------------------------------------------------------------
    top_k = _TOP_CANDIDATE_FACTOR * max_circles
    if len(cand_votes) > top_k:
        k_idx = np.argpartition(-cand_votes, top_k)[:top_k]
        sub_votes = cand_votes[k_idx]
        sub_cols = cand_cols[k_idx]
        sub_rows = cand_rows[k_idx]
        sub_radii = cand_radii[k_idx]
        order = np.argsort(-sub_votes)
        cands = [
            (int(sub_cols[i]), int(sub_rows[i]), int(sub_radii[i]), int(sub_votes[i]))
            for i in order
        ]
    else:
        order = np.argsort(-cand_votes)
        cands = [
            (int(cand_cols[i]), int(cand_rows[i]), int(cand_radii[i]), int(cand_votes[i]))
            for i in order
        ]

    # ------------------------------------------------------------------
    # 6. Duplicate removal via shared NMS
    # ------------------------------------------------------------------
    accepted, scores = _common_suppress_circles(
        cands, min_center_distance, max_circles
    )

    if return_scores:
        return accepted, [int(s) for s in scores]
    return accepted


# ---------------------------------------------------------------------------
# Ellipse Detection
# ---------------------------------------------------------------------------
# All internal helpers are prefixed _ell_ to avoid clashes.

def _ell_connected_components(
    xs: np.ndarray, ys: np.ndarray, link_distance: int, min_points: int
) -> list[np.ndarray]:
    """Group edge pixels into connected components using stack-based flood fill.

    Parameters
    ----------
    xs, ys : 1-D int arrays of edge pixel coordinates (col, row).
    link_distance : pixels within this Chebyshev distance are neighbors.
    min_points : discard components smaller than this.

    Returns
    -------
    List of np.ndarray, each holding global indices into xs/ys.
    Complexity: O(N) amortized.
    """
    n = len(xs)
    if n == 0:
        return []

    # Build a lookup: (row, col) -> global index for O(1) neighbor lookup
    coord_to_idx: dict[tuple[int, int], int] = {}
    for i in range(n):
        coord_to_idx[(int(ys[i]), int(xs[i]))] = i

    # Precompute neighbor offsets (Chebyshev ball of given radius, excl. center)
    offsets = []
    for dy in range(-link_distance, link_distance + 1):
        for dx in range(-link_distance, link_distance + 1):
            if dy == 0 and dx == 0:
                continue
            offsets.append((dy, dx))

    visited = np.zeros(n, dtype=np.bool_)
    components: list[np.ndarray] = []

    for start in range(n):
        if visited[start]:
            continue
        # Stack-based iterative flood fill
        stack = [start]
        visited[start] = True
        comp: list[int] = []
        while stack:
            idx = stack.pop()
            comp.append(idx)
            r0, c0 = int(ys[idx]), int(xs[idx])
            for dy, dx in offsets:
                nb = coord_to_idx.get((r0 + dy, c0 + dx))
                if nb is not None and not visited[nb]:
                    visited[nb] = True
                    stack.append(nb)
        if len(comp) >= min_points:
            components.append(np.array(comp, dtype=np.intp))

    return components


def _ell_fit_ellipse(px: np.ndarray, py: np.ndarray):
    """Fit an ellipse to 2-D points using the Halir-Flusser direct method.

    Minimizes algebraic distance A*x^2 + B*x*y + C*y^2 + D*x + E*y + F = 0
    under the constraint 4*A*C - B^2 = 1 (guarantees an ellipse).

    Parameters
    ----------
    px, py : 1-D float arrays of point coordinates.

    Returns
    -------
    (cx, cy, a, b, angle) or None on failure.
    a >= b, angle = rotation of the major axis in radians [-pi/2, pi/2).
    """
    n = len(px)
    if n < 6:
        return None

    # -- Normalize points for numerical stability --
    mx, my = px.mean(), py.mean()
    sx = px.std()
    sy = py.std()
    if sx < 1e-12 or sy < 1e-12:
        return None  # degenerate (collinear)
    xn = (px - mx) / sx
    yn = (py - my) / sy

    # -- Build design matrix columns --
    x2 = xn * xn
    xy = xn * yn
    y2 = yn * yn

    # D1 = [x^2, xy, y^2]  (quadratic part)
    # D2 = [x, y, 1]       (linear part)
    # Scatter matrices  S1 = D1^T D1,  S2 = D1^T D2,  S3 = D2^T D2
    # Each is 3x3, computed via dot-product sums over N points.

    ones = np.ones(n, dtype=np.float64)
    d1 = np.column_stack([x2, xy, y2])   # (N, 3)
    d2 = np.column_stack([xn, yn, ones]) # (N, 3)

    S1 = d1.T @ d1  # (3, 3)
    S2 = d1.T @ d2  # (3, 3)
    S3 = d2.T @ d2  # (3, 3)

    # Constraint matrix for the quadratic part: 4AC - B^2 = 1
    # In the split form the constraint is only on the first 3 coefficients.
    # C1 = [[0, 0, 2], [0, -1, 0], [2, 0, 0]]
    C1 = np.array([[0, 0, 2], [0, -1, 0], [2, 0, 0]], dtype=np.float64)

    try:
        S3_inv = np.linalg.inv(S3)
    except np.linalg.LinAlgError:
        return None

    # Reduced system: (S1 - S2 S3^{-1} S2^T) a1 = lambda C1 a1
    T = S3_inv @ S2.T               # (3, 3)
    M = S1 - S2 @ T                  # (3, 3)
    # Multiply by C1^{-1}
    try:
        C1_inv = np.linalg.inv(C1)
    except np.linalg.LinAlgError:
        return None
    Mprime = C1_inv @ M              # (3, 3)

    eigenvalues, eigenvectors = np.linalg.eig(Mprime)

    # Pick the eigenvector where 4*a1[0]*a1[2] - a1[1]^2 > 0
    # (the ellipse constraint)
    cond = np.zeros(3, dtype=np.float64)
    for i in range(3):
        a1 = eigenvectors[:, i].real
        cond[i] = 4.0 * a1[0] * a1[2] - a1[1] * a1[1]

    valid = np.where(cond > 0)[0]
    if len(valid) == 0:
        return None
    # Among valid, pick the one with the smallest positive eigenvalue
    best = valid[np.argmin(np.abs(eigenvalues[valid].real))]
    a1 = eigenvectors[:, best].real

    # Recover the linear part: a2 = -S3^{-1} S2^T a1
    a2 = -T @ a1

    # Full conic coefficients in normalized coordinates
    A, B, C = a1
    D, E, F = a2

    # -- De-normalize: transform back to original coordinates --
    # x_orig = xn * sx + mx  =>  xn = (x - mx) / sx
    # Substitute into A*xn^2 + B*xn*yn + C*yn^2 + D*xn + E*yn + F = 0
    Ao = A / (sx * sx)
    Bo = B / (sx * sy)
    Co = C / (sy * sy)
    Do = D / sx - 2.0 * A * mx / (sx * sx) - B * my / (sx * sy)
    Eo = E / sy - 2.0 * C * my / (sy * sy) - B * mx / (sx * sy)
    Fo = (A * mx * mx / (sx * sx) + B * mx * my / (sx * sy)
          + C * my * my / (sy * sy)
          - D * mx / sx - E * my / sy + F)

    return _ell_conic_to_params(np.array([Ao, Bo, Co, Do, Eo, Fo]))


def _ell_conic_to_params(c: np.ndarray):
    """Convert conic coefficients [A, B, C, D, E, F] to ellipse parameters.

    A*x^2 + B*x*y + C*y^2 + D*x + E*y + F = 0

    Returns (cx, cy, a, b, angle) or None if degenerate.
    a >= b, angle in [-pi/2, pi/2).
    """
    A, B, C, D, E, F = c

    # Ensure A > 0 (or C > 0 if A == 0) for positive eigenvalues
    if A < 0 or (abs(A) < 1e-12 and C < 0):
        A, B, C, D, E, F = -A, -B, -C, -D, -E, -F

    # Discriminant for a conic to be an ellipse: 4AC - B^2 > 0
    disc = 4.0 * A * C - B * B
    if disc <= 1e-14:
        return None

    # Center of the ellipse:
    # cx = (BE - 2CD) / (4AC - B^2),  cy = (BD - 2AE) / (4AC - B^2)
    cx = (B * E - 2.0 * C * D) / disc
    cy = (B * D - 2.0 * A * E) / disc

    # Build the 3x3 matrix form for the conic
    # M = [[A, B/2, D/2],
    #      [B/2, C, E/2],
    #      [D/2, E/2, F]]
    M33 = np.array([
        [A,      B / 2.0, D / 2.0],
        [B / 2.0, C,      E / 2.0],
        [D / 2.0, E / 2.0, F      ],
    ], dtype=np.float64)
    det_M33 = np.linalg.det(M33)
    if abs(det_M33) < 1e-30:
        return None

    # Semi-axes and orientation from eigendecomposition of the 2x2 quadratic block
    M22 = np.array([[A, B / 2.0], [B / 2.0, C]], dtype=np.float64)
    eigvals_22, eigvecs_22 = np.linalg.eigh(M22)
    lam1, lam2 = eigvals_22  # sorted ascending: lam1 <= lam2

    if lam1 <= 0 or lam2 <= 0:
        return None

    # Semi-axes: a_i = sqrt(-det(M33) / (lambda_i * det(M22)))
    det_M22 = A * C - (B / 2.0) ** 2  # == disc / 4
    if abs(det_M22) < 1e-30:
        return None

    val1 = -det_M33 / (lam1 * det_M22)
    val2 = -det_M33 / (lam2 * det_M22)
    if val1 <= 0 or val2 <= 0:
        return None

    axis1 = np.sqrt(val1)
    axis2 = np.sqrt(val2)

    # Since lam1 <= lam2, axis1 >= axis2: axis1 is semi-major, axis2 is semi-minor
    a = axis1
    b = axis2

    # Major axis direction is the eigenvector for the smaller eigenvalue lam1 (col 0)
    v_major = eigvecs_22[:, 0]
    angle = float(np.arctan2(v_major[1], v_major[0]))

    # Normalize angle to [-pi/2, pi/2)
    while angle >= np.pi / 2.0:
        angle -= np.pi
    while angle < -np.pi / 2.0:
        angle += np.pi

    return (cx, cy, a, b, angle)


def _ell_perimeter(a: float, b: float) -> float:
    """Ramanujan approximation for the perimeter of an ellipse."""
    h = ((a - b) / (a + b)) ** 2 if (a + b) > 0 else 0.0
    return np.pi * (a + b) * (1.0 + 3.0 * h / (10.0 + np.sqrt(4.0 - 3.0 * h)))


def _ell_point_distances(
    px: np.ndarray, py: np.ndarray,
    cx: float, cy: float, a: float, b: float, angle: float,
) -> np.ndarray:
    """Approximate distance of each point from the ellipse boundary.

    Uses the scaled-radial method: rotate points into the ellipse frame,
    compute r_scaled = sqrt((x'/a)^2 + (y'/b)^2), and approximate distance
    as |r_scaled - 1| * local_radius where local_radius is the ellipse
    radius at that angle.

    Returns an array of distances (pixels), one per input point.
    """
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    dx = px - cx
    dy = py - cy
    # Rotate into ellipse-aligned frame
    x_rot = dx * cos_a + dy * sin_a
    y_rot = -dx * sin_a + dy * cos_a

    # Scaled radial coordinate
    xn = x_rot / a if a > 0 else x_rot * 1e12
    yn = y_rot / b if b > 0 else y_rot * 1e12
    r_scaled = np.sqrt(xn * xn + yn * yn)
    r_scaled = np.maximum(r_scaled, 1e-12)  # avoid division by zero

    # Angle in the ellipse frame
    theta = np.arctan2(y_rot, x_rot)
    # Radius of the ellipse at this angle
    ct = np.cos(theta)
    st = np.sin(theta)
    denom = np.sqrt((ct / a) ** 2 + (st / b) ** 2)
    denom = np.maximum(denom, 1e-12)
    r_ellipse = 1.0 / denom

    # Approximate Euclidean distance
    dist = np.abs(r_scaled - 1.0) * r_ellipse
    return dist


def _ell_ransac_split(
    px: np.ndarray, py: np.ndarray,
    a_min: float, a_max: float,
    ransac_iters: int,
    dist_tolerance: float,
    min_points: int,
):
    """Try to split a group of points into two ellipses using RANSAC.

    Randomly samples 6 points, fits an ellipse, checks inlier count; repeats on
    the remaining points (up to 4 ellipses). Returns list of (cx, cy, a, b, angle, inlier_indices) for found ellipses,
    or empty list if nothing found.
    """
    n = len(px)
    if n < 2 * min_points or ransac_iters <= 0:
        return []

    best_fits: list[tuple] = []
    remaining_mask = np.ones(n, dtype=np.bool_)
    rng = np.random.RandomState(42)

    for _attempt in range(4):  # peel up to 4 ellipses per merged group
        if remaining_mask.sum() < min_points:
            break
        rem_idx = np.where(remaining_mask)[0]
        best_inliers = None
        best_params = None
        best_count = 0

        for _ in range(ransac_iters):
            if len(rem_idx) < 6:
                break
            sample = rng.choice(rem_idx, 6, replace=False)
            result = _ell_fit_ellipse(px[sample], py[sample])
            if result is None:
                continue
            cx, cy, a, b, angle = result
            if a < a_min or a > a_max or b < 1.0 or a / b > 20.0:
                continue
            # Count inliers among remaining points
            dists = _ell_point_distances(px[rem_idx], py[rem_idx],
                                         cx, cy, a, b, angle)
            inl = dists < dist_tolerance
            count = int(inl.sum())
            if count > best_count:
                best_count = count
                best_inliers = rem_idx[inl]
                best_params = (cx, cy, a, b, angle)

        if best_params is not None and best_count >= min_points:
            # Refine fit on all inliers
            refit = _ell_fit_ellipse(px[best_inliers], py[best_inliers])
            if refit is not None:
                rcx, rcy, ra, rb, rang = refit
                if a_min <= ra <= a_max and rb >= 1.0 and ra / rb <= 20.0:
                    best_params = refit
                    dists = _ell_point_distances(px[rem_idx], py[rem_idx],
                                                 rcx, rcy, ra, rb, rang)
                    best_inliers = rem_idx[dists < dist_tolerance]
            best_fits.append((*best_params, best_inliers))
            remaining_mask[best_inliers] = False
        else:
            break

    return best_fits


def detect_ellipses(
    edges: np.ndarray,
    a_min: float = 5,
    a_max: float | None = None,
    threshold: float = 0.5,
    max_ellipses: int = 50,
    link_distance: int = 1,
    dist_tolerance: float = 2.0,
    min_points: int = 20,
    max_overlap: float = 0.5,
    ransac_iters: int = 300,
    **kwargs,
) -> list[Ellipse]:
    """Detect ellipses in a binary edge map.

    Uses connected-component grouping followed by direct least-squares
    ellipse fitting (Halir-Flusser / Fitzgibbon method).
    No Hough accumulator -- works on pixel groups, O(N) time and memory.

    Parameters
    ----------
    edges : 2-D binary edge map (any non-zero pixel is an edge).
    a_min : minimum allowed semi-major axis (pixels).
    a_max : maximum allowed semi-major axis (None = half of larger image side).
    threshold : normalized (0,1]. Accept ellipse only if support pixels
                >= threshold * Ramanujan_perimeter.
    max_ellipses : maximum number of ellipses to return.
    link_distance : Chebyshev distance for pixel connectivity (1 = 8-conn).
    dist_tolerance : max distance (px) from fitted ellipse to count as support.
    min_points : ignore components with fewer pixels.
    max_overlap : reject candidate if this fraction of its support is already used.
    ransac_iters : if > 0, try RANSAC split on failed components.
    **kwargs : legacy GUI parameters (min_axis, max_axis, vote_threshold).

    Returns
    -------
    list[Ellipse] sorted by number of support pixels (descending).

    Known limitation
    ----------------
    Touching or overlapping ellipses merge into one connected component.
    Set ransac_iters > 0 to attempt splitting such groups.
    """
    # -- Map legacy GUI keyword arguments --
    if "min_axis" in kwargs:
        a_min = float(kwargs["min_axis"])
    if "max_axis" in kwargs:
        a_max_val = kwargs["max_axis"]
        if a_max_val is not None:
            a_max = float(a_max_val)
    if "vote_threshold" in kwargs:
        threshold = float(kwargs["vote_threshold"])

    # ---------------------------------------------------------------
    # Step 1: Input validation
    # ---------------------------------------------------------------
    if not isinstance(edges, np.ndarray):
        raise ValueError("edges must be a numpy ndarray")
    if edges.ndim != 2:
        raise ValueError(f"edges must be 2-D, got {edges.ndim}-D")
    H, W = edges.shape
    if H == 0 or W == 0:
        return []

    if a_max is None:
        a_max = max(H, W) / 2.0
    if a_min <= 0:
        raise ValueError(f"a_min must be > 0, got {a_min}")
    if a_max < a_min:
        raise ValueError(f"a_max ({a_max}) must be >= a_min ({a_min})")
    if not (0 < threshold <= 1):
        raise ValueError(f"threshold must be in (0, 1], got {threshold}")
    if link_distance < 1:
        raise ValueError(f"link_distance must be >= 1, got {link_distance}")
    if not (0 <= max_overlap <= 1):
        raise ValueError(f"max_overlap must be in [0, 1], got {max_overlap}")

    # ---------------------------------------------------------------
    # Step 2: Extract edge pixel coordinates
    # ---------------------------------------------------------------
    edge_map = edges > 0  # works for 0/255 and 0/1
    ys_all, xs_all = np.nonzero(edge_map)
    N = len(xs_all)
    if N == 0:
        return []

    xs_all = xs_all.astype(np.float64)
    ys_all = ys_all.astype(np.float64)

    # ---------------------------------------------------------------
    # Step 3: Connected components (stack-based flood fill)
    # ---------------------------------------------------------------
    components = _ell_connected_components(
        xs_all.astype(np.intp), ys_all.astype(np.intp),
        link_distance, min_points
    )

    # ---------------------------------------------------------------
    # Step 4 & 5: Fit ellipse to each component + verify
    # ---------------------------------------------------------------
    # Each candidate: (support_count, cx, cy, a, b, angle, support_indices)
    candidates: list[tuple[int, float, float, float, float, float, np.ndarray]] = []

    for comp_indices in components:
        px = xs_all[comp_indices]
        py = ys_all[comp_indices]

        result = _ell_fit_ellipse(px, py)
        if result is None:
            # Step 6: If RANSAC enabled, try splitting
            if ransac_iters > 0:
                splits = _ell_ransac_split(
                    px, py, a_min, a_max, ransac_iters,
                    dist_tolerance, min_points
                )
                for cx, cy, a, b, angle, inl_local in splits:
                    # inl_local indexes into px/py; map back to global
                    global_inl = comp_indices[inl_local]
                    perim = _ell_perimeter(a, b)
                    req = threshold * perim
                    if len(global_inl) >= req:
                        candidates.append(
                            (len(global_inl), cx, cy, a, b, angle, global_inl)
                        )
            continue

        cx, cy, a, b, angle = result

        # Reject if axes are out of range or aspect ratio is extreme
        if a < a_min or a > a_max or b < 1.0 or a / b > 20.0:
            # Try RANSAC split if enabled
            if ransac_iters > 0:
                splits = _ell_ransac_split(
                    px, py, a_min, a_max, ransac_iters,
                    dist_tolerance, min_points
                )
                for scx, scy, sa, sb, sang, inl_local in splits:
                    global_inl = comp_indices[inl_local]
                    perim = _ell_perimeter(sa, sb)
                    req = threshold * perim
                    if len(global_inl) >= req:
                        candidates.append(
                            (len(global_inl), scx, scy, sa, sb, sang, global_inl)
                        )
            continue

        # Compute support: points within dist_tolerance of the fitted ellipse
        dists = _ell_point_distances(px, py, cx, cy, a, b, angle)
        support_mask = dists < dist_tolerance
        support_count = int(support_mask.sum())

        # Check threshold: required votes = threshold * perimeter
        perim = _ell_perimeter(a, b)
        required = threshold * perim
        if support_count < required or (ransac_iters > 0 and len(px) >= 2 * min_points and support_count < 0.7 * len(px)):
            # Try RANSAC split if enabled
            if ransac_iters > 0:
                splits = _ell_ransac_split(
                    px, py, a_min, a_max, ransac_iters,
                    dist_tolerance, min_points
                )
                if splits:
                    for scx, scy, sa, sb, sang, inl_local in splits:
                        global_inl = comp_indices[inl_local]
                        perim2 = _ell_perimeter(sa, sb)
                        req2 = threshold * perim2
                        if len(global_inl) >= req2:
                            candidates.append(
                                (len(global_inl), scx, scy, sa, sb, sang, global_inl)
                            )
                    continue
            if support_count < required:
                continue

        # Re-fit on just the support pixels for a tighter estimate
        sup_idx = np.where(support_mask)[0]
        refit = _ell_fit_ellipse(px[sup_idx], py[sup_idx])
        if refit is not None:
            cx2, cy2, a2, b2, angle2 = refit
            if a_min <= a2 <= a_max and b2 >= 1.0 and a2 / b2 <= 20.0:
                # Re-check support with the refined ellipse
                dists2 = _ell_point_distances(px, py, cx2, cy2, a2, b2, angle2)
                sup_mask2 = dists2 < dist_tolerance
                sc2 = int(sup_mask2.sum())
                perim2 = _ell_perimeter(a2, b2)
                if sc2 >= threshold * perim2:
                    cx, cy, a, b, angle = cx2, cy2, a2, b2, angle2
                    support_mask = sup_mask2
                    support_count = sc2

        global_support = comp_indices[np.where(support_mask)[0]]
        candidates.append((support_count, cx, cy, a, b, angle, global_support))

    if not candidates:
        return []

    # ---------------------------------------------------------------
    # Step 7: Anti-redundancy (used-pixel mask) + sort by votes desc
    # ---------------------------------------------------------------
    candidates.sort(key=lambda t: t[0], reverse=True)

    used = np.zeros(N, dtype=np.bool_)
    results: list[Ellipse] = []

    for support_count, cx, cy, a, b, angle, sup_indices in candidates:
        if len(results) >= max_ellipses:
            break
        # Check overlap: fraction of this candidate's support already used
        already_used = used[sup_indices].sum()
        if len(sup_indices) > 0 and already_used / len(sup_indices) > max_overlap:
            continue
        # Near-identical parameters to an accepted ellipse -> duplicate
        if any(abs(cx - r.cx) < 0.15 * r.b and abs(cy - r.cy) < 0.15 * r.b
               and abs(a - r.a) < 0.15 * r.a and abs(b - r.b) < 0.15 * r.b
               for r in results):
            continue
        # Accept this ellipse
        results.append(Ellipse(cx=cx, cy=cy, a=a, b=b, angle=angle))
        used[sup_indices] = True

    # ---------------------------------------------------------------
    # Step 8: Return sorted by support count descending (already sorted)
    # ---------------------------------------------------------------
    return results


# ===========================================================================
# Gradient-based Circular Hough Transform  (2-stage, 2-D accumulator)
# ===========================================================================
# Blur / Sobel / box filter / neighbourhood-max all come from core.canny_lines,
# so nothing image-processing related is re-implemented here.

_GRAD_PEAK_FACTOR: int = 8          # centre peaks kept = factor * max_circles
_GRAD_COS_TOL: float = float(np.cos(np.radians(30.0)))
_BOX3 = np.ones(3, dtype=np.float64)


def _grad_edge_directions(edges, ys, xs, gx, gy):
    """Gradient (gx, gy, magnitude) at each edge pixel, vectorised.

    On a thin edge map the gradient *at* the ridge is ~0, so the strongest
    3x3 neighbour is sampled instead.  The sign is irrelevant because every
    consumer uses |cos|.
    """
    if gx is None or gy is None:
        smooth = gaussian_blur((edges > 0).astype(np.float64), 1.5)
        gx, gy, _, _ = sobel_gradients(smooth)
    gx = np.asarray(gx, dtype=np.float64)
    gy = np.asarray(gy, dtype=np.float64)
    gxp, gyp, magp = (np.pad(a, 1) for a in (gx, gy, np.hypot(gx, gy)))

    off = np.array([(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)])
    rr = ys[:, None] + 1 + off[:, 0]
    cc = xs[:, None] + 1 + off[:, 1]
    best = magp[rr, cc].argmax(axis=1)
    rows = np.arange(ys.size)
    rk, ck = rr[rows, best], cc[rows, best]
    return gxp[rk, ck], gyp[rk, ck], magp[rk, ck]


def _grad_vote_centers(xs, ys, ux, uy, radii, H, W):
    """Stage 1: every edge pixel votes both ways along its gradient."""
    t = np.concatenate([radii, -radii])
    chunk = max(1, _VOTE_CHUNK_POINTS // t.size)
    acc = np.zeros(H * W, dtype=np.int32)
    for s in range(0, xs.size, chunk):
        sl = slice(s, s + chunk)
        a = np.rint(xs[sl, None] + t * ux[sl, None]).astype(np.intp)
        b = np.rint(ys[sl, None] + t * uy[sl, None]).astype(np.intp)
        ok = (a >= 0) & (a < W) & (b >= 0) & (b < H)
        acc += np.bincount(b[ok] * W + a[ok], minlength=H * W).astype(np.int32)
    return acc.reshape(H, W)


def _grad_radius_candidates(xs, ys, ux, uy, peaks, r_min, r_max, r_tol, loose):
    """Stage 2: per centre, radial histogram (+-1 px tolerant) -> radius peaks."""
    radii = np.arange(r_min, r_max + 1)
    need = loose * 2.0 * np.pi * radii
    out = []
    for cx, cy in peaks:
        dx, dy = xs - cx, ys - cy
        dist = np.hypot(dx, dy)
        sel = (dist >= r_min - r_tol) & (dist <= r_max + r_tol)
        sel &= np.abs(ux * dx + uy * dy) >= _GRAD_COS_TOL * dist
        if not sel.any():
            continue
        bins = np.clip(np.rint(dist[sel]).astype(np.intp) - r_min, 0, radii.size - 1)
        p = np.pad(np.bincount(bins, minlength=radii.size), 1)
        h = p[:-2] + p[1:-1] + p[2:]                 # 3-bin sum = +-1 px tolerance
        ph = np.pad(h, 1)
        pk = (h >= ph[:-2]) & (h >= ph[2:]) & (h >= need)
        out.extend((cx, cy, int(r), int(v)) for r, v in zip(radii[pk], h[pk]))
    return out


def _grad_support(xs, ys, ux, uy, free, cx, cy, r, r_tol, n_sectors=12):
    """Free edge pixels lying on circle (cx, cy, r) with a radial gradient.

    Returns (indices, angular coverage in [0, 1]).
    """
    dx, dy = xs - cx, ys - cy
    dist = np.hypot(dx, dy)
    idx = np.flatnonzero(free & (np.abs(dist - r) <= r_tol))
    if idx.size:
        idx = idx[np.abs(ux[idx] * dx[idx] + uy[idx] * dy[idx])
                  >= _GRAD_COS_TOL * dist[idx]]
    if idx.size == 0:
        return idx, 0.0
    sector = ((np.arctan2(dy[idx], dx[idx]) + np.pi)
              * (n_sectors / (2.0 * np.pi))).astype(np.intp) % n_sectors
    return idx, np.unique(sector).size / n_sectors


def _grad_fit_circle(x, y):
    """Kasa algebraic circle fit (centred for conditioning) -> (cx, cy, r) or None."""
    if x.size < 3:
        return None
    mx, my = x.mean(), y.mean()
    u, v = x - mx, y - my
    sol, *_ = np.linalg.lstsq(
        np.column_stack([2 * u, 2 * v, np.ones_like(u)]), u * u + v * v, rcond=None
    )
    r2 = sol[2] + sol[0] ** 2 + sol[1] ** 2
    if r2 <= 0:
        return None
    return mx + sol[0], my + sol[1], float(np.sqrt(r2))


def hough_circles_gradient(
    edges: np.ndarray,
    r_min: int,
    r_max: int,
    threshold: float = 0.5,
    min_center_distance: int | None = None,
    max_circles: int = 50,
    gx: np.ndarray | None = None,
    gy: np.ndarray | None = None,
    min_gradient: float = 1e-3,
    center_threshold: float | None = None,
    r_tol: float = 1.0,
    min_coverage: float | None = None,
) -> list[Circle]:
    """Gradient-based Circular Hough Transform with verify-and-peel.

    1. Centre voting: one H x W accumulator, O(N_edge * N_radii).
    2. Radius candidates: per centre peak, a 1-D radial histogram.
    3. Verify-and-peel: candidates are refined with a circle fit, re-scored
       on a +-r_tol ring (count AND angular coverage), and the pixels of each
       accepted circle are removed so they cannot support another one.
       This replaces distance-based NMS, so overlapping circles survive and
       duplicates die on their own.

    Parameters
    ----------
    threshold : accept if support >= threshold * 2*pi*r.
    min_center_distance : optional; if given, centre peaks are separated by
        at least ~this many pixels (default: 5x5 window).
    r_tol : ring half-width in pixels used when counting support.
    min_coverage : fraction of the circle's angular sectors that must hold
        support (default 1.2 * threshold, i.e. 0.6). Raise to ~0.75 when ellipses
        are present (their curved ends otherwise look like small circles);
        lower it for heavily occluded circles.
    """
    if not isinstance(edges, np.ndarray):
        raise ValueError("edges must be a numpy ndarray.")
    if edges.ndim != 2:
        raise ValueError(f"edges must be a 2-D array, got {edges.ndim}-D (shape {edges.shape}).")
    if edges.size == 0:
        return []
    r_min, r_max, max_circles = int(r_min), int(r_max), int(max_circles)
    if r_min < 1:
        raise ValueError(f"r_min must be >= 1, got {r_min}.")
    if r_max < r_min:
        raise ValueError(f"r_max ({r_max}) must be >= r_min ({r_min}).")
    if not (0 < threshold <= 1):
        raise ValueError(f"threshold must be in (0, 1], got {threshold}.")
    if max_circles < 1:
        raise ValueError(f"max_circles must be >= 1, got {max_circles}.")
    for name, g in (("gx", gx), ("gy", gy)):
        if g is not None and g.shape != edges.shape:
            raise ValueError(f"{name} shape {g.shape} does not match edges shape {edges.shape}.")

    H, W = edges.shape
    ys, xs = np.nonzero(edges)
    if ys.size == 0:
        return []

    gx_e, gy_e, mag = _grad_edge_directions(edges, ys, xs, gx, gy)
    strong = mag >= min_gradient
    if not strong.any():
        return []
    xs = xs[strong].astype(np.float64)
    ys = ys[strong].astype(np.float64)
    ux, uy = gx_e[strong] / mag[strong], gy_e[strong] / mag[strong]

    # ---- Stage 1: centre accumulator (3x3 box via the shared 1-D correlator)
    acc = _grad_vote_centers(xs, ys, ux, uy,
                             np.arange(r_min, r_max + 1, dtype=np.float64), H, W)
    acc = correlate_1d(correlate_1d(acc, _BOX3, axis=0), _BOX3, axis=1)

    if center_threshold is None:
        center_threshold = 0.5 * threshold * 2.0 * np.pi * r_min
    win = 2 if min_center_distance is None else max(1, int(min_center_distance) // 2)
    py, px = np.nonzero((acc >= center_threshold) & (acc >= _common_neighborhood_max(acc, win)))
    if py.size == 0:
        return []
    votes = acc[py, px]
    keep = np.argsort(-votes)[: _GRAD_PEAK_FACTOR * max_circles]
    peaks = list(zip(px[keep].tolist(), py[keep].tolist()))

    # ---- Stage 2: radius candidates
    cands = _grad_radius_candidates(xs, ys, ux, uy, peaks, r_min, r_max, r_tol,
                                    loose=0.5 * threshold)
    if not cands:
        return []
    cands.sort(key=lambda c: c[3], reverse=True)

    # ---- Stage 3: verify-and-peel
    min_cov = min(1.0, 1.2 * threshold) if min_coverage is None else min_coverage
    free = np.ones(xs.size, dtype=bool)
    accepted: list[Circle] = []
    for cx, cy, r, _ in cands:
        # loose gather -> refit (fixes +-2 px centre error) -> strict re-score
        idx, _cov = _grad_support(xs, ys, ux, uy, free, cx, cy, r, r_tol + 2.0)
        fit = _grad_fit_circle(xs[idx], ys[idx])
        if fit is None or not (r_min <= fit[2] <= r_max):
            continue
        cx, cy, r = fit
        idx, cov = _grad_support(xs, ys, ux, uy, free, cx, cy, r, r_tol)
        if idx.size < threshold * 2.0 * np.pi * r or cov < min_cov:
            continue
        accepted.append(Circle(cx=int(round(cx)), cy=int(round(cy)), r=int(round(r))))
        free[idx] = False
        if len(accepted) >= max_circles:
            break
    return accepted


# ===========================================================================
# Dispatcher
# ===========================================================================

def detect_circles(
    edges: np.ndarray,
    r_min: int,
    r_max: int,
    method: str = "standard",
    **kwargs,
) -> list[Circle] | tuple[list[Circle], list[int]]:
    """Detect circles using the chosen method.

    Parameters
    ----------
    edges : np.ndarray
        H x W binary edge map.
    r_min, r_max : int
        Radius search range.
    method : str
        ``"standard"`` for the sliding 3-D accumulator Hough transform, or
        ``"gradient"`` for the faster gradient-based 2-stage version.
    **kwargs
        Forwarded to the chosen implementation (e.g. threshold, max_circles,
        return_scores for standard).

    Returns
    -------
    list[Circle] or tuple[list[Circle], list[int]]

    Raises
    ------
    ValueError
        If *method* is not recognised.
    """
    if method == "standard":
        return hough_circles(edges, r_min, r_max, **kwargs)
    if method == "gradient":
        return hough_circles_gradient(edges, r_min, r_max, **kwargs)
    raise ValueError(
        f"Unknown circle detection method {method!r}. "
        f"Use 'standard' or 'gradient'."
    )