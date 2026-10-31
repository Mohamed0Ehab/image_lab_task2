
"""Optimized, from-scratch Canny edge detection and Hough line transform.

Implementation uses NumPy only. No OpenCV or image-processing library calls.

Canny:
    1. Separable Gaussian smoothing
    2. Separable Sobel gradients
    3. Vectorized non-maximum suppression
    4. Double thresholding
    5. Edge tracking by hysteresis

Hough:
    1. Chunked, vectorized voting
    2. Accumulator peak extraction
    3. Local maximum filtering
    4. Duplicate-line suppression
"""

from __future__ import annotations

import numpy as np

from core.types import Line


# ============================================================================
# Validation
# ============================================================================

def _validate_grayscale(image: np.ndarray, name: str = "image") -> None:
    if not isinstance(image, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")

    if image.ndim != 2:
        raise ValueError(f"{name} must be a 2D grayscale image")

    if image.size == 0:
        raise ValueError(f"{name} cannot be empty")


# ============================================================================
# Gaussian smoothing
# ============================================================================

def gaussian_kernel_1d(sigma: float) -> np.ndarray:
    """Generate a normalized 1D Gaussian kernel."""

    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be greater than zero")

    radius = int(np.ceil(3.0 * sigma))
    coordinates = np.arange(-radius, radius + 1, dtype=np.float64)

    kernel = np.exp(-(coordinates ** 2) / (2.0 * sigma ** 2))
    kernel /= kernel.sum()

    return kernel


def correlate_1d(
    image: np.ndarray,
    kernel: np.ndarray,
    axis: int,
) -> np.ndarray:
    """Apply a 1D correlation along one axis using reflection padding.

    The implementation avoids Python loops over image pixels. It only loops
    over kernel coefficients, applying each coefficient to an entire image
    slice at once.
    """

    if axis not in (0, 1):
        raise ValueError("axis must be 0 or 1")

    image = np.asarray(image, dtype=np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)

    if kernel.ndim != 1 or kernel.size % 2 == 0:
        raise ValueError("kernel must be a 1D array with odd length")

    radius = kernel.size // 2

    if radius == 0:
        return image * kernel[0]

    height, width = image.shape

    if axis == 1:
        padded = np.pad(
            image,
            ((0, 0), (radius, radius)),
            mode="reflect",
        )

        output = np.zeros_like(image)

        for k, weight in enumerate(kernel):
            if weight != 0:
                output += weight * padded[:, k:k + width]

    else:
        padded = np.pad(
            image,
            ((radius, radius), (0, 0)),
            mode="reflect",
        )

        output = np.zeros_like(image)

        for k, weight in enumerate(kernel):
            if weight != 0:
                output += weight * padded[k:k + height, :]

    return output


def gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """Apply separable Gaussian smoothing."""

    kernel = gaussian_kernel_1d(sigma)

    horizontal = correlate_1d(image, kernel, axis=1)
    blurred = correlate_1d(horizontal, kernel, axis=0)

    return blurred


# Keep the original function name available for compatibility.
def gaussian_kernel(sigma: float) -> np.ndarray:
    """Generate a normalized 2D Gaussian kernel."""

    kernel_1d = gaussian_kernel_1d(sigma)
    return np.outer(kernel_1d, kernel_1d)


def convolve(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """General 2D correlation for compatibility with older code.

    This function is not used in the optimized Canny pipeline.
    """

    image = np.asarray(image, dtype=np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)

    if image.ndim != 2 or kernel.ndim != 2:
        raise ValueError("image and kernel must both be 2D")

    if kernel.shape[0] % 2 == 0 or kernel.shape[1] % 2 == 0:
        raise ValueError("kernel dimensions must be odd")

    pad_y = kernel.shape[0] // 2
    pad_x = kernel.shape[1] // 2

    padded = np.pad(
        image,
        ((pad_y, pad_y), (pad_x, pad_x)),
        mode="reflect",
    )

    height, width = image.shape
    output = np.zeros_like(image)

    for ky in range(kernel.shape[0]):
        for kx in range(kernel.shape[1]):
            weight = kernel[ky, kx]

            if weight != 0:
                output += weight * padded[
                    ky:ky + height,
                    kx:kx + width,
                ]

    return output


# ============================================================================
# Sobel gradients
# ============================================================================

def sobel_gradients(
    image: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate Sobel gradients using separable filters.

    Returns:
        gx: Horizontal gradient.
        gy: Vertical gradient.
        magnitude: Gradient magnitude.
        direction: Gradient direction in degrees, in [-180, 180].
    """

    image = np.asarray(image, dtype=np.float64)

    derivative = np.array([-1.0, 0.0, 1.0])
    smoothing = np.array([1.0, 2.0, 1.0])

    # Horizontal derivative, vertical smoothing.
    gx_horizontal = correlate_1d(image, derivative, axis=1)
    gx = correlate_1d(gx_horizontal, smoothing, axis=0)

    # Horizontal smoothing, vertical derivative.
    gy_horizontal = correlate_1d(image, smoothing, axis=1)
    gy = correlate_1d(gy_horizontal, derivative, axis=0)

    magnitude = np.hypot(gx, gy)
    direction = np.rad2deg(np.arctan2(gy, gx))

    return gx, gy, magnitude, direction


# ============================================================================
# Non-maximum suppression
# ============================================================================

def non_maximum_suppression(
    magnitude: np.ndarray,
    direction: np.ndarray,
) -> np.ndarray:
    """Thin edges using vectorized directional comparisons."""

    if magnitude.shape != direction.shape:
        raise ValueError("magnitude and direction must have the same shape")

    if magnitude.ndim != 2:
        raise ValueError("magnitude and direction must be 2D")

    height, width = magnitude.shape

    output = np.zeros_like(magnitude, dtype=np.float64)

    if height < 3 or width < 3:
        return output

    angle = direction % 180.0

    # Only interior pixels are considered.
    center = magnitude[1:-1, 1:-1]

    left = magnitude[1:-1, :-2]
    right = magnitude[1:-1, 2:]

    top = magnitude[:-2, 1:-1]
    bottom = magnitude[2:, 1:-1]

    top_left = magnitude[:-2, :-2]
    top_right = magnitude[:-2, 2:]

    bottom_left = magnitude[2:, :-2]
    bottom_right = magnitude[2:, 2:]

    angle = angle[1:-1, 1:-1]

    # Four quantized gradient directions.
    horizontal = (angle < 22.5) | (angle >= 157.5)

    diagonal_45 = (angle >= 22.5) & (angle < 67.5)

    vertical = (angle >= 67.5) & (angle < 112.5)

    diagonal_135 = (angle >= 112.5) & (angle < 157.5)

    # Compare along the gradient direction.
    # With image y increasing downward:
    #   45 degrees  -> top-left and bottom-right
    #   135 degrees -> top-right and bottom-left
    keep = (
        (
            horizontal
            & (center >= left)
            & (center >= right)
        )
        | (
            diagonal_45
            & (center >= top_left)
            & (center >= bottom_right)
        )
        | (
            vertical
            & (center >= top)
            & (center >= bottom)
        )
        | (
            diagonal_135
            & (center >= top_right)
            & (center >= bottom_left)
        )
    )

    output[1:-1, 1:-1] = np.where(keep, center, 0.0)

    return output


# ============================================================================
# Double threshold
# ============================================================================

def double_threshold(
    image: np.ndarray,
    low_t: float,
    high_t: float,
) -> np.ndarray:
    """Classify pixels as non-edge (0), weak (1), or strong (2)."""

    if low_t < 0 or high_t <= 0 or low_t > high_t:
        raise ValueError("Require 0 <= low_t <= high_t and high_t > 0")

    result = np.zeros(image.shape, dtype=np.uint8)

    strong = image >= high_t
    weak = (image >= low_t) & (image < high_t) & (image > 0)

    result[weak] = 1
    result[strong] = 2

    return result


# ============================================================================
# Hysteresis
# ============================================================================

def hysteresis(image: np.ndarray) -> np.ndarray:
    """Keep strong pixels and weak pixels connected to strong pixels.

    Uses a stack of flattened pixel indices. Each weak pixel is promoted
    at most once, avoiding repeated processing.
    """

    if image.ndim != 2:
        raise ValueError("image must be 2D")

    height, width = image.shape

    strong = image == 2
    weak = image == 1

    result = np.zeros((height, width), dtype=np.uint8)
    result[strong] = 255

    # Flattened indices use less memory than storing (y, x) tuples.
    stack = np.flatnonzero(strong).tolist()

    while stack:
        index = stack.pop()

        y, x = divmod(index, width)

        # Visit the 8-connected neighborhood.
        y_start = max(0, y - 1)
        y_end = min(height, y + 2)

        x_start = max(0, x - 1)
        x_end = min(width, x + 2)

        neighborhood = weak[y_start:y_end, x_start:x_end]

        if not neighborhood.any():
            continue

        # Find weak pixels in this neighborhood.
        local_y, local_x = np.nonzero(neighborhood)

        local_y += y_start
        local_x += x_start

        # Promote them immediately so they cannot be queued twice.
        weak[local_y, local_x] = False
        result[local_y, local_x] = 255

        stack.extend(
            (local_y * width + local_x).tolist()
        )

    return result


# ============================================================================
# Complete Canny pipeline
# ============================================================================

def canny(
    gray: np.ndarray,
    sigma: float,
    low_t: float,
    high_t: float,
) -> np.ndarray:
    """Perform Canny edge detection on a grayscale image.

    Input:
        2D grayscale NumPy array.

    Output:
        uint8 binary edge image containing 0 and 255.
    """

    _validate_grayscale(gray, "gray")

    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be greater than zero")

    if (
        not np.isfinite(low_t)
        or not np.isfinite(high_t)
        or low_t < 0
        or high_t <= 0
        or low_t > high_t
    ):
        raise ValueError("Require 0 <= low_t <= high_t and high_t > 0")

    # Use float64 for numerical stability.
    gray = gray.astype(np.float64, copy=False)

    # 1. Gaussian smoothing.
    blurred = gaussian_blur(gray, sigma)

    # 2. Gradient calculation.
    _, _, magnitude, direction = sobel_gradients(blurred)

    # 3. Non-maximum suppression.
    thinned = non_maximum_suppression(magnitude, direction)

    # 4. Double threshold.
    classified = double_threshold(thinned, low_t, high_t)

    # 5. Edge tracking by hysteresis.
    return hysteresis(classified)


# ============================================================================
# Hough line transform
# ============================================================================

def hough_lines(
    edges: np.ndarray,
    threshold: int,
    theta_step: int = 1,
    min_rho_distance: int = 4,
    min_theta_distance: int = 3,
    chunk_size: int = 4096,
) -> list[Line]:
    """Detect straight lines from a binary edge image.

    Args:
        edges:
            2D edge image. Nonzero pixels are treated as edge pixels.

        threshold:
            Minimum number of votes required for a line.

        theta_step:
            Angular resolution in degrees.

        min_rho_distance:
            Minimum rho separation between returned lines.

        min_theta_distance:
            Minimum angular separation between returned lines.

        chunk_size:
            Number of edge pixels processed per voting batch.

    Returns:
        List of Line(rho, theta) objects. Theta is in radians.
    """

    _validate_grayscale(edges, "edges")

    if threshold < 1:
        raise ValueError("threshold must be at least 1")

    if theta_step < 1 or theta_step > 45:
        raise ValueError("theta_step must be between 1 and 45 degrees")

    if min_rho_distance < 0:
        raise ValueError("min_rho_distance cannot be negative")

    if min_theta_distance < 0:
        raise ValueError("min_theta_distance cannot be negative")

    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")

    height, width = edges.shape

    # Extract only edge coordinates.
    ys, xs = np.nonzero(edges)

    if xs.size == 0:
        return []

    xs = xs.astype(np.float64, copy=False)
    ys = ys.astype(np.float64, copy=False)

    # Generate angular values.
    theta_degrees = np.arange(
        0,
        180,
        theta_step,
        dtype=np.float64,
    )

    thetas = np.deg2rad(theta_degrees)

    n_theta = thetas.size

    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)

    # Determine accumulator dimensions.
    max_rho = int(np.ceil(np.hypot(height - 1, width - 1)))

    rho_bins = 2 * max_rho + 1

    accumulator = np.zeros(
        (rho_bins, n_theta),
        dtype=np.uint32,
    )

    # Precompute theta offsets for flattened accumulator indices.
    theta_offsets = np.arange(n_theta, dtype=np.int64)

    # ------------------------------------------------------------------------
    # Voting
    # ------------------------------------------------------------------------
    # Process edge pixels in batches to avoid allocating a huge
    # (number_of_edges x number_of_angles) array.
    #
    # np.bincount is used instead of np.add.at because it is generally
    # more efficient for accumulating repeated integer indices.

    for start in range(0, xs.size, chunk_size):
        end = min(start + chunk_size, xs.size)

        batch_x = xs[start:end]
        batch_y = ys[start:end]

        rho_values = (
            batch_x[:, None] * cos_t[None, :]
            + batch_y[:, None] * sin_t[None, :]
        )

        rho_indices = (
            np.rint(rho_values).astype(np.int32) + max_rho
        )

        flat_indices = (
            rho_indices.astype(np.int64) * n_theta
            + theta_offsets[None, :]
        ).ravel()

        votes = np.bincount(
            flat_indices,
            minlength=rho_bins * n_theta,
        )

        accumulator += votes.reshape(rho_bins, n_theta).astype(
            np.uint32,
            copy=False,
        )

    # ------------------------------------------------------------------------
    # Local maximum filtering
    # ------------------------------------------------------------------------

    # Compare each accumulator cell with its 8 neighbors.
    # Padding prevents out-of-range accesses.
    padded = np.pad(
        accumulator,
        ((1, 1), (1, 1)),
        mode="constant",
        constant_values=0,
    )

    local_maximum = np.zeros_like(accumulator)

    for dy in range(3):
        for dx in range(3):
            if dy == 1 and dx == 1:
                continue

            neighbor = padded[
                dy:dy + rho_bins,
                dx:dx + n_theta,
            ]

            np.maximum(
                local_maximum,
                neighbor,
                out=local_maximum,
            )

    peak_mask = (
        (accumulator >= threshold)
        & (accumulator >= local_maximum)
    )

    peak_rho, peak_theta = np.nonzero(peak_mask)

    if peak_rho.size == 0:
        return []

    peak_votes = accumulator[peak_rho, peak_theta]

    # Process strongest candidates first.
    order = np.argsort(peak_votes)[::-1]

    peak_rho = peak_rho[order]
    peak_theta = peak_theta[order]

    # ------------------------------------------------------------------------
    # Duplicate suppression
    # ------------------------------------------------------------------------

    detected_lines: list[Line] = []

    for rho_idx, theta_idx in zip(peak_rho, peak_theta):
        rho = float(rho_idx - max_rho)
        theta = float(thetas[theta_idx])

        duplicate = False

        for existing in detected_lines:
            delta_theta = abs(
                np.rad2deg(theta - existing.theta)
            )

            delta_rho = abs(rho - existing.rho)

            # (rho, theta) and (-rho, theta + 180 degrees)
            # describe the same geometric line.
            if delta_theta > 90.0:
                delta_theta = 180.0 - delta_theta
                delta_rho = abs(rho + existing.rho)

            if (
                delta_theta <= min_theta_distance
                and delta_rho <= min_rho_distance
            ):
                duplicate = True
                break

        if not duplicate:
            detected_lines.append(Line(rho, theta))

    return detected_lines
