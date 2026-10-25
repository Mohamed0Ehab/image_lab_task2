"""ImageData: load/save, cached gray conversion, channel split. Used by the whole app.

Lives in core package as the central data representation for images.
"""
from __future__ import annotations

from functools import cached_property
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


class AppError(Exception):
    """Base class for application-specific errors with user-friendly messages."""


class ImageLoadError(AppError):
    """Raised when an image file cannot be read, decoded or saved."""


class UnsupportedImageError(AppError):
    """Raised when image data is invalid or unsupported."""


SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
)


def to_gray(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB array to 2D grayscale using standard luminance 0.299 R + 0.587 G + 0.114 B."""
    if rgb.ndim == 2:
        return rgb
    if rgb.shape[2] >= 3:
        r = rgb[..., 0].astype(np.float32)
        g = rgb[..., 1].astype(np.float32)
        b = rgb[..., 2].astype(np.float32)
        return 0.299 * r + 0.587 * g + 0.114 * b
    return rgb[..., 0]


def to_uint8(arr: np.ndarray) -> np.ndarray:
    """Clip and convert array to uint8."""
    if arr.dtype == np.uint8:
        return arr
    return np.clip(np.round(arr), 0, 255).astype(np.uint8)


class ImageData:
    """ImageData: load/save, cached gray conversion, channel split. Used by the whole app."""

    def __init__(self, array: np.ndarray, path: str | None = None) -> None:
        if not isinstance(array, np.ndarray) or array.size == 0:
            raise UnsupportedImageError("Invalid image data.")
        self.array = array
        self.path = path

    @classmethod
    def load(cls, path: str, as_gray: bool = False) -> ImageData:
        """OpenCV is allowed here (task 1: read RGB and gray images)."""
        p = Path(path)
        if not p.exists():
            raise ImageLoadError(f"File not found: {p.name}")
        if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ImageLoadError(
                f"Unsupported file type '{p.suffix}'.\n"
                f"Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )
        try:
            # np.fromfile + imdecode so Windows paths with non-ASCII characters work.
            data = np.fromfile(str(p), dtype=np.uint8)
            img = cv2.imdecode(
                data, cv2.IMREAD_GRAYSCALE if as_gray else cv2.IMREAD_COLOR
            )
        except Exception as exc:
            raise ImageLoadError(f"Cannot read '{p.name}': {exc}") from exc

        if img is None:
            raise ImageLoadError(f"Could not read '{p.name}' (corrupt file?).")

        if not as_gray:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # the ONLY place channel order changes

        return cls(img, str(p))

    @classmethod
    def from_array(cls, array: np.ndarray, path: str | None = None) -> ImageData:
        return cls(array, path)

    @staticmethod
    def save(path: str, image: np.ndarray) -> None:
        p = Path(path)
        ext = p.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise ImageLoadError(f"Unsupported file type '{ext}'.")
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if image.ndim == 3 else image
        ok, buf = cv2.imencode(ext, bgr)
        if not ok:
            raise ImageLoadError("Could not encode the image.")
        buf.tofile(str(p))

    @property
    def is_gray(self) -> bool:
        return self.array.ndim == 2

    @property
    def shape(self) -> tuple:
        return self.array.shape

    @property
    def channels(self) -> tuple:
        """(R, G, B) arrays, or (gray,) for a gray image."""
        if self.is_gray:
            return (self.array,)
        return tuple(self.array[..., i] for i in range(min(3, self.array.shape[2])))

    @cached_property
    def gray(self) -> np.ndarray:
        """0.299 R + 0.587 G + 0.114 B. Cached: computed once per image."""
        if self.is_gray:
            return self.array
        return to_uint8(to_gray(self.array[..., :3]))

    @property
    def rgb(self) -> np.ndarray:
        """3-channel RGB array."""
        if self.is_gray:
            return cv2.cvtColor(self.array, cv2.COLOR_GRAY2RGB)
        if self.array.ndim == 3 and self.array.shape[2] == 4:
            return self.array[:, :, :3]
        return self.array

    def as_mode(self, mode: str) -> np.ndarray:
        """Return image array in the requested mode ('Color RGB' or 'Grayscale')."""
        if mode == "Grayscale":
            return self.gray
        return self.rgb
