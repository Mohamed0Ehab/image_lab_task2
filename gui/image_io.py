"""Image I/O, colour-mode conversion, and QPixmap rendering.

Uses ImageData from the core package as the central image container.

Public API
----------
ImageData                           – Central image data container (from core.image_data).
load_image(path, mode)              -> ImageData
convert_mode(source, mode, **opts)  -> ndarray
save_image(path, array)
ensure_gray(array)                  -> ndarray
array_to_qpixmap(array)             -> QPixmap

Constants
---------
MODES         – 2 display modes: ["Color RGB", "Grayscale"].
SUPPORTED_EXT – set of lower-case file extensions that can be opened.
"""
from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np
from PySide6.QtGui import QImage, QPixmap

from core.image_data import (
    ImageData,
    ImageLoadError,
    UnsupportedImageError,
    SUPPORTED_EXTENSIONS as SUPPORTED_EXT,
    to_gray,
    to_uint8,
)

logger = logging.getLogger(__name__)

MODES: list[str] = [
    "Color RGB",
    "Grayscale",
]


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def load_image(path: str, mode: str = "Color RGB") -> ImageData:
    """Load *path* and return an ImageData object."""
    return ImageData.load(path, as_gray=(mode == "Grayscale"))


def convert_mode(source: np.ndarray | ImageData, mode: str, **opts: Any) -> np.ndarray:
    """Convert *source* to the display *mode* ('Color RGB' or 'Grayscale')."""
    if isinstance(source, ImageData):
        return source.as_mode(mode)
    if mode == "Grayscale":
        return to_uint8(to_gray(source))
    return _to_rgb(source)


def save_image(path: str, array: np.ndarray | ImageData) -> None:
    """Save *array* (or ImageData) to *path*."""
    arr = array.array if isinstance(array, ImageData) else array
    ImageData.save(path, arr)


def ensure_gray(array: np.ndarray | ImageData) -> np.ndarray:
    """Return a 2-D grayscale uint8 array."""
    if isinstance(array, ImageData):
        return array.gray
    return to_uint8(to_gray(array))


def array_to_qpixmap(array: np.ndarray | ImageData) -> QPixmap:
    """Convert a numpy uint8 array or ImageData to a ``QPixmap``."""
    arr = array.array if isinstance(array, ImageData) else array

    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    arr = np.ascontiguousarray(arr)

    if arr.ndim == 2:
        h, w = arr.shape
        qimg = QImage(arr.data, w, h, w, QImage.Format.Format_Grayscale8)

    elif arr.ndim == 3:
        h, w, c = arr.shape
        if c == 3:
            qimg = QImage(arr.data, w, h, w * 3, QImage.Format.Format_RGB888)
        elif c == 4:
            qimg = QImage(arr.data, w, h, w * 4, QImage.Format.Format_RGBA8888)
        else:
            arr2 = np.ascontiguousarray(arr[:, :, 0])
            qimg = QImage(arr2.data, w, h, w, QImage.Format.Format_Grayscale8)
    else:
        raise ValueError(
            f"Cannot convert array with ndim={arr.ndim} to QPixmap"
        )

    return QPixmap.fromImage(qimg.copy())


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _to_rgb(arr: np.ndarray) -> np.ndarray:
    """Return a 3-channel RGB copy of *arr*."""
    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    if arr.ndim == 3 and arr.shape[2] == 4:
        return arr[:, :, :3].copy()
    return arr.copy()
