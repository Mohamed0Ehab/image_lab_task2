"""Centralised error handling for ImageLab.

Public API
----------
show_error(exc, parent)  – display the right dialog for any exception.
run_safely(fn, parent)   – call fn() and route exceptions to show_error.

Custom exception classes
------------------------
AppError              – base; shows a QMessageBox.warning.
ImageLoadError        – file I/O problems.
InvalidParameterError – bad algorithm parameters.
"""
from __future__ import annotations

import logging
import traceback
from typing import Any, Callable, Optional

from PySide6.QtWidgets import QMessageBox, QWidget

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------

from core.image_data import AppError, ImageLoadError, UnsupportedImageError


class InvalidParameterError(AppError):
    """Raised when algorithm parameters are outside valid ranges."""


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def show_error(exc: Exception, parent: Optional[QWidget] = None) -> None:
    """Log *exc* and display an appropriate Qt message box.

    * ``NotImplementedError``           → Information box ("not implemented yet").
    * ``AppError`` (and subclasses)     → Warning box with the exception message.
    * Everything else                   → Critical box with collapsible traceback.
    """
    tb = traceback.format_exc()
    logger.error("Exception caught: %s: %s\n%s", type(exc).__name__, exc, tb)

    if isinstance(exc, NotImplementedError):
        box = QMessageBox(parent)
        box.setIcon(QMessageBox.Icon.Information)
        box.setWindowTitle("Not implemented")
        box.setText(
            "This feature is not implemented yet\n"
            "(a teammate is still working on it)."
        )
        box.exec()
        return

    if isinstance(exc, AppError):
        box = QMessageBox(parent)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("Warning")
        box.setText(str(exc))
        box.exec()
        return

    # Unexpected exception
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Critical)
    box.setWindowTitle("Unexpected Error")
    box.setText(
        "An unexpected error occurred.  The application will continue.\n"
        f"({type(exc).__name__}: {exc})"
    )
    box.setDetailedText(tb)
    box.exec()


def run_safely(
    fn: Callable[[], Any],
    parent: Optional[QWidget] = None,
) -> Any:
    """Execute *fn()* inside try/except and route any exception to show_error.

    Returns the return value of *fn()* on success, or ``None`` on failure.
    Every button, menu action and signal handler should use this wrapper.
    """
    try:
        return fn()
    except Exception as exc:
        show_error(exc, parent)
        return None
