"""Small cross-platform filesystem operations for secret-safe writers."""

from __future__ import annotations

import os
from pathlib import Path


def apply_open_file_mode(descriptor: int, path: Path, mode: int) -> None:
    """Apply a mode through the descriptor when supported, otherwise by path."""

    fchmod = getattr(os, "fchmod", None)
    if callable(fchmod):
        fchmod(descriptor, mode)
        return
    os.chmod(path, mode)


__all__ = ["apply_open_file_mode"]
