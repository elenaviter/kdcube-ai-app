"""Productivity services exposed by the built-in KDCube services app."""

from .google_sheets import (
    GoogleSheetsService,
    bind_service,
    fetch_google_sheets_snapshot,
)

__all__ = [
    "GoogleSheetsService",
    "bind_service",
    "fetch_google_sheets_snapshot",
]
