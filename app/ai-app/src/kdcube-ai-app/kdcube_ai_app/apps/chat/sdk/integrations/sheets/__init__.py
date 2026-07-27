# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""Provider-neutral spreadsheet named-service integration."""

from kdcube_ai_app.apps.chat.sdk.integrations.sheets.named_service import (
    SHEETS_NAMESPACE,
    SheetsNamedServiceProvider,
    make_sheets_named_service_provider,
    parse_sheets_ref,
    sheets_named_service_spec,
)

__all__ = [
    "SHEETS_NAMESPACE",
    "SheetsNamedServiceProvider",
    "make_sheets_named_service_provider",
    "parse_sheets_ref",
    "sheets_named_service_spec",
]
