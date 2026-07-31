# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""Provider-neutral document named-service integration."""

from kdcube_ai_app.apps.chat.sdk.integrations.docs.named_service import (
    DOCS_NAMESPACE,
    DocsNamedServiceProvider,
    document_export_ref,
    docs_named_service_spec,
    make_docs_named_service_provider,
    parse_docs_export_ref,
    parse_docs_ref,
)

__all__ = [
    "DOCS_NAMESPACE",
    "DocsNamedServiceProvider",
    "document_export_ref",
    "docs_named_service_spec",
    "make_docs_named_service_provider",
    "parse_docs_export_ref",
    "parse_docs_ref",
]
