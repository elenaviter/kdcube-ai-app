# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

# chat/sdk/tools/md_utils.py

import re, json
from typing import Optional, Dict, List, Set

from kdcube_ai_app.apps.chat.sdk.tools.citations import (
    # re-exports for backward compatibility
    CITE_TOKEN_RE,
    replace_citation_tokens_streaming as replace_citation_tokens_streaming,
    replace_citation_tokens_batch as replace_citation_tokens,  # batch replacement
    split_safe_citation_prefix as _split_safe_citation_prefix,
    append_sources_section_if_missing as _append_sources_section,
    create_clean_references_section as _create_clean_sources_section,
    _replace_citation_tokens as _replace_citation_tokens,
    _normalize_sources as _normalize_sources,
    build_citation_map_from_citations as build_citation_map,
)

CODE_FENCE_RE = re.compile(r"^```[a-zA-Z0-9_-]*\s*|\s*```$", re.M)
