# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

# chat/sdk/streaming/streaming.py

from __future__ import annotations

import logging
import re
from typing import Dict, Any, Optional, Callable, Tuple, List

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from kdcube_ai_app.infra.service_hub.inventory import ModelServiceBase, AgentLogger
from kdcube_ai_app.apps.chat.sdk.util import _json_loads_loose

logger = logging.getLogger(__name__)

# =============================================================================
# Paranoid markers (ONLY these are supported)
# =============================================================================

PARA_INT_OPEN  = r"<<<\s*BEGIN\s+INTERNAL\s+THINKING\s*>>>"
PARA_USER_OPEN = r"<<<\s*BEGIN\s+USER[-\u2013\u2014]?\s*FACING\s+THINKING\s*>>>"
PARA_JSON_OPEN = r"<<<\s*BEGIN\s+STRUCTURED\s+JSON\s*>>>"

PARA_INT_CLOSE  = r"<<<\s*END\s+INTERNAL\s+THINKING\s*>>>"
PARA_USER_CLOSE = r"<<<\s*END\s+USER[-\u2013\u2014]?\s*FACING\s+THINKING\s*>>>"
PARA_JSON_CLOSE = r"<<<\s*END\s+STRUCTURED\s+JSON\s*>>>"

INT_RE  = re.compile(PARA_INT_OPEN, re.I)
USER_RE = re.compile(PARA_USER_OPEN, re.I)
JSON_RE = re.compile(PARA_JSON_OPEN, re.I)

# We never *require* closers, but if a model emits them, strip from user-facing output.
USER_CLOSER_STRIPPERS = [
    re.compile(PARA_USER_CLOSE, re.I),
    re.compile(PARA_INT_CLOSE, re.I),
    re.compile(PARA_JSON_CLOSE, re.I),
    # also safeguard: if any open markers appear inside the user section, strip them
    USER_RE, INT_RE, JSON_RE,
]

# Rolling safety window so we never cut a marker across chunk boundaries
MAX_MARKER_LEN = max(
    len("<<< BEGIN INTERNAL THINKING >>>"),
    len("<<< BEGIN USER-FACING THINKING >>>"),
    len("<<< BEGIN STRUCTURED JSON >>>"),
    len("<<< END INTERNAL THINKING >>>"),
    len("<<< END USER-FACING THINKING >>>"),
    len("<<< END STRUCTURED JSON >>>"),
)
HOLDBACK = MAX_MARKER_LEN + 8


# =============================================================================
# Public helper: append strict 3-section protocol (paranoid markers only)
# =============================================================================

def _add_3section_protocol(base: str, json_shape_hint: str) -> str:
    """
    Appends a strict output protocol to the system prompt telling the model to use the
    paranoid markers. We explicitly tell it NOT to emit END markers.
    """
    return (
            base.rstrip()
            + "\n\nCRITICAL OUTPUT PROTOCOL — FOLLOW EXACTLY:\n"
              "• Produce THREE sections in this exact order. Use each START marker below exactly once.\n"
              "• Do NOT write any closing tags/markers like <<< END … >>> or </…>.\n"
              "• The third section must be a fenced JSON block and contain ONLY JSON.\n\n"
              "1) <<< BEGIN INTERNAL THINKING >>>\n"
              "   (private notes in Markdown; not shown to user)\n"
              "2) <<< BEGIN USER-FACING THINKING >>>\n"
              "   (short, user-friendly status; Markdown only — no JSON)\n"
              "3) <<< BEGIN STRUCTURED JSON >>>\n"
              f"   ```json\n{json_shape_hint}\n```\n"
              "Return exactly these three sections, in order, once."
    )


# =============================================================================
# Internal streaming parser (used by both the runtime and unit tests)
# =============================================================================

def _skip_ws(s: str, i: int) -> int:
    while i < len(s) and s[i] in (" ", "\t", "\r", "\n"):
        i += 1
    return i


def _strip_service_markers_from_user(text: str) -> str:
    out = text
    for pat in USER_CLOSER_STRIPPERS:
        out = pat.sub("", out)
    return out


def _find_earliest(buf: str, start_hint: int, patterns: List[Tuple[re.Pattern, str]]) -> Tuple[Optional[re.Match], Optional[str]]:
    """
    Search all patterns starting a little earlier to catch split markers.
    Return (match, which_name) for the earliest match; else (None, None).
    """
    start = max(0, start_hint - HOLDBACK)
    found = []
    for pat, name in patterns:
        m = pat.search(buf, start)
        if m:
            found.append((m, name))
    if not found:
        return None, None
    m, name = min(found, key=lambda t: t[0].start())
    return m, name

# Enhanced patterns and buffer management for closing marker detection
CLOSING_MARKER_PATTERNS = [
    re.compile(PARA_USER_CLOSE, re.I),
    re.compile(PARA_INT_CLOSE, re.I),
    re.compile(PARA_JSON_CLOSE, re.I),
    # Also catch any stray opening markers that appear in user content
    re.compile(PARA_INT_OPEN, re.I),
    re.compile(PARA_USER_OPEN, re.I),
    re.compile(PARA_JSON_OPEN, re.I),
]

# Increase holdback to be more conservative with closing markers
CLOSING_MARKER_HOLDBACK = 50  # Hold back more aggressively in user mode


# class _SectionsStreamParser:
#     """
#     Enhanced parser that prevents streaming of closing markers by using
#     aggressive buffering and pre-scanning in user mode.
#     """
#
#     def __init__(self, on_user: Callable[[str, bool], Any]):
#         self.buf = ""
#         self.mode: str = "pre"  # pre -> internal -> user -> json
#         self.emit_from: int = 0
#         self.internal_parts: List[str] = []
#         self.user_parts: List[str] = []
#         self.json_tail: str = ""
#         self._on_user = on_user
#         self._user_buffer = ""  # Special buffer for user content to prevent marker leakage
#
#     def _safe_window_end(self) -> int:
#         # Everything before this index is safe to emit (no risk of cutting a marker)
#         return max(self.emit_from, len(self.buf) - HOLDBACK)
#
#     def _safe_user_window_end(self) -> int:
#         # More conservative window for user mode to prevent closing marker leakage
#         return max(self.emit_from, len(self.buf) - CLOSING_MARKER_HOLDBACK)
#
#     def _contains_closing_marker(self, text: str) -> bool:
#         """Check if text contains any closing markers."""
#         for pattern in CLOSING_MARKER_PATTERNS:
#             if pattern.search(text):
#                 return True
#         return False
#
#     def _extract_safe_user_content(self, text: str) -> Tuple[str, str]:
#         """
#         Extract content that's safe to emit, stopping before any closing marker.
#         Returns (safe_content, remaining_text).
#         """
#         # Find the earliest closing marker
#         earliest_match = None
#         earliest_pos = len(text)
#
#         for pattern in CLOSING_MARKER_PATTERNS:
#             match = pattern.search(text)
#             if match and match.start() < earliest_pos:
#                 earliest_match = match
#                 earliest_pos = match.start()
#
#         if earliest_match:
#             # Extract content before the marker
#             safe_content = text[:earliest_pos]
#             remaining = text[earliest_pos:]
#             return safe_content, remaining
#         else:
#             return text, ""
#
#     async def _flush_user_buffer(self, final: bool = False):
#         """Flush accumulated user content, checking for closing markers."""
#         if not self._user_buffer:
#             return
#
#         if final:
#             # At end of stream, emit everything after stripping markers
#             cleaned = _strip_service_markers_from_user(self._user_buffer)
#             if cleaned:
#                 await self._on_user(cleaned, False)
#                 self.user_parts.append(cleaned)
#             self._user_buffer = ""
#         else:
#             # Extract safe content that doesn't contain closing markers
#             safe_content, remaining = self._extract_safe_user_content(self._user_buffer)
#
#             if safe_content:
#                 cleaned = _strip_service_markers_from_user(safe_content)
#                 if cleaned:
#                     await self._on_user(cleaned, False)
#                     self.user_parts.append(cleaned)
#
#             self._user_buffer = remaining
#
#     async def feed(self, piece: str, prev_len_hint: Optional[int] = None):
#         """Feed a new streamed piece."""
#         if not piece:
#             return
#         prev_len = len(self.buf) if prev_len_hint is None else prev_len_hint
#         self.buf += piece
#
#         # Iterate until we can't progress further for this piece
#         while True:
#             if self.mode == "pre":
#                 m, which = _find_earliest(
#                     self.buf,
#                     prev_len,
#                     [(INT_RE, "internal"), (USER_RE, "user"), (JSON_RE, "json")],
#                 )
#                 if not m:
#                     break  # wait for more bytes
#                 self.emit_from = _skip_ws(self.buf, m.end())
#                 self.mode = which
#                 continue
#
#             if self.mode == "internal":
#                 # Next can be USER or JSON (model may skip USER)
#                 m, which = _find_earliest(
#                     self.buf, prev_len, [(USER_RE, "user"), (JSON_RE, "json")]
#                 )
#                 if m and m.start() >= self.emit_from:
#                     self.internal_parts.append(self.buf[self.emit_from : m.start()])
#                     self.emit_from = _skip_ws(self.buf, m.end())
#                     if which == "json":
#                         self.json_tail = self.buf[self.emit_from :]
#                         self.mode = "json"
#                         break
#                     self.mode = "user"
#                     continue
#
#                 # Accumulate safe internal slice (not emitted)
#                 safe_end = self._safe_window_end()
#                 if safe_end > self.emit_from:
#                     self.internal_parts.append(self.buf[self.emit_from : safe_end])
#                     self.emit_from = safe_end
#                 break
#
#             if self.mode == "user":
#                 # Check for JSON transition
#                 m, _ = _find_earliest(self.buf, prev_len, [(JSON_RE, "json")])
#                 if m and m.start() >= self.emit_from:
#                     # Add remaining content to user buffer before transitioning
#                     chunk = self.buf[self.emit_from : m.start()]
#                     if chunk:
#                         self._user_buffer += chunk
#
#                     # Flush user buffer and transition to JSON
#                     await self._flush_user_buffer(final=True)
#                     self.emit_from = _skip_ws(self.buf, m.end())
#                     self.json_tail = self.buf[self.emit_from :]
#                     self.mode = "json"
#                     break
#
#                 # Accumulate user content in buffer with conservative windowing
#                 safe_end = self._safe_user_window_end()
#                 if safe_end > self.emit_from:
#                     chunk = self.buf[self.emit_from : safe_end]
#                     if chunk:
#                         self._user_buffer += chunk
#                         self.emit_from = safe_end
#
#                         # Try to flush safe content from buffer
#                         await self._flush_user_buffer(final=False)
#                 break
#
#             if self.mode == "json":
#                 self.json_tail += piece
#                 break
#
#             break  # safety
#
#     async def finalize(self):
#         """Flush any remaining content at end-of-stream."""
#         if self.mode == "internal" and self.emit_from < len(self.buf):
#             self.internal_parts.append(self.buf[self.emit_from :])
#         elif self.mode == "user":
#             # Add any remaining buffer content
#             if self.emit_from < len(self.buf):
#                 self._user_buffer += self.buf[self.emit_from :]
#             # Flush user buffer with final=True to strip markers
#             await self._flush_user_buffer(final=True)
#
#     # Convenience results
#     @property
#     def internal(self) -> str:
#         return "".join(self.internal_parts).strip()
#
#     @property
#     def user(self) -> str:
#         return "".join(self.user_parts).strip()
#
#     @property
#     def json(self) -> str:
#         return self.json_tail


class _SectionsStreamParser:
    """
    Modes:
      pre   -> before any BEGIN markers; we buffer until we see a BEGIN
      internal -> after <<< BEGIN INTERNAL THINKING >>>, captured only
      user  -> after <<< BEGIN USER-FACING THINKING >>>, streamed via callback
      json  -> after <<< BEGIN STRUCTURED JSON >>>, captured only
    """
    import re as _re

    # BEGIN markers (existing)
    _INT_RE  = _re.compile(r"<<<?\s*BEGIN\s+INTERNAL\s+THINKING\s*>>>?", _re.I)
    _USER_RE = _re.compile(r"<<<?\s*BEGIN\s+USER[-–—]?\s*FACING\s*THINKING\s*>>>?", _re.I)
    _JSON_RE = _re.compile(r"<<<?\s*BEGIN\s+STRUCTURED\s+JSON\s*>>>?", _re.I)

    # NEW: things we never want to leak to the UI during USER streaming
    _UNWANTED_USER_MARKERS = [
        _re.compile(r"<<<?\s*END\s+USER[-–—]?\s*FACING\s*THINKING\s*>>>?", _re.I),
        _re.compile(r"<<<?\s*END\s+INTERNAL\s+THINKING\s*>>>?", _re.I),
        _re.compile(r"<<<?\s*END\s+STRUCTURED\s+JSON\s*>>>?", _re.I),
        # If the model echoes a BEGIN back into the user-facing stream, scrub it too:
        _re.compile(r"<<<?\s*BEGIN\s+USER[-–—]?\s*FACING\s*THINKING\s*>>>?", _re.I),
    ]

    def __init__(self, *, on_user):
        self.buf = ""
        self.mode = "pre"
        self.emit_from = 0
        self.on_user = on_user
        self.internal = ""
        self.user = ""
        self.json = ""
        # keep small holdback so we don’t cut markers across chunk boundaries
        self.HOLDBACK = 48

    def _strip_unwanted_user_markers(self, s: str) -> str:
        for pat in self._UNWANTED_USER_MARKERS:
            s = pat.sub("", s)
        return s

    def _skip_ws(self, i: int) -> int:
        while i < len(self.buf) and self.buf[i] in (" ", "\t", "\r", "\n"):
            i += 1
        return i

    async def feed(self, piece: str):
        if not piece:
            return
        prev_len = len(self.buf)
        self.buf += piece

        def _search(pat):
            start = max(0, prev_len - self.HOLDBACK, self.emit_from - self.HOLDBACK)
            return pat.search(self.buf, start)

        while True:
            if self.mode == "pre":
                m = (_SectionsStreamParser._INT_RE.search(self.buf, max(0, prev_len - self.HOLDBACK))
                     or _SectionsStreamParser._USER_RE.search(self.buf, max(0, prev_len - self.HOLDBACK))
                     or _SectionsStreamParser._JSON_RE.search(self.buf, max(0, prev_len - self.HOLDBACK)))
                if not m:
                    # nothing yet; just wait (do not emit anything in pre)
                    break

                tag = m.re
                pre_slice = self.buf[self.emit_from:m.start()]
                if pre_slice:
                    # internal prelude lives here; we don’t show it
                    self.internal += pre_slice
                self.emit_from = self._skip_ws(m.end())

                if tag is _SectionsStreamParser._INT_RE:
                    self.mode = "internal"
                elif tag is _SectionsStreamParser._USER_RE:
                    self.mode = "user"
                else:
                    self.mode = "json"
                continue

            if self.mode == "internal":
                # look for next begin (USER or JSON)
                m_user = _search(_SectionsStreamParser._USER_RE)
                m_json = _search(_SectionsStreamParser._JSON_RE)
                m = min([x for x in (m_user, m_json) if x], key=lambda x: x.start()) if (m_user or m_json) else None
                if not m:
                    # accumulate internal silently
                    # keep small safe end (no need to stream)
                    safe_end = max(self.emit_from, len(self.buf) - self.HOLDBACK)
                    if safe_end > self.emit_from:
                        self.internal += self.buf[self.emit_from:safe_end]
                        self.emit_from = safe_end
                    break

                self.internal += self.buf[self.emit_from:m.start()]
                self.emit_from = self._skip_ws(m.end())
                self.mode = "user" if m.re is _SectionsStreamParser._USER_RE else "json"
                continue

            if self.mode == "user":
                # look for JSON begin (end of user section)
                m_json = _search(_SectionsStreamParser._JSON_RE)
                if not m_json:
                    # stream a safe window (but scrub unwanted markers)
                    safe_end = max(self.emit_from, len(self.buf) - self.HOLDBACK)
                    if safe_end > self.emit_from:
                        raw = self.buf[self.emit_from:safe_end]
                        cleaned = self._strip_unwanted_user_markers(raw)
                        if cleaned:
                            await self.on_user(cleaned, completed=False)
                            self.user += cleaned
                        self.emit_from = safe_end
                    break

                # emit everything up to JSON marker (scrubbed), then switch
                raw = self.buf[self.emit_from:m_json.start()]
                cleaned = self._strip_unwanted_user_markers(raw)
                if cleaned:
                    await self.on_user(cleaned, completed=False)
                    self.user += cleaned
                self.emit_from = self._skip_ws(m_json.end())
                self.mode = "json"
                # json section begins; loop to handle tail capture
                continue

            if self.mode == "json":
                # rest is JSON tail; just accumulate
                self.json += self.buf[self.emit_from:]
                self.emit_from = len(self.buf)
                break

            break  # safety

    async def finalize(self):
        # flush any remaining text per mode
        if self.mode == "internal":
            self.internal += self.buf[self.emit_from:]
        elif self.mode == "user":
            raw = self.buf[self.emit_from:]
            cleaned = self._strip_unwanted_user_markers(raw)
            if cleaned:
                await self.on_user(cleaned, completed=False)
                self.user += cleaned
        elif self.mode == "json":
            self.json += self.buf[self.emit_from:]
        # caller will send the completed=True signal

# =============================================================================
# Public streaming entrypoint
# =============================================================================

# async def _stream_agent_sections_to_json(
#         svc: ModelServiceBase,
#         *,
#         client_name: str,
#         client_role: str,
#         sys_prompt: str,
#         user_msg: str,
#         schema_model,
#         on_thinking_delta=None,
#         temperature: float = 0.2,
#         max_tokens: int = 1200,
#         ctx: Optional[Any] = None,
# ) -> Dict[str, Any]:
#     """
#     Stream 3 sections using ONLY paranoid markers:
#       1) <<< BEGIN INTERNAL THINKING >>> (capture only)
#       2) <<< BEGIN USER-FACING THINKING >>> (stream via on_thinking_delta)
#       3) <<< BEGIN STRUCTURED JSON >>> (strict; returned)
#
#     The parser keeps a small rolling HOLDBACK so we never cut a marker across chunk
#     boundaries, but otherwise streams promptly. Service markers are never emitted
#     to the UI.
#     """
#     run_logger = AgentLogger("ThreeSectionStream", getattr(svc.config, "log_level", "INFO"))
#     run_logger.start_operation(
#         "three_section_stream",
#         client_role=client_role,
#         client_name=client_name,
#         system_prompt_len=len(sys_prompt),
#         user_msg_len=len(user_msg),
#         expected_format=getattr(schema_model, "__name__", str(schema_model)),
#     )
#
#     deltas = 0
#
#     async def _emit_user(piece: str, completed: bool = False, **kwargs):
#         nonlocal deltas
#         if not piece and not completed:
#             return
#         if on_thinking_delta is not None:
#             await on_thinking_delta(piece or "", completed, **kwargs)
#             if piece:
#                 deltas += 1
#
#     parser = _SectionsStreamParser(on_user=_emit_user)
#
#     async def on_delta(piece: str):
#         await parser.feed(piece)
#
#     async def on_complete(ret):
#         await parser.finalize()
#         # send the "completed" signal to the UI
#         await _emit_user("", completed=True)
#         run_logger.log_step("stream_complete", {"deltas": deltas})
#
#     # ---- stream with model ----
#     client = svc.get_client(client_name)
#     cfg = svc.describe_client(client, role=client_role)
#
#     messages = [SystemMessage(content=sys_prompt), HumanMessage(content=user_msg)]
#     await svc.stream_model_text_tracked(
#         client,
#         messages,
#         on_delta=on_delta,
#         on_complete=on_complete,
#         temperature=temperature,
#         max_tokens=max_tokens,
#         client_cfg=cfg,
#         role=client_role,
#     )
#
#     internal_thinking = parser.internal
#     user_thinking = parser.user
#     json_tail = parser.json
#
#     # ---- parse JSON (tolerate fenced block) ----
#     def _try_parse_json(raw_json: str) -> Optional[Dict[str, Any]]:
#         raw = (raw_json or "").strip()
#         # handle fenced block if present
#         if raw.startswith("```"):
#             nl = raw.find("\n")
#             raw = raw[nl + 1 :] if nl >= 0 else ""
#             end = raw.rfind("```")
#             if end >= 0:
#                 raw = raw[:end]
#         raw = raw.strip().strip("`")
#         if not raw:
#             return None
#         m = re.search(r"\{.*\}\s*$", raw, re.S)
#         if not m:
#             return None
#         try:
#             loaded = _json_loads_loose(m.group(0)) or {}
#             return schema_model.model_validate(loaded).model_dump()
#         except Exception:
#             return None
#
#     data = _try_parse_json(json_tail)
#     raw_data = json_tail
#     error = None
#
#     if data is None and json_tail.strip():
#         # try format fixer once
#         fix = await svc.format_fixer.fix_format(
#             raw_output=json_tail,
#             expected_format=getattr(schema_model, "__name__", str(schema_model)),
#             input_data=user_msg,
#             system_prompt=sys_prompt,
#         )
#         if fix.get("success"):
#             try:
#                 data = schema_model.model_validate(fix["data"]).model_dump()
#             except Exception as ex:
#                 error = f"JSON parse after format fix failed: {ex}"
#                 data = None
#
#
#     # Retry streaming once if JSON missing entirely (model stopped early)
#     # if data is None and not json_tail.strip():
#     #     produced = []
#     #     if internal_thinking or parser.mode in ("user", "json"):
#     #         produced.append("<<< BEGIN INTERNAL THINKING >>>\n" + internal_thinking)
#     #     if user_thinking or parser.mode == "json":
#     #         produced.append("<<< BEGIN USER-FACING THINKING >>>\n" + user_thinking)
#     #     seed_assistant = "\n".join(produced).strip()
#     #
#     #     continue_note = (
#     #         "You already produced the earlier sections shown above. "
#     #         "Continue by producing ONLY the remaining sections in the exact order, starting with "
#     #         "<<< BEGIN STRUCTURED JSON >>>. "
#     #         "Return the remaining sections exactly once."
#     #     )
#     #     sys_prompt_retry = sys_prompt + "\n\n[CONTINUE]\n" + continue_note
#     #
#     #     buf2 = ""
#     #     mode2 = "pre"
#     #     emit_from2 = 0
#     #     json_tail2 = ""
#     #
#     #     def _safe_window_end2(current_len: int) -> int:
#     #         return max(emit_from2, current_len - HOLDBACK)
#     #
#     #     async def on_delta_retry(piece: str):
#     #         nonlocal buf2, mode2, emit_from2, json_tail2
#     #         if not piece:
#     #             return
#     #         prev_len2 = len(buf2)
#     #         buf2 += piece
#     #
#     #         if mode2 == "pre":
#     #             m = JSON_RE.search(buf2, max(0, prev_len2 - HOLDBACK))
#     #             if m:
#     #                 emit_from2 = _skip_ws(buf2, m.end())
#     #                 json_tail2 = buf2[emit_from2:]
#     #                 mode2 = "json"
#     #             return
#     #
#     #         if mode2 == "json":
#     #             json_tail2 += piece
#     #             return
#     #
#     #     retry_msgs = [SystemMessage(content=sys_prompt_retry), HumanMessage(content=user_msg)]
#     #     if seed_assistant:
#     #         retry_msgs.append(AIMessage(content=seed_assistant))
#     #
#     #     await svc.stream_model_text_tracked(
#     #         client,
#     #         retry_msgs,
#     #         on_delta=on_delta_retry,
#     #         on_complete=None,
#     #         temperature=temperature,
#     #         max_tokens=max_tokens,
#     #         client_cfg=cfg,
#     #         role=client_role,
#     #     )
#     #
#     #     data = _try_parse_json(json_tail2)
#     #     if data is None and json_tail2.strip():
#     #         fix2 = await svc.format_fixer.fix_format(
#     #             raw_output=json_tail2,
#     #             expected_format=getattr(schema_model, "__name__", str(schema_model)),
#     #             input_data=user_msg,
#     #             system_prompt=sys_prompt_retry,
#     #         )
#     #         if fix2.get("success"):
#     #             try:
#     #                 data = schema_model.model_validate(fix2["data"]).model_dump()
#     #             except Exception:
#     #                 data = None
#
#     if data is None:
#         try:
#             data = schema_model.model_validate({}).model_dump()
#         except Exception:
#             data = {}
#
#     try:
#         run_logger.log_step(
#             "streaming_with_structured_output",
#             {
#                 "error": error,
#                 "internal_chars": len(internal_thinking),
#                 "user_chars": len(user_thinking),
#                 "json_tail_chars": len(parser.json),
#                 "provider": getattr(cfg, "provider", None),
#                 "model": getattr(cfg, "model_name", None),
#                 "role": client_role,
#             },
#         )
#     except Exception:
#         pass
#
#     run_logger.finish_operation(
#         True,
#         "three_section_stream_complete",
#         result_preview={
#             "error": error,
#             "internal_len": len(internal_thinking),
#             "user_len": len(user_thinking),
#             "has_agent_response": bool(data),
#             "provider": getattr(cfg, "provider", None),
#             "model": getattr(cfg, "model_name", None),
#         },
#     )
#
#     return {
#         "agent_response": data,
#         "log": {
#             "error": error,
#             "raw_data": raw_data
#         },
#         "internal_thinking": internal_thinking,
#         "user_thinking": user_thinking,
#     }

async def _stream_agent_sections_to_json(
        svc: ModelServiceBase,
        *,
        client_name: str,
        client_role: str,
        sys_prompt: str,
        user_msg: str,
        schema_model,
        on_thinking_delta=None,
        temperature: float = 0.2,
        max_tokens: int = 1200,
        ctx: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Stream 3 sections using ONLY paranoid markers:
      1) <<< BEGIN INTERNAL THINKING >>> (capture only)
      2) <<< BEGIN USER-FACING THINKING >>> (stream via on_thinking_delta)
      3) <<< BEGIN STRUCTURED JSON >>> (strict; returned)

    The parser keeps a small rolling HOLDBACK so we never cut a marker across chunk
    boundaries, but otherwise streams promptly. Service markers are never emitted
    to the UI.
    """
    run_logger = AgentLogger("ThreeSectionStream", getattr(svc.config, "log_level", "INFO"))
    run_logger.start_operation(
        "three_section_stream",
        client_role=client_role,
        client_name=client_name,
        system_prompt_len=len(sys_prompt),
        user_msg_len=len(user_msg),
        expected_format=getattr(schema_model, "__name__", str(schema_model)),
    )

    deltas = 0

    async def _emit_user(piece: str, completed: bool = False, **kwargs):
        nonlocal deltas
        if not piece and not completed:
            return
        if on_thinking_delta is not None:
            await on_thinking_delta(piece or "", completed, **kwargs)
            if piece:
                deltas += 1

    parser = _SectionsStreamParser(on_user=_emit_user)

    async def on_delta(piece: str):
        await parser.feed(piece)

    async def on_complete(ret):
        await parser.finalize()
        # send the "completed" signal to the UI
        await _emit_user("", completed=True)
        run_logger.log_step("stream_complete", {"deltas": deltas})

    # ---- stream with model ----
    client = svc.get_client(client_name)
    cfg = svc.describe_client(client, role=client_role)

    messages = [SystemMessage(content=sys_prompt), HumanMessage(content=user_msg)]
    await svc.stream_model_text_tracked(
        client,
        messages,
        on_delta=on_delta,
        on_complete=on_complete,
        temperature=temperature,
        max_tokens=max_tokens,
        client_cfg=cfg,
        role=client_role,
    )

    internal_thinking = parser.internal
    user_thinking = parser.user
    json_tail = parser.json

    # ---- parse JSON (tolerate fenced block) ----
    def _try_parse_json(raw_json: str) -> Optional[Dict[str, Any]]:
        raw = (raw_json or "").strip()
        # handle fenced block if present
        if raw.startswith("```"):
            nl = raw.find("\n")
            raw = raw[nl + 1 :] if nl >= 0 else ""
            end = raw.rfind("```")
            if end >= 0:
                raw = raw[:end]
        raw = raw.strip().strip("`")
        if not raw:
            return None
        m = re.search(r"\{.*\}\s*$", raw, re.S)
        if not m:
            return None
        try:
            loaded = _json_loads_loose(m.group(0)) or {}
            return schema_model.model_validate(loaded).model_dump()
        except Exception:
            return None

    data = _try_parse_json(json_tail)
    raw_data = json_tail
    error = None

    if data is None and json_tail.strip():
        # try format fixer once
        fix = await svc.format_fixer.fix_format(
            raw_output=json_tail,
            expected_format=getattr(schema_model, "__name__", str(schema_model)),
            input_data=user_msg,
            system_prompt=sys_prompt,
        )
        if fix.get("success"):
            try:
                data = schema_model.model_validate(fix["data"]).model_dump()
            except Exception as ex:
                error = f"JSON parse after format fix failed: {ex}"
                data = None

    if data is None:
        try:
            data = schema_model.model_validate({}).model_dump()
        except Exception:
            data = {}

    try:
        run_logger.log_step(
            "streaming_with_structured_output",
            {
                "error": error,
                "internal_chars": len(internal_thinking),
                "user_chars": len(user_thinking),
                "json_tail_chars": len(parser.json),
                "provider": getattr(cfg, "provider", None),
                "model": getattr(cfg, "model_name", None),
                "role": client_role,
            },
        )
    except Exception:
        pass

    run_logger.finish_operation(
        True,
        "three_section_stream_complete",
        result_preview={
            "error": error,
            "internal_len": len(internal_thinking),
            "user_len": len(user_thinking),
            "has_agent_response": bool(data),
            "provider": getattr(cfg, "provider", None),
            "model": getattr(cfg, "model_name", None),
        },
    )

    return {
        "agent_response": data,
        "log": {
            "error": error,
            "raw_data": raw_data
        },
        "internal_thinking": internal_thinking,
        "user_thinking": user_thinking,
    }

# =============================================================================
# Simple (non-sectioned) structured-output streaming
# =============================================================================

async def _stream_simple_structured_json(
        svc: ModelServiceBase,
        *,
        client_name: str,
        client_role: str,
        sys_prompt: str,
        user_msg: str,
        schema_model,
        temperature: float = 0.2,
        max_tokens: int = 1200,
        ctx: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Stream plain text (no 3-part protocol), buffer everything, then parse it as JSON
    matching `schema_model`. If parsing fails, run the service format fixer once and
    validate again. Returns {agent_response, log:{error, raw_data}}.

    Prompting tip: tell the model "Return ONLY a JSON object ..." and (optionally)
    include a fenced `json` schema hint. The parser tolerates code fences in output.
    """
    run_logger = AgentLogger("SimpleStructuredStream", getattr(svc.config, "log_level", "INFO"))
    run_logger.start_operation(
        "simple_structured_stream",
        client_role=client_role,
        client_name=client_name,
        system_prompt_len=len(sys_prompt),
        user_msg_len=len(user_msg),
        expected_format=getattr(schema_model, "__name__", str(schema_model)),
    )

    raw_text_parts: List[str] = []

    async def on_delta(piece: str):
        if piece:
            raw_text_parts.append(piece)

    async def on_complete(_):
        run_logger.log_step("stream_complete", {"chars": sum(len(p) for p in raw_text_parts)})

    # ---- stream with model ----
    client = svc.get_client(client_name)
    cfg = svc.describe_client(client, role=client_role)
    messages = [SystemMessage(content=sys_prompt), HumanMessage(content=user_msg)]

    await svc.stream_model_text_tracked(
        client,
        messages,
        on_delta=on_delta,
        on_complete=on_complete,
        temperature=temperature,
        max_tokens=max_tokens,
        client_cfg=cfg,
        role=client_role,
    )

    raw_text = "".join(raw_text_parts)

    # ---- parse helpers ----
    def _strip_fences(s: str) -> str:
        s = s.strip()
        if s.startswith("```"):
            nl = s.find("\n")
            s = s[nl + 1:] if nl >= 0 else ""
            end = s.rfind("```")
            if end >= 0:
                s = s[:end]
        return s.strip().strip("`")

    def _extract_object_tail(s: str) -> Optional[str]:
        s = _strip_fences(s)
        if not s:
            return None
        m = re.search(r"\{.*\}\s*$", s, re.S)
        return m.group(0) if m else None

    def _try_parse(s: str) -> Optional[Dict[str, Any]]:
        obj = _extract_object_tail(s or "")
        if not obj:
            return None
        try:
            loaded = _json_loads_loose(obj) or {}
            return schema_model.model_validate(loaded).model_dump()
        except Exception:
            return None

    data = _try_parse(raw_text)
    error = None

    if data is None and raw_text.strip():
        # one-shot format fix
        fix = await svc.format_fixer.fix_format(
            raw_output=raw_text,
            expected_format=getattr(schema_model, "__name__", str(schema_model)),
            input_data=user_msg,
            system_prompt=sys_prompt,
        )
        if fix.get("success"):
            try:
                data = schema_model.model_validate(fix["data"]).model_dump()
            except Exception as ex:
                error = f"JSON parse after format fix failed: {ex}"
                data = None

    if data is None:
        # safest default
        try:
            data = schema_model.model_validate({}).model_dump()
        except Exception:
            data = {}

    try:
        run_logger.finish_operation(
            True,
            "simple_structured_stream_complete",
            result_preview={
                "error": error,
                "chars": len(raw_text),
                "has_agent_response": bool(data),
                "provider": getattr(cfg, "provider", None),
                "model": getattr(cfg, "model_name", None),
            },
        )
    except Exception:
        pass

    return {
        "agent_response": data,
        "log": {
            "error": error,
            "raw_data": raw_text,
        },
    }