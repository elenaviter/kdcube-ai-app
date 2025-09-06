# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

# chat/sdk/tools/llm_tools.py


import json
from typing import Annotated, Optional, List, Dict, Any

import semantic_kernel as sk
try:
    from semantic_kernel.functions import kernel_function
except Exception:
    from semantic_kernel.utils.function_decorator import kernel_function

# Bound at runtime by ToolManager (__init__ calls mod.bind_service(self.svc))
_SERVICE = None

def bind_service(svc):  # ToolManager will call this
    global _SERVICE
    _SERVICE = svc


class SummarizerTools:
    """
    LLM-backed summarizer with TWO explicit modes:

    1) input_mode="text"    → Summarize the `text` argument. Ignores sources_json and cite_sources.
    2) input_mode="sources" → Summarize a list of sources passed via `sources_json`
                              and (optionally) insert inline citation tokens [[S:<sid>]].

    Source schema (each item):
      {
        "sid": int,                 # local source id (1..N)
        "title": str,               # short title (used for grounding)
        "url": str,                 # optional, not emitted; used by downstream to link
        "text": str                 # main text/body to summarize (prefer this field)
      }

    Return:
      Markdown string. If input_mode="sources" and cite_sources=true, the string may contain tokens
      like [[S:1]] or [[S:1,3]] at the end of sentences/bullets. These are easy to post-process via
      regex:  r'\\[\\[S:(\\d+(?:,\\d+)*)\\]\\]'.
    """

    @kernel_function(
        name="summarize_llm",
        description=(
                "Summarize either free text (input_mode='text') or a list of sources (input_mode='sources'). "
                "In sources mode, may add inline citation tokens [[S:<sid>]] to mark provenance."
        )
    )
    async def summarize_llm(
            self,
            input_mode: Annotated[str, "text|sources", {"enum": ["text", "sources"]}] = "text",
            text: Annotated[str, "When input_mode='text': the text to summarize (≤10k chars)."] = "",
            sources_json: Annotated[str, "When input_mode='sources': JSON array of {sid,int; title,str; url,str; text,str}."] = "[]",
            style: Annotated[str, "brief|bullets|one_line", {"enum": ["brief","bullets","one_line"]}] = "brief",
            cite_sources: Annotated[bool, "In sources mode: insert [[S:<sid>]] tokens after claims."] = False,
            max_tokens: Annotated[int, "LLM output cap.", {"min": 64, "max": 800}] = 300,
    ) -> Annotated[str, "Markdown summary (string, may include [[S:<sid>]] tokens)."]:
        if _SERVICE is None:
            return "ERROR: summarizer not bound to service."

        from langchain_core.messages import SystemMessage, HumanMessage

        # --- normalize inputs ---
        if style not in ("brief", "bullets", "one_line"):
            style = "brief"
        mode = "sources" if input_mode == "sources" else "text"

        # ----- Build prompt -----
        # Base rules apply to both modes
        sys_lines = [
            "Summarize for a busy reader. Be factual and non-speculative.",
            "- style=brief   → one short paragraph (3–5 concise sentences).",
            "- style=bullets → 3–6 compact bullets.",
            "- style=one_line→ ≤ 28 words.",
            "No preface. Markdown only.",
        ]

        # Construct user payload depending on mode
        if mode == "text":
            # Summarize the provided text; ignore citation features entirely
            src_block = ""
            content = (text or "")[:10000]
            user = f"mode=text; style={style}\n\n{content}"

        else:
            # Summarize provided sources; optionally emit [[S:<sid>]] tokens
            # Parse sources_json and build a compact, bounded digest
            try:
                raw_sources = json.loads(sources_json) if sources_json else []
            except Exception:
                raw_sources = []

            # Normalize source rows
            rows: List[Dict[str, Any]] = []
            for s in raw_sources or []:
                if not isinstance(s, dict):  # skip garbage
                    continue
                sid = s.get("sid")
                title = s.get("title") or ""
                url = s.get("url") or s.get("href") or ""
                body = s.get("text") or s.get("body") or s.get("content") or ""
                if sid is None:
                    continue
                rows.append({"sid": sid, "title": title, "url": url, "text": body})

            # Bound total budget to 10k chars; distribute fairly across sources
            total_budget = 10000
            per = max(600, total_budget // max(1, len(rows)))  # ≥ 600 chars each if few sources
            parts = []
            for r in rows:
                t = (r["text"] or "")[:per]
                # Add a sid tag in the digest so the model can anchor claims
                parts.append(f"[sid:{r['sid']}] {r['title']}\n{t}".strip())
            digest = "\n\n---\n\n".join(parts)[:total_budget]

            # Add explicit citation rules only in sources mode
            if cite_sources:
                sys_lines += [
                    "CITATIONS:",
                    "- Insert inline citation tokens at the end of the sentence/bullet they support: [[S:<sid>]]",
                    "- Multiple sources allowed: [[S:1,3]]. Use only provided sid values; never invent.",
                    "- If a claim is general, you may omit a token.",
                ]

            # Provide a compact sid→title map to reduce hallucination
            compact_map = "\n".join([f"- {r['sid']}: {r['title'][:80]}" for r in rows]) if rows else ""
            src_block = f"SOURCE IDS:\n{compact_map}\n" if compact_map else ""
            user = f"mode=sources; style={style}; cite_sources={bool(cite_sources)}\n{src_block}\n{digest}"

        sys_prompt = "\n".join(sys_lines)

        # ----- stream infer -----
        buf: List[str] = []

        async def on_delta(piece: str):
            if piece:
                buf.append(piece)

        async def on_complete(_):  # noqa: ARG001
            pass

        await _SERVICE.stream_model_text_tracked(
            _SERVICE.get_client("tool.summarizer"),
            [SystemMessage(content=sys_prompt), HumanMessage(content=user)],
            on_delta=on_delta,
            on_complete=on_complete,
            temperature=0.2,
            max_tokens=max_tokens,
            client_cfg=_SERVICE.describe_client(_SERVICE.answer_generator_client, role="answer_generator"),
            role="answer_generator",
        )
        return "".join(buf).strip()


kernel = sk.Kernel()
tools = SummarizerTools()
kernel.add_plugin(tools, "agent_llm_tools")

print()
