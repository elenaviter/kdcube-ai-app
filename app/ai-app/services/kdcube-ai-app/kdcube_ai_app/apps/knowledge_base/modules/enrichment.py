# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

# kdcube_ai_app/apps/knowledge_base/modules/segment_enrichment_unified.py

from __future__ import annotations

import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from dataclasses import asdict

from pydantic import BaseModel, Field, ValidationError
from langchain_core.messages import SystemMessage, HumanMessage

from kdcube_ai_app.apps.knowledge_base.modules.base import ProcessingModule
from kdcube_ai_app.apps.knowledge_base.modules.contracts.segmentation import SegmentType
from kdcube_ai_app.infra.service_hub.inventory import ModelServiceBase, ConfigRequest, create_workflow_config


class ChunkMetadata(BaseModel):
    summary: str = Field(description="A concise 1-2 sentence summary of the chunk.")
    keywords: List[str] = Field(description="A list of 5-7 key topics or entities mentioned.")
    hypothetical_questions: List[str] = Field(description="A list of 3-5 questions this chunk could answer.")
    table_summary: Optional[str] = Field(default=None, description="If the chunk is a table, a natural language summary of its key insights.")


def create_embedding_text(meta: ChunkMetadata, content: str) -> str:
    kws = ", ".join(meta.keywords or [])
    body = content[:1000]
    return f"Summary: {meta.summary}\nKeywords: {kws}\nContent: {body}\n"


def _build_prompt(chunk_text: str, is_table: bool) -> Tuple[str, str]:
    sys = (
        "You are an expert analyst. Produce strictly valid JSON with fields: "
        "summary, keywords (5-7 strings), hypothetical_questions (3-5 strings), table_summary (optional). "
        "Do not include commentary. No markdown fences."
    )
    table_note = (
        "This chunk is a TABLE. Your summary should describe main data points and trends.\n"
        if is_table else ""
    )
    usr = f"{table_note}Chunk:\n---\n{chunk_text[:3000]}\n---\nReturn ONLY JSON."
    return sys, usr


class EnrichmentModule(ProcessingModule):
    """
    Stage 'enrichment': LLM-enrich retrieval segments.
    Outputs per-segment JSON, per-segment embedding text, and writes an 'enriched' retrieval set.
    """

    @property
    def stage_name(self) -> str:
        return "enrichment"

    async def process(self, resource_id: str, version: str, force_reprocess: bool = False, **kwargs) -> Dict[str, Any]:
        if not force_reprocess and self.is_processed(resource_id, version):
            self.logger.info(f"Enrichment already exists for {resource_id} v{version}, skipping")
            return self.get_results(resource_id, version) or {}

        # model_service can be passed in; else build minimal one from keys
        model_service: Optional[ModelServiceBase] = kwargs.get("model_service")
        if not model_service:
            req = ConfigRequest(
                openai_api_key=kwargs.get("openai_api_key"),
                claude_api_key=kwargs.get("claude_api_key"),
                selected_model=kwargs.get("selected_model") or "gpt-4o-mini",
                role_models={
                    "segment_enrichment": kwargs.get("segment_enrichment_role", {"provider": "anthropic", "model": "claude-3-7-sonnet-20250219"})
                }
            )
            cfg = create_workflow_config(req)
            model_service = ModelServiceBase(config=cfg)

        role_name = kwargs.get("role_name") or "segment_enrichment"

        # grab retrieval segments (default set)
        retr = self._read_retrieval_segments(resource_id, version)
        if not retr:
            raise ValueError("No retrieval segments found")

        enriched_records = []
        total = 0
        ok = 0

        # map base guid -> metadata (to check is_table/text_as_html)
        base = self._read_base_segments(resource_id, version)
        base_lut: Dict[str, Dict[str, Any]] = {b.get("guid"): b for b in base}

        enriched_retrieval_segments = []

        for i, comp in enumerate(retr):
            total += 1
            bguids = comp.get("base_segment_guids") or []
            bseg = base_lut.get(bguids[0]) if bguids else None
            if not bseg:
                continue

            text = bseg.get("text") or ""
            meta = (bseg.get("metadata") or {})
            is_table = bool(meta.get("is_table"))
            html = meta.get("text_as_html")
            content_for_llm = html if (is_table and isinstance(html, str) and html.strip()) else text

            sys, usr = _build_prompt(content_for_llm, is_table)

            # stream to the model; we ignore deltas, we want the final text and parse as JSON
            client = model_service.get_client(role_name, temperature=0.0)
            messages = [SystemMessage(content=sys), HumanMessage(content=usr)]
            result = await model_service.stream_model_text_tracked(client, messages, on_delta=lambda *_a, **_k: None)

            raw = result.get("text") or ""
            parsed: Optional[ChunkMetadata] = None
            parse_err = None

            # strict parse with Pydantic; if fails, call the built-in FormatFixer once
            try:
                data = json.loads(raw)
                parsed = ChunkMetadata.model_validate(data)
            except Exception as e:
                parse_err = str(e)
                fix = await model_service.format_fixer.fix_format(
                    raw_output=raw,
                    expected_format="ChunkMetadata",
                    input_data=usr,
                    system_prompt=sys
                )
                if fix.get("success"):
                    try:
                        parsed = ChunkMetadata.model_validate(fix["data"])
                    except Exception as ee:
                        parse_err = f"after fix: {ee}"

            if not parsed:
                # record failure, keep going
                rec = {
                    "segment_id": comp.get("guid"),
                    "success": False,
                    "error": parse_err or "unknown parse error",
                    "rn": f"ef:{self.tenant}:{self.project}:knowledge_base:{self.stage_name}:{resource_id}:{version}:segment:{comp.get('guid')}"
                }
                enriched_records.append(rec)
                self._save_enrichment_record(resource_id, version, comp.get("guid"), rec)
                continue

            ok += 1

            # compose embedding text and retrieval doc body
            emb_text = create_embedding_text(parsed, content_for_llm)
            retrieval_doc = self._compose_retrieval_doc(parsed, is_table, content_for_llm)

            rec = {
                "segment_id": comp.get("guid"),
                "success": True,
                "metadata": parsed.model_dump(),
                "is_table": is_table,
                "embedding_text": emb_text,
                "retrieval_doc": retrieval_doc,
                "rn": f"ef:{self.tenant}:{self.project}:knowledge_base:{self.stage_name}:{resource_id}:{version}:segment:{comp.get('guid')}"
            }
            enriched_records.append(rec)
            self._save_enrichment_record(resource_id, version, comp.get("guid"), rec)

            # also prepare an enriched retrieval segment that replaces its text with retrieval_doc
            comp_enriched = dict(comp)
            comp_enriched["enriched"] = True
            comp_enriched["text"] = retrieval_doc
            enriched_retrieval_segments.append(comp_enriched)

        # write an alternate retrieval set for the segmentator: segmentation/retrieval_enriched/segments.json
        self._write_enriched_retrieval(resource_id, version, enriched_retrieval_segments)

        summary = {
            "resource_id": resource_id,
            "version": version,
            "segments_total": total,
            "segments_enriched": ok,
            "segments_failed": total - ok,
            "role": role_name,
            "timestamp": datetime.now().isoformat(),
            "rn": f"ef:{self.tenant}:{self.project}:knowledge_base:{self.stage_name}:{resource_id}:{version}"
        }
        self.save_results(resource_id, version, summary)
        return summary

    def _read_base_segments(self, resource_id: str, version: str) -> List[Dict[str, Any]]:
        content = self.storage.get_stage_content("segmentation", resource_id, version, "segments.json", as_text=True)
        return json.loads(content) if content else []

    def _read_retrieval_segments(self, resource_id: str, version: str) -> List[Dict[str, Any]]:
        content = self.storage.get_stage_content("segmentation", resource_id, version, "segments.json",
                                                 subfolder=SegmentType.RETRIEVAL.value, as_text=True)
        return json.loads(content) if content else []

    def _save_enrichment_record(self, resource_id: str, version: str, seg_id: str, payload: Dict[str, Any]) -> None:
        self.storage.save_stage_content(self.stage_name, resource_id, version,
                                        f"segment_{seg_id}_enrichment.json", json.dumps(payload, indent=2))

    def _write_enriched_retrieval(self, resource_id: str, version: str, segments: List[Dict[str, Any]]) -> None:
        content = json.dumps(segments, indent=2)
        # keep enriched set under segmentation so downstream loaders can prefer it transparently
        self.storage.save_stage_content("segmentation", resource_id, version, "segments.json",
                                        content, subfolder=f"{SegmentType.RETRIEVAL.value}_enriched")

    def _compose_retrieval_doc(self, meta: ChunkMetadata, is_table: bool, content: str) -> str:
        lines = []
        lines.append(f"Summary: {meta.summary}")
        if meta.keywords:
            lines.append("Keywords: " + ", ".join(meta.keywords))
        if is_table and meta.table_summary:
            lines.append(f"Table summary: {meta.table_summary}")
        # include a trimmed body for recall; keep it modest to avoid overshadowing summary
        body = content[:1200]
        lines.append("Content:")
        lines.append(body)
        return "\n".join(lines)
