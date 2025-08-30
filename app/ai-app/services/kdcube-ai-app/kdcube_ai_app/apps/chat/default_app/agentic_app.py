# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

from __future__ import annotations

import aiohttp
import json
import asyncio
import logging
import time
from typing import List, Dict, Any, Optional, TypedDict, Annotated
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, AnyMessage
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from datetime import datetime

from kdcube_ai_app.apps.chat.emitters import ChatCommunicator
from kdcube_ai_app.apps.chat.sdk.inventory import Config, AgentLogger, _mid
from kdcube_ai_app.apps.chat.sdk.util import _json_schema_of
from kdcube_ai_app.infra.accounting import with_accounting

# Loader imports
from kdcube_ai_app.infra.plugin.agentic_loader import (
    agentic_initial_state,
    agentic_workflow,
)
from kdcube_ai_app.storage.storage import create_storage_backend


def _now_ms() -> int:
    return int(time.time() * 1000)


try:
    from .inventory import ThematicBotModelService, BUNDLE_ID, project_app_state, _history_to_seed_messages
    from .integrations.rag import RAGService
except ImportError:  # fallback when running as a script
    from inventory import ThematicBotModelService, BUNDLE_ID, project_app_state, _history_to_seed_messages
    from integrations.rag import RAGService


@agentic_initial_state(name=f"{BUNDLE_ID}-initial-state", priority=200)
def create_initial_state(payload: Dict[str, Any]):
    return {
        "messages": [],
        "summarized_messages": [],
        "context": {"bundle": BUNDLE_ID},
        "user_message": payload.get("user_message"),
        "is_our_domain": None,
        "classification_reasoning": None,
        "rag_queries": None,
        "retrieved_docs": None,
        "reranked_docs": None,
        "final_answer": None,
        "thinking": "",
        "followups": [],
        "error_message": None,
        "format_fix_attempts": 0,
        "search_hits": None,
        "execution_id": f"exec_{int(time.time() * 1000)}",
        "start_time": time.time(),
        "step_logs": [],
        "performance_metrics": {},
        # turn_id gets injected by the workflow before invocation
    }


# Import SummarizationNode
try:
    from langmem.short_term import SummarizationNode
except ImportError:
    print("Warning: langmem not available. Install with: pip install langmem")
    SummarizationNode = None

# ===========================================
# Pydantic Models for Structured Outputs
# ===========================================

class ClassificationOutput(BaseModel):
    is_our_domain: bool
    confidence: float
    reasoning: str


class QueryWriterOutput(BaseModel):
    chain_of_thought: Optional[str]
    queries: List[Dict[str, Any]]


class RerankingOutput(BaseModel):
    reranked_docs: List[Dict[str, Any]]
    reasoning: str


class FinalAnswerOutput(BaseModel):
    answer: str
    sources_used: List[str]
    confidence: float


# ===========================================
# Proper State Definition for LangGraph
# ===========================================

class ChatGraphState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    summarized_messages: List[AnyMessage]
    context: Dict[str, Any]
    user_message: str

    is_our_domain: Optional[bool]
    classification_reasoning: Optional[str]
    rag_queries: Optional[List[Dict[str, Any]]]
    retrieved_docs: Optional[List[Dict[str, Any]]]
    reranked_docs: Optional[List[Dict[str, Any]]]
    final_answer: Optional[str]

    thinking: Optional[str]           # streamed “thinking” markdown
    followups: Optional[List[str]]         # parsed from FOLLOWUP JSON

    error_message: Optional[str]
    format_fix_attempts: int

    search_hits: Optional[List[Dict[str, Any]]]

    execution_id: str
    start_time: float
    step_logs: List[Dict[str, Any]]
    performance_metrics: Dict[str, Any]

    turn_id: str


# ===========================================
# Utility
# ===========================================

def add_step_log(state: ChatGraphState, step: str, data: Dict[str, Any]):
    log_entry = {
        "step": step,
        "timestamp": datetime.now().isoformat(),
        "elapsed_time": f"{time.time() - state['start_time']:.2f}s",
        "data": data
    }
    state["step_logs"].append(log_entry)


def get_execution_summary(state: ChatGraphState) -> Dict[str, Any]:
    return {
        "execution_id": state["execution_id"],
        "total_time": f"{time.time() - state['start_time']:.2f}s",
        "total_steps": len(state["step_logs"]),
        "performance_metrics": state["performance_metrics"],
        "step_logs": state["step_logs"]
    }


# ===========================================
# Services / Agents
# ===========================================

class FormatFixerService:
    """Fixes malformed JSON responses using Claude"""

    def __init__(self, config: Config):
        self.config = config
        self.logger = AgentLogger(f"{BUNDLE_ID}.FormatFixer", config.log_level)
        try:
            import anthropic
            self.claude_client = anthropic.Anthropic(api_key=config.claude_api_key)
            self.logger.log_step("claude_client_initialized", {"model": config.format_fixer_model})
        except ImportError:
            self.claude_client = None
            self.logger.log_error(ImportError("anthropic package not available"), "Claude client initialization")

    async def fix_format(self, raw_output: str, expected_format: str, input_data: str, system_prompt: str) -> Dict[str, Any]:
        op = self.logger.start_operation("format_fixing",
                                         raw_output_length=len(raw_output),
                                         expected_format=expected_format,
                                         input_data_length=len(input_data))
        if not self.claude_client:
            msg = "Claude client not available"
            self.logger.finish_operation(False, msg)
            return {"success": False, "error": msg, "raw": raw_output}

        try:
            fix_prompt = (
                "You are a JSON format fixer. You receive malformed JSON output and need to fix it to match the expected format.\n\n"
                f"Original system prompt: {system_prompt}\n"
                f"Original input: {input_data}\n"
                f"Expected format: {expected_format}\n"
                f"Malformed output: {raw_output}\n\n"
                "Please fix the JSON to match the expected format. Return only the fixed JSON, no additional text."
            )

            resp = self.claude_client.messages.create(
                model=self.config.format_fixer_model,
                max_tokens=1000,
                messages=[{"role": "user", "content": fix_prompt}]
            )
            fixed_content = resp.content[0].text
            try:
                parsed = json.loads(fixed_content)
                self.logger.finish_operation(True, "Format fixing successful")
                return {"success": True, "data": parsed, "raw": fixed_content}
            except json.JSONDecodeError:
                self.logger.finish_operation(False, "Fixed content still invalid")
                return {"success": False, "error": "Fixed content is still not valid JSON", "raw": fixed_content}
        except Exception as e:
            self.logger.log_error(e, "Format fixing failed")
            self.logger.finish_operation(False, f"Format fixing failed: {e}")
            return {"success": False, "error": str(e), "raw": raw_output}


class KBSearchAgent:
    def __init__(self, config: Config):
        self.url = config.kb_search_url
        self.logger = AgentLogger(f"{BUNDLE_ID}.KBSearchAgent", config.log_level)

    async def search(self, query: str, top_k: int = 5, resource_id: Optional[str] = None):
        self.logger.start_operation("kb_search", query=query, top_k=top_k)
        payload = {"query": query, "top_k": top_k}
        if resource_id:
            payload["resource_id"] = resource_id
        try:
            async with aiohttp.ClientSession() as sess:
                async with sess.post(self.url, json=payload) as resp:
                    data = await resp.json()
            self.logger.finish_operation(True, result_summary=f"Got {data.get('total_results')} hits")
            return data["results"]
        except Exception as e:
            self.logger.log_error(e, "KB search failed")
            self.logger.finish_operation(False, "KB search failed")
            return []

    async def search_with_state(self, state: ChatGraphState) -> ChatGraphState:
        if not self.url:
            add_step_log(state, "kb_search_skipped", {"reason": "No KB URL configured"})
            return state
        try:
            results = await self.search(state["user_message"])
            state["search_hits"] = results
            add_step_log(state, "kb_search_completed", {
                "results_count": len(results),
                "query": state["user_message"]
            })
        except Exception as e:
            add_step_log(state, "kb_search_failed", {"error": str(e), "query": state["user_message"]})
            state["search_hits"] = []
        return state


class ClassifierAgent:
    def __init__(self, config: Config):
        self.config = config
        self.logger = AgentLogger(f"{BUNDLE_ID}.ClassifierAgent", config.log_level)
        self.model_service = ThematicBotModelService(config)
        self.format_fixer = FormatFixerService(config)

    async def classify(self, state: ChatGraphState) -> ChatGraphState:
        self.logger.start_operation("classify_query",
                                    user_message=state["user_message"][:100] + "..." if len(state["user_message"]) > 100 else state["user_message"],
                                    execution_id=state["execution_id"])
        if not self.config.has_classifier:
            state["is_our_domain"] = True
            state["classification_reasoning"] = f"Model {self.config.selected_model} processes all queries without classification"
            add_step_log(state, "classification", {"success": True, "skipped": True})
            self.logger.finish_operation(True, "Classification skipped")
            return state

        summary_context = ""
        if state["context"].get("running_summary"):
            summary_context = f"Conversation summary: {state['context']['running_summary']}\n\n"

        schema_json = _json_schema_of(ClassificationOutput)

        system_prompt = f"""You are a domain classifier for a Houseplants & Home Gardening assistant.
[... your domain guidance here ...]

Return a SINGLE JSON object that VALIDATES against the JSON Schema for type "{ClassificationOutput.__name__}".

JSON_SCHEMA[{ClassificationOutput.__name__}]:
{schema_json}

RULES:
- Output JSON only (no prose, no code fences).
- "confidence" must be in [0.0, 1.0].
- "reasoning" should be brief and concrete.
"""
        with with_accounting("chat.agentic.classifier", metadata={"message": state["user_message"]}):
            result = await self.model_service.call_model_with_structure(
                self.model_service.classifier_client,
                system_prompt,
                state["user_message"],
                ClassificationOutput,
                client_cfg=self.model_service.describe_client(self.model_service.classifier_client, role="classifier")
            )

        usage = result.get("usage", {})
        add_step_log(state, "usage", {"step": "classifier", "usage": usage})

        if result["success"]:
            classification_data = result["data"]
            state["is_our_domain"] = classification_data["is_our_domain"]
            state["classification_reasoning"] = classification_data["reasoning"]
            add_step_log(state, "classification", {
                "success": True,
                "is_our_domain": state["is_our_domain"],
                "confidence": classification_data["confidence"]
            })
            self.logger.finish_operation(True, "Classification ok")
        else:
            if state["format_fix_attempts"] < 3:
                state["format_fix_attempts"] += 1
                fix_result = await self.format_fixer.fix_format(
                    result["raw"], "ClassificationOutput", state["user_message"], system_prompt
                )
                if fix_result["success"]:
                    validated = ClassificationOutput.model_validate(fix_result["data"])
                    state["is_our_domain"] = validated.is_our_domain
                    state["classification_reasoning"] = validated.reasoning
                    add_step_log(state, "classification_fixed", {"success": True, "attempt": state["format_fix_attempts"]})
                    self.logger.finish_operation(True, "Classification fixed")
                else:
                    state["error_message"] = fix_result["error"]
                    state["is_our_domain"] = True
                    add_step_log(state, "classification_failed", {"success": False, "fallback_used": True})
                    self.logger.finish_operation(False, "Classification failed (fallback)")
            else:
                state["error_message"] = result["error"]
                state["is_our_domain"] = True
                add_step_log(state, "classification_failed", {"success": False, "max_attempts_reached": True})
                self.logger.finish_operation(False, "Classification failed")
        return state


class QueryWriterAgent:
    def __init__(self, config: Config):
        self.config = config
        self.logger = AgentLogger(f"{BUNDLE_ID}.QueryWriterAgent", config.log_level)
        self.model_service = ThematicBotModelService(config)
        self.format_fixer = FormatFixerService(config)

    async def write_queries(self, state: ChatGraphState) -> ChatGraphState:
        self.logger.start_operation("write_queries",
                                    user_message=state["user_message"][:100] + "..." if len(state["user_message"]) > 100 else state["user_message"],
                                    execution_id=state["execution_id"])
        schema_json = _json_schema_of(QueryWriterOutput)
        system_prompt = f"""You are a Query Writer for a RAG system.

Return a SINGLE JSON object that VALIDATES against the JSON Schema for type "{QueryWriterOutput.__name__}" shown below.
Do not include any extra text, explanations, code fences, or trailing commas. Output must be parseable JSON.

JSON_SCHEMA[{QueryWriterOutput.__name__}]:
{schema_json}

REQUIREMENTS:
- Populate "queries" with 3–6 diverse, non-overlapping items.
- Each item MUST include:
  - "query": string
  - "weight": number in [0.0, 1.0] (use higher for more important queries)
  - "reasoning": short, practical justification (1–2 sentences max)
- Sort "queries" by descending "weight".
- "chain_of_thought" is optional; if included keep it brief or set it to null.
"""
        with with_accounting("chat.agentic.query_writer", metadata={"message": state["user_message"]}):
            result = await self.model_service.call_model_with_structure(
                self.model_service.query_writer_client,
                system_prompt,
                state["user_message"],
                QueryWriterOutput,
                client_cfg=self.model_service.describe_client(self.model_service.query_writer_client, role="query_writer")
            )
        if result["success"]:
            data = result["data"]
            state["rag_queries"] = data["queries"]
            add_step_log(state, "query_generation", {
                "success": True,
                "query_count": len(state["rag_queries"]),
                "total_weight": sum(q.get("weight", 0.0) for q in state["rag_queries"])
            })
            self.logger.finish_operation(True, "queries ok")
        else:
            if state["format_fix_attempts"] < 3:
                state["format_fix_attempts"] += 1
                fix = await self.format_fixer.fix_format(result["raw"], "QueryWriterOutput", state["user_message"], system_prompt)
                if fix["success"]:
                    validated = QueryWriterOutput.model_validate(fix["data"])
                    state["rag_queries"] = validated.queries
                    add_step_log(state, "query_generation_fixed", {"success": True, "attempt": state["format_fix_attempts"]})
                    self.logger.finish_operation(True, "queries fixed")
                else:
                    state["rag_queries"] = [{
                        "query": state["user_message"], "weight": 1.0, "reasoning": "fallback - original message"
                    }]
                    add_step_log(state, "query_generation_failed", {"success": False, "fallback_used": True})
                    self.logger.finish_operation(False, "queries failed → fallback")
            else:
                state["rag_queries"] = [{
                    "query": state["user_message"], "weight": 1.0, "reasoning": "fallback - original message"
                }]
                add_step_log(state, "query_generation_failed", {"success": False, "max_attempts_reached": True})
                self.logger.finish_operation(False, "queries failed")
        return state


class RAGAgent:
    def __init__(self, config: Config):
        self.config = config
        self.logger = AgentLogger(f"{BUNDLE_ID}.RAGAgent", config.log_level)
        self.rag_service = RAGService(config)

    async def retrieve(self, state: ChatGraphState) -> ChatGraphState:
        self.logger.start_operation("retrieve_documents", execution_id=state["execution_id"],
                                    query_count=len(state["rag_queries"]) if state["rag_queries"] else 0)
        if not state["rag_queries"]:
            state["error_message"] = "No queries available for retrieval"
            add_step_log(state, "retrieval_failed", {"success": False, "error": state["error_message"]})
            self.logger.finish_operation(False, state["error_message"])
            return state
        try:
            docs = await self.rag_service.retrieve_documents(state["rag_queries"])
            state["retrieved_docs"] = docs
            add_step_log(state, "retrieval", {"success": True, "document_count": len(docs)})
            self.logger.finish_operation(True, f"retrieved {len(docs)}")
        except Exception as e:
            state["error_message"] = f"Document retrieval failed: {e}"
            state["retrieved_docs"] = []
            add_step_log(state, "retrieval_failed", {"success": False, "error": state["error_message"]})
            self.logger.finish_operation(False, state["error_message"])
        return state


class RerankingAgent:
    def __init__(self, config: Config):
        self.config = config
        self.logger = AgentLogger(f"{BUNDLE_ID}.RerankingAgent", config.log_level)
        self.model_service = ThematicBotModelService(config)
        self.format_fixer = FormatFixerService(config)

    async def rerank(self, state: ChatGraphState) -> ChatGraphState:
        self.logger.start_operation("rerank_documents", execution_id=state["execution_id"],
                                    document_count=len(state["retrieved_docs"]) if state["retrieved_docs"] else 0)
        if not state["retrieved_docs"]:
            state["reranked_docs"] = []
            add_step_log(state, "reranking_skipped", {"success": True, "reason": "no_documents"})
            self.logger.finish_operation(True, "no docs to rerank")
            return state

        schema_json = _json_schema_of(RerankingOutput)
        system_prompt = f"""You are a document reranking expert.

Return a SINGLE JSON object that VALIDATES against the JSON Schema for type "{RerankingOutput.__name__}" shown below.
Do not include any extra text, explanations, code fences, or trailing commas. Output must be parseable JSON.

JSON_SCHEMA[{RerankingOutput.__name__}]:
{schema_json}

INSTRUCTIONS:
- Given the user question and the retrieved documents, assign each a "relevance_score" in [0.0, 1.0] and a "ranking_position" (1 = most relevant).
- Sort the "reranked_docs" array by descending "relevance_score".
- "reasoning" should summarize the top factors that influenced the ranking (brief, not a full chain-of-thought).
"""
        docs_text = json.dumps(state["retrieved_docs"], indent=2)
        user_msg = f"User question: {state['user_message']}\n\nDocuments to rerank:\n{docs_text}"

        with with_accounting("chat.agentic.reranking", metadata={"message": user_msg}):
            result = await self.model_service.call_model_with_structure(
                self.model_service.reranker_client,
                system_prompt,
                user_msg,
                RerankingOutput,
                client_cfg=self.model_service.describe_client(self.model_service.reranker_client, role="reranker")
            )

        if result["success"]:
            data = result["data"]
            state["reranked_docs"] = data["reranked_docs"]
            add_step_log(state, "reranking", {
                "success": True,
                "reranked_count": len(state["reranked_docs"]),
                "avg_relevance_score": (
                    sum(d.get("relevance_score", 0.0) for d in state["reranked_docs"]) / len(state["reranked_docs"])
                    if state["reranked_docs"] else 0.0
                )
            })
            self.logger.finish_operation(True, "rerank ok")
        else:
            state["reranked_docs"] = []
            for i, doc in enumerate(state["retrieved_docs"]):
                dc = dict(doc)
                dc["relevance_score"] = 1.0 - (i * 0.1)
                dc["ranking_position"] = i + 1
                state["reranked_docs"].append(dc)
            add_step_log(state, "reranking_failed", {"success": False, "fallback_used": True})
            self.logger.finish_operation(False, "rerank failed → fallback")
        return state


class AnswerGeneratorAgent:
    def __init__(self, config: Config, delta_emitter, step_emitter, streaming: bool = True):
        self.config = config
        self.logger = AgentLogger(f"{BUNDLE_ID}.AnswerGeneratorAgent", config.log_level)
        self.model_service = ThematicBotModelService(config)
        self.emit_delta = delta_emitter or (lambda *_: asyncio.sleep(0))
        self.emit_step = step_emitter or (lambda *_: asyncio.sleep(0))
        self.streaming = streaming

    async def generate_answer(self, state: ChatGraphState) -> ChatGraphState:
        self.logger.start_operation(
            "generate_answer",
            execution_id=state["execution_id"],
            is_our_domain=state["is_our_domain"],
            document_count=len(state["reranked_docs"]) if state["reranked_docs"] else 0
        )

        # --- Build context snippet block (unchanged) ---
        context_docs = ""
        doc_sources = []
        if state["reranked_docs"]:
            for i, doc in enumerate(state["reranked_docs"][:5]):
                context_docs += f"Document {i+1}:\n{doc.get('content','')}\n\n"
                src = (doc.get("metadata") or {}).get("source")
                if src: doc_sources.append(src)

        summary_context = ""
        if state["context"].get("running_summary"):
            summary_context = f"Previous conversation summary: {state['context']['running_summary']}\n\n"

        SUGGESTION_RULES = """
FOLLOW-UP SUGGESTIONS (strict):
- Phrased as first-person, user-side imperatives: executable as-is (e.g., "Learn more about plating begonia in winter").
- Start with a strong verb; no "you", no questions, no "please".
- No generic meta-asks such as: "Specify your needs", "Provide more details", "What else can I help with".
- ≤ 120 characters each, end with a period.
- Mirror the user’s language and context.
- 0–3 items. If the turn is only greetings/capabilities → return [].
"""
        # --- Output protocol & guardrails ---
        system_prompt = (
            f"{summary_context}You are a helpful assistant. Use the provided context snippets if relevant; otherwise answer from general knowledge.\n\n"
            "STYLE:\n"
            "- Be direct, structured, and actionable. Use short sections or lists.\n"
            "- Only cite the provided snippets if you actually used them.\n"
            "- The THINKING section must be high-level (key considerations), not step-by-step chain-of-thought.\n\n"
            "OUTPUT PROTOCOL (strict):\n"
            "1) Write exactly this marker on its own line:\n"
            "<HERE GOES THINKING PART>\n"
            "Then a brief high-level plan/rationale in Markdown (bullets/short lines; no numbered reasoning steps).\n"
            "2) Then this marker on its own line:\n"
            "<HERE GOES ANSWER FOR USER>\n"
            "Then the final user-facing answer in Markdown.\n"
            "3) Then this marker on its own line:\n"
            "<HERE GOES FOLLOWUP>\n"
            "{ \"followups\": [ /* 0–3 concise, user-imperative actions; or [] */ ] }\n"
            "Rules for followups: start with a verb; no questions; <=120 chars; no emojis.\n"
            "Return ONLY these three sections in this exact order.\n"
            f"{SUGGESTION_RULES}"
        )

        user_content = (
            f"Context snippets (may be empty):\n{context_docs}\n"
            f"User question:\n{state.get('user_message','')}"
        )

        # --- Streaming parser: THINKING -> ANSWER -> FOLLOWUP(JSON) ---
        import re, json as _json

        THINK_RE = re.compile(r"<\s*here\s+goes\s+thinking\s+part\s*>", re.I)
        ANS_RE   = re.compile(r"<\s*here\s+goes\s+answer\s+for\s+user\s*>", re.I)
        FUP_RE   = re.compile(r"<\s*here\s+goes\s+followup\s*>", re.I)

        MAX_BASE = max(len("here goes thinking part"),
                       len("here goes answer for user"),
                       len("here goes followup"))
        HOLDBACK = MAX_BASE + 8  # tolerate split markers

        buf = ""
        tail = ""
        mode = "pre"     # pre -> thinking -> answer -> followup
        emit_from = 0
        deltas = 0
        thinking_text = ""
        answer_text = ""

        def _skip_ws(i: int) -> int:
            while i < len(buf) and buf[i] in (" ", "\t", "\r", "\n"):
                i += 1
            return i

        async def _emit(kind: str, text: str):
            nonlocal deltas, thinking_text, answer_text
            if not text:
                return
            idx = deltas
            await self.emit_delta(text, idx, {"marker": ("thinking" if kind == "thinking" else "answer")})
            deltas += 1
            if kind == "thinking":
                thinking_text += text
            else:
                answer_text += text

        def _find(pat: re.Pattern, start_hint: int):
            start = max(0, start_hint - HOLDBACK)
            return pat.search(buf, start)

        async def on_delta(piece: str):
            nonlocal buf, mode, emit_from, tail
            if not piece:
                return
            prev_len = len(buf)
            buf += piece

            while True:
                if mode == "pre":
                    m = _find(THINK_RE, prev_len)
                    if not m: break
                    emit_from = _skip_ws(m.end())
                    mode = "thinking"

                if mode == "thinking":
                    m = _find(ANS_RE, prev_len)
                    if m and m.start() >= emit_from:
                        chunk = buf[emit_from:m.start()].rstrip()
                        await _emit("thinking", chunk)
                        emit_from = _skip_ws(m.end())
                        mode = "answer"
                        continue
                    safe_end = max(emit_from, len(buf) - HOLDBACK)
                    if safe_end > emit_from:
                        await _emit("thinking", buf[emit_from:safe_end])
                        emit_from = safe_end
                    break

                if mode == "answer":
                    m = _find(FUP_RE, prev_len)
                    if m and m.start() >= emit_from:
                        chunk = buf[emit_from:m.start()].rstrip()
                        await _emit("answer", chunk)
                        fup_start = _skip_ws(m.end())
                        tail = buf[fup_start:]
                        mode = "followup"
                        break
                    safe_end = max(emit_from, len(buf) - HOLDBACK)
                    if safe_end > emit_from:
                        await _emit("answer", buf[emit_from:safe_end])
                        emit_from = safe_end
                    break

                if mode == "followup":
                    tail += piece
                    break

                break

        try:
            usage_out: Dict[str, Any] = {}
            with with_accounting("chat.agentic.answer_generator", metadata={"message": state["user_message"]}):
                if self.streaming:
                    await self.model_service.stream_model_text_tracked(
                        self.model_service.answer_generator_client,
                        [
                            SystemMessage(content=system_prompt, id=_mid("sys")),
                            HumanMessage(content=user_content, id=_mid("user")),
                        ],
                        on_delta=on_delta,
                        temperature=0.3,
                        max_tokens=2000,
                        client_cfg=self.model_service.describe_client(
                            self.model_service.answer_generator_client, role="answer_generator"
                        ),
                    )
                else:
                    # Non-streaming fallback: just call and then parse once
                    res = await self.model_service.call_model_text(
                        self.model_service.answer_generator_client,
                        [
                            SystemMessage(content=system_prompt, id=_mid("sys")),
                            HumanMessage(content=user_content, id=_mid("user")),
                        ],
                        temperature=0.3,
                        max_tokens=2000,
                        client_cfg=self.model_service.describe_client(
                            self.model_service.answer_generator_client, role="answer_generator"
                        ),
                    )
                    await on_delta(res["text"])

            # Final flush (if stream ended mid-section)
            if mode == "thinking" and emit_from < len(buf):
                await _emit("thinking", buf[emit_from:].rstrip())
            elif mode == "answer" and emit_from < len(buf):
                await _emit("answer", buf[emit_from:].rstrip())

            # Parse followups JSON (tolerant)
            followups: List[str] = []
            if tail:
                raw = tail.strip().strip("`").lstrip(">")
                m = re.search(r"\{.*\}\s*$", raw, re.S)
                if m:
                    try:
                        obj = _json.loads(m.group(0))
                        vals = obj.get("followups") or obj.get("followup") or []
                        if isinstance(vals, list):
                            followups = [str(v).strip() for v in vals if str(v).strip()]
                    except Exception:
                        pass

            # Persist to state
            state["final_answer"] = answer_text or (buf.strip() if buf else "")
            state["thinking"] = (thinking_text or "").strip() or None
            state["followups"] = followups

            if followups:
                await self.emit_step(
                    "followups",
                    "completed",
                    {"items": followups, "turn_id": state.get("turn_id")},
                    title="Follow-ups"
                )
            else:
                await self.emit_step(
                    "followups",
                    "skipped",
                    {"reason": "none", "turn_id": state.get("turn_id")},
                    title="Follow-ups"
                )

            # Record usage marker (optional)
            add_step_log(state, "answer_generation_usage", {"usage": usage_out, "followups": followups})

            # Keep the assistant message as the visible ANSWER (not the thinking)
            ai_msg = AIMessage(content=state["final_answer"], id=_mid("ai"))
            state["messages"].append(ai_msg)

            self.logger.finish_operation(True, f"Generated answer, {len(state['final_answer'])} chars; followups={len(followups)}")
        except Exception as e:
            state["error_message"] = f"Answer generation failed: {e}"
            state["final_answer"] = "I encountered an error generating the response."
            add_step_log(state, "answer_generation_failed", {"success": False, "error": state["error_message"]})
            self.logger.finish_operation(False, state["error_message"])
        return state

# ===========================================
# LangGraph Workflow (Communicator-based)
# ===========================================

@agentic_workflow(name=f"{BUNDLE_ID}", version="1.0.0", priority=150)
class ChatWorkflow:
    """Main workflow orchestrator using ChatCommunicator for emissions"""

    def __init__(self, config: Config, communicator: ChatCommunicator, streaming: bool = True):
        self.config = config
        self.logger = AgentLogger(f"{BUNDLE_ID}.ChatWorkflow", config.log_level)

        # unified communicator (async)
        self.comm = communicator

        # current turn id (propagated in step/delta envelopes via communicator.conversation.turn_id)
        self._turn_id: Optional[str] = None

        # emit helpers
        async def _emit_step(step_name: str, status: str, payload: dict | None = None, title: Optional[str] = None):
            await self.comm.step(step=step_name, status=status, title=title, data=payload or {})

        async def _emit_delta(text: str, idx: int, meta: dict | None = None):
            marker = (meta or {}).get("marker", "answer")
            await self.comm.delta(text=text, index=idx, marker=marker)

        self.emit_step = _emit_step
        self.emit_delta = _emit_delta

        # agents
        self.classifier = ClassifierAgent(config)
        self.query_writer = QueryWriterAgent(config)
        self.rag_agent = RAGAgent(config)
        self.reranking_agent = RerankingAgent(config)
        self.answer_generator = AnswerGeneratorAgent(config,
                                                     delta_emitter=self.emit_delta,
                                                     step_emitter=self.emit_step,
                                                     streaming=streaming)
        self.kb_search_agent = KBSearchAgent(config)

        # memory + graph
        self.memory = MemorySaver()
        self.graph = self._build_graph()

        self.logger.log_step("workflow_initialized", {
            "selected_model": config.selected_model,
            "has_classifier": config.has_classifier,
            "provider": config.provider,
            "embedding_type": "custom" if config.custom_embedding_endpoint else "openai",
            "kb_search_available": bool(config.kb_search_url)
        })

    def set_state(self, state: Dict[str, Any]):
        self._app_state = dict(state or {})
        self._turn_id = self._app_state.get("turn_id")

    async def run(self, **params) -> Dict[str, Any]:
        # keep the turn id coming from the handler/processor (communicator already carries it)
        self._turn_id = self._turn_id or _mid("turn")

        text = (params.get("text") or self._app_state.get("text") or "").strip()
        thread_id = self._app_state.get("conversation_id") or "default"

        initial_state = create_initial_state({"user_message": text})
        initial_state["turn_id"] = self._turn_id

        seed = _history_to_seed_messages(self._app_state.get("history"))
        if seed:
            initial_state["messages"].extend(seed)

        result = await self.graph.ainvoke(
            initial_state,
            config={"configurable": {"thread_id": thread_id}},
        )
        return project_app_state(result)

    # ----- step payload helpers -----

    def _step_start_payload(self, state: ChatGraphState, step: str, turn_id: str) -> dict:
        return {
            "classifier": {"message": "Classifying domain...", "turn_id": turn_id},
            "query_writer": {"message": "Generating RAG queries...", "turn_id": turn_id},
            "rag_retrieval": {"message": "Retrieving documents...", "turn_id": turn_id},
            "reranking": {"message": "Reranking documents...", "turn_id": turn_id},
            "answer_generator": {"message": "Generating final answer...", "turn_id": turn_id},
        }.get(step, {"turn_id": turn_id})

    def _step_end_payload(self, state: ChatGraphState, step: str, turn_id: str) -> dict:
        if step == "classifier":
            return {"is_our_domain": state.get("is_our_domain"),
                    "message": state.get("classification_reasoning"), "turn_id": turn_id}
        if step == "query_writer":
            return {"query_count": len(state.get("rag_queries") or []),
                    "queries": [q["query"] for q in (state.get("rag_queries") or [])][:6],
                    "turn_id": turn_id}
        if step == "rag_retrieval":
            return {"retrieved_count": len(state.get("retrieved_docs") or []),
                    "kb_search_results": {
                        "results": (state.get("retrieved_docs") or [])[:10],
                        "total_results": len(state.get("retrieved_docs") or []),
                        "query": state.get("user_message"),
                        "turn_id": turn_id
                    }}
        if step == "reranking":
            reranked = state.get("reranked_docs") or []
            avg_rel = sum(d.get("relevance_score", 0.0) for d in reranked) / len(reranked) if reranked else 0.0
            return {"avg_relevance": avg_rel, "reranked_count": len(reranked)}
        if step == "answer_generator":
            ans = (state.get("final_answer") or "")
            return {
                "answer_length": len(ans),
                "followups": state.get("followups") or [],
                "has_thinking": bool(state.get("thinking")),
                "thinking_preview": (state.get("thinking") or "")[:160]
            }
        return {"turn_id": turn_id}

    def _wrap_node(self, fn, step_name: str):
        async def _wrapped(state: ChatGraphState) -> ChatGraphState:
            await self.emit_step(step_name, "started", self._step_start_payload(state, step_name, self._turn_id))
            try:
                out = fn(state)
                if asyncio.iscoroutine(out):
                    out = await out
                await self.emit_step(step_name, "completed", self._step_end_payload(out, step_name, self._turn_id))
                return out
            except Exception as e:
                await self.emit_step(step_name, "error", {"error": str(e), "turn_id": self._turn_id})
                raise
        return _wrapped

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(ChatGraphState)

        def add_user_message(state: ChatGraphState) -> Dict[str, Any]:
            msg = HumanMessage(content=state["user_message"], id=_mid("user"))
            return {"messages": [msg]}

        if SummarizationNode:
            chat_model = init_chat_model(f"openai:{self.config.answer_generator_model}")
            summarization_model = chat_model.bind(max_tokens=256)
            summarization_node = SummarizationNode(
                model=summarization_model,
                max_tokens=1000,
                max_tokens_before_summary=500,
                max_summary_tokens=200,
                input_messages_key="messages",
                output_messages_key="summarized_messages",
                token_counter=lambda msgs: sum(len(str(msg.content)) for msg in msgs),
            )
            workflow.add_node("summarize", summarization_node)
        else:
            def simple_summarize(state: ChatGraphState) -> ChatGraphState:
                state["summarized_messages"] = state["messages"][-10:]
                return state
            workflow.add_node("summarize", simple_summarize)

        if self.config.has_classifier:
            workflow.add_node("classifier", self._wrap_node(self.classifier.classify, "classifier"))

        workflow.add_node("query_writer", self._wrap_node(self.query_writer.write_queries, "query_writer"))
        workflow.add_node("rag_retrieval", self._wrap_node(self.rag_agent.retrieve, "rag_retrieval"))
        workflow.add_node("reranking", self._wrap_node(self.reranking_agent.rerank, "reranking"))
        workflow.add_node("answer_generator", self._wrap_node(self.answer_generator.generate_answer, "answer_generator"))

        workflow.add_node("workflow_start", self._wrap_node(add_user_message, "workflow_start"))
        workflow.add_edge(START, "workflow_start")
        workflow.add_edge("workflow_start", "summarize")

        if self.config.has_classifier:
            workflow.add_edge("summarize", "classifier")
            workflow.add_edge("classifier", "query_writer")
        else:
            workflow.add_edge("summarize", "query_writer")

        workflow.add_edge("query_writer", "rag_retrieval")
        workflow.add_edge("rag_retrieval", "reranking")
        workflow.add_edge("reranking", "answer_generator")

        async def _emit_workflow_complete(state: ChatGraphState) -> ChatGraphState:
            await self.emit_step(
                "workflow_complete",
                "completed",
                {
                    "message": "Workflow complete",
                    "turn_id": self._turn_id,
                    "followups": state.get("followups") or [],    # ← NEW
                    "has_thinking": bool(state.get("thinking"))   # ← NEW
                }
            )
            return state

        workflow.add_node("workflow_complete", _emit_workflow_complete)
        workflow.add_edge("answer_generator", "workflow_complete")
        workflow.add_edge("workflow_complete", END)

        return workflow.compile(checkpointer=self.memory)

    # Convenience API (optional)

    async def get_conversation_history(self, thread_id: str = "default") -> List[AnyMessage]:
        try:
            state = await self.graph.aget_state(config={"configurable": {"thread_id": thread_id}})
            return state.values.get("messages", []) if state.values else []
        except Exception as e:
            self.logger.log_error(e, f"Failed to get conversation history for thread {thread_id}")
            return []

    async def get_conversation_summary(self, thread_id: str = "default") -> str:
        try:
            state = await self.graph.aget_state(config={"configurable": {"thread_id": thread_id}})
            if state.values and state.values.get("context"):
                running_summary = state.values["context"].get("running_summary")
                return str(running_summary) if running_summary else ""
            return ""
        except Exception as e:
            self.logger.log_error(e, f"Failed to get conversation summary for thread {thread_id}")
            return ""

    async def get_execution_logs(self, thread_id: str = "default") -> List[Dict[str, Any]]:
        try:
            state = await self.graph.aget_state(config={"configurable": {"thread_id": thread_id}})
            return state.values.get("step_logs", []) if state.values else []
        except Exception as e:
            self.logger.log_error(e, f"Failed to get execution logs for thread {thread_id}")
            return []

    def _get_workflow_node_names(self) -> List[str]:
        nodes = ["workflow_start", "summarize", "query_writer", "rag_retrieval", "reranking", "answer_generator"]
        if self.config.has_classifier:
            nodes.insert(2, "classifier")
        return nodes

    def suggestions(self):
        return [
            "What light, watering, and soil do my common houseplants need?",
            "Why are my leaves yellow/brown/curling, and how do I fix it?",
            "How can I prevent and treat pests like spider mites and fungus gnats?",
            "When should I repot, and what potting mix should I use?"
        ]


# ===========================================
# Example usage (runs standalone with a no-op communicator)
# ===========================================

async def example_usage():
    from kdcube_ai_app.infra.accounting.envelope import AccountingEnvelope, bind_accounting
    from kdcube_ai_app.infra.accounting import with_accounting
    import os

    class _NoopEmitter:
        async def emit(self, event: str, data: dict, *, room=None, target_sid=None, session_id=None):
            return None

    # minimal no-op communicator so this script can run
    comm = ChatCommunicator(
        emitter=_NoopEmitter(),
        service={"request_id": "demo", "tenant": "demo", "project": "demo", "user": "user_123"},
        conversation={"session_id": "session_456", "conversation_id": "conv_456", "turn_id": _mid("turn")},
    )

    user_id = "user_123"
    session_id = "session_456"
    tenant_id = "home"
    project_id = "test-project"
    COMPONENT = BUNDLE_ID
    acct_dict = {
        "user_id": user_id,
        "session_id": session_id,
        "tenant_id": tenant_id,
        "project_id": project_id,
        "component": COMPONENT,
        "metadata": {"description": "Chat App Workflow Example", "tags": ["demo", "chat", "workflow"]}
    }

    envelope = AccountingEnvelope.from_dict(acct_dict)
    kdcube_path = os.environ.get("KDCUBE_STORAGE_PATH")
    storage_backend = create_storage_backend(kdcube_path, **{})

    openai_api_key = os.environ.get("OPENAI_API_KEY")
    claude_api_key = os.environ.get("ANTHROPIC_API_KEY")
    embedding_model = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")

    config = Config(selected_model="gpt-4o",
                    openai_api_key=openai_api_key,
                    claude_api_key=claude_api_key,
                    embedding_model=embedding_model)

    workflow = ChatWorkflow(config, communicator=comm)

    thread_id = "user_123_conversation"

    async with bind_accounting(envelope, storage_backend, enabled=True):
        prompt = "What are recent incidents on jail-breaking the enterprise agents?"
        async with with_accounting(COMPONENT, metadata={"prompt": prompt}):
            result1 = await workflow.run(text=prompt)
    print(f"Response 1: {result1.get('final_answer','')[:300]}...")
    print(f"Step logs present? {bool(result1.get('step_logs'))}")


async def example_conversation():
    from kdcube_ai_app.infra.accounting.envelope import AccountingEnvelope, bind_accounting
    from kdcube_ai_app.infra.accounting import with_accounting
    import os

    class _NoopEmitter:
        async def emit(self, event: str, data: dict, *, room=None, target_sid=None, session_id=None):
            return None

    comm = ChatCommunicator(
        emitter=_NoopEmitter(),
        service={"request_id": "demo", "tenant": "demo", "project": "demo", "user": "user_123"},
        conversation={"session_id": "session_456", "conversation_id": "conv_456", "turn_id": _mid("turn")},
    )

    user_id = "user_123"
    session_id = "session_456"
    tenant_id = "home"
    project_id = "test-project"
    COMPONENT = BUNDLE_ID
    acct_dict = {
        "user_id": user_id,
        "session_id": session_id,
        "tenant_id": tenant_id,
        "project_id": project_id,
        "component": COMPONENT,
        "metadata": {"description": "Chat App Workflow Example", "tags": ["demo", "chat", "workflow"]}
    }

    envelope = AccountingEnvelope.from_dict(acct_dict)
    kdcube_path = os.environ.get("KDCUBE_STORAGE_PATH")
    storage_backend = create_storage_backend(kdcube_path, **{})

    openai_api_key = os.environ.get("OPENAI_API_KEY")
    claude_api_key = os.environ.get("ANTHROPIC_API_KEY")
    embedding_model = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")

    config = Config(selected_model="gpt-4o",
                    openai_api_key=openai_api_key,
                    claude_api_key=claude_api_key,
                    embedding_model=embedding_model)

    workflow = ChatWorkflow(config, communicator=comm)

    thread_id = "user_123_conversation"

    async with bind_accounting(envelope, storage_backend, enabled=True):
        prompt = "What are recent incidents on jail-breaking the enterprise agents?"
        async with with_accounting(COMPONENT, metadata={"prompt": prompt}):
            result1 = await workflow.run(text=prompt)
    print(f"Response 1: {result1.get('final_answer','')[:300]}...")

    prompt = "How can I mitigate the risk caused by the first incident you mentioned?"
    async with with_accounting(COMPONENT, metadata={"prompt": prompt}):
        result2 = await workflow.run(text=prompt)
    print(f"Response 2: {result2.get('final_answer','')[:300]}...")


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(), logging.FileHandler('agent_execution.log', mode='a')]
    )


if __name__ == "__main__":
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())
    setup_logging()
    asyncio.run(example_usage())
