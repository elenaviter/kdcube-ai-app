# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

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

from kdcube_ai_app.apps.chat.emitters import StepEmitter, DeltaEmitter
from kdcube_ai_app.apps.chat.inventory import Config, AgentLogger, _mid
from kdcube_ai_app.infra.accounting import with_accounting

# Loader imports
from kdcube_ai_app.infra.plugin.agentic_loader import agentic_initial_state, agentic_workflow_factory, agentic_workflow
from kdcube_ai_app.storage.storage import create_storage_backend

try:
    from .inventory import ThematicBotModelService, BUNDLE_ID, project_app_state
    from .integrations.rag import RAGService
except ImportError:  # fallback when running as a script
    from inventory import ThematicBotModelService
    from integrations.rag import RAGService

@agentic_initial_state(name=f"{BUNDLE_ID}-initial-state", priority=200)
def create_initial_state(user_message: str):
    """
    Start state for the LangGraph workflow. You can keep it identical to your current
    create_initial_state(...) or slightly extend it with bundle-specific context.
    """
    import time

    return {
        "messages": [],
        "summarized_messages": [],
        "context": {"bundle": BUNDLE_ID},
        "user_message": user_message,
        "is_our_domain": None,
        "classification_reasoning": None,
        "rag_queries": None,
        "retrieved_docs": None,
        "reranked_docs": None,
        "final_answer": None,
        "error_message": None,
        "format_fix_attempts": 0,
        "search_hits": None,
        "execution_id": f"exec_{int(time.time() * 1000)}",
        "start_time": time.time(),
        "step_logs": [],
        "performance_metrics": {},
    }

# @agentic_workflow_factory(
#     name=f"{BUNDLE_ID}-factory",
#     version="1.0.0",
#     priority=300,     # wins over other decorated items with lower priority
#     singleton=True,   # bundle prefers singleton (can be overridden by spec/env)
# )
# def create_workflow(config: Config, step_emitter=None, delta_emitter=None):
#     """
#     Factory that returns a workflow instance. The loader calls this.
#     You can inject your custom RAG, logging, etc. here if your Workflow accepts it.
#     """
#
#     workflow = ChatWorkflow(
#         config,
#         step_emitter=step_emitter,
#         delta_emitter=delta_emitter
#     )
#     return workflow


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
    """Output format for the classification agent"""
    is_our_domain: bool = Field(description="Whether this problem should be solved by our specialized model")
    confidence: float = Field(description="Confidence score from 0.0 to 1.0")
    reasoning: str = Field(description="Explanation of the classification decision")

class QueryWriterOutput(BaseModel):
    """Output format for the query writer agent"""
    chain_of_thought: str = Field(description="Reasoning process for breaking down the user question")
    queries: List[Dict[str, Any]] = Field(description="List of queries with weights")

class RerankingOutput(BaseModel):
    """Output format for the reranking agent"""
    reranked_docs: List[Dict[str, Any]] = Field(description="Documents reranked by relevance with scores")
    reasoning: str = Field(description="Explanation of reranking decisions")

class FinalAnswerOutput(BaseModel):
    """Final response from the model"""
    answer: str = Field(description="Final answer to the user question")
    sources_used: List[str] = Field(description="List of sources/documents used in the answer")
    confidence: float = Field(description="Confidence score from 0.0 to 1.0")

# ===========================================
# Proper State Definition for LangGraph
# ===========================================

class ChatGraphState(TypedDict):
    """State for the LangGraph workflow - must be TypedDict for proper serialization"""

    # Core conversation data - these will be used by SummarizationNode
    messages: Annotated[List[AnyMessage], add_messages]  # Original messages (input to summarization)
    summarized_messages: List[AnyMessage]  # Output from summarization (input to LLM)

    # Context for summarization node
    context: Dict[str, Any]  # Contains running_summary and other context

    # Current user message being processed
    user_message: str

    # Processing state
    is_our_domain: Optional[bool]
    classification_reasoning: Optional[str]
    rag_queries: Optional[List[Dict[str, Any]]]
    retrieved_docs: Optional[List[Dict[str, Any]]]
    reranked_docs: Optional[List[Dict[str, Any]]]
    final_answer: Optional[str]
    error_message: Optional[str]
    format_fix_attempts: int

    # Search integration
    search_hits: Optional[List[Dict[str, Any]]]

    # Execution tracking
    execution_id: str
    start_time: float
    step_logs: List[Dict[str, Any]]
    performance_metrics: Dict[str, Any]

# ===========================================
# Utility Functions for State Management
# ===========================================

def add_step_log(state: ChatGraphState, step: str, data: Dict[str, Any]):
    """Add a step log to the state - utility function since TypedDict cannot have methods"""
    log_entry = {
        "step": step,
        "timestamp": datetime.now().isoformat(),
        "elapsed_time": f"{time.time() - state['start_time']:.2f}s",
        "data": data
    }
    state["step_logs"].append(log_entry)

def get_execution_summary(state: ChatGraphState) -> Dict[str, Any]:
    """Get execution summary from state"""
    return {
        "execution_id": state["execution_id"],
        "total_time": f"{time.time() - state['start_time']:.2f}s",
        "total_steps": len(state["step_logs"]),
        "performance_metrics": state["performance_metrics"],
        "step_logs": state["step_logs"]
    }

# ===========================================
# Enhanced Model Services with Logging
# ===========================================

class FormatFixerService:
    """Fixes malformed JSON responses using Claude"""

    def __init__(self, config: Config):
        self.config = config
        self.logger = AgentLogger(f"{BUNDLE_ID}.FormatFixer", config.log_level)

        # Always use Claude for format fixing (independent model)
        try:
            import anthropic
            self.claude_client = anthropic.Anthropic(api_key=config.claude_api_key)
            self.logger.log_step("claude_client_initialized", {"model": config.format_fixer_model})
        except ImportError:
            self.claude_client = None
            self.logger.log_error(ImportError("anthropic package not available"), "Claude client initialization")

    async def fix_format(self, raw_output: str, expected_format: str, input_data: str, system_prompt: str) -> Dict[str, Any]:
        """Fix malformed JSON using Claude"""
        operation_start = self.logger.start_operation("format_fixing",
            raw_output_length=len(raw_output),
            expected_format=expected_format,
            input_data_length=len(input_data)
        )

        if not self.claude_client:
            error_msg = "Claude client not available"
            self.logger.log_error(Exception(error_msg), "Client unavailable")
            self.logger.finish_operation(False, error_msg)
            return {"success": False, "error": error_msg, "raw": raw_output}

        try:
            fix_prompt = f"""You are a JSON format fixer. You receive malformed JSON output and need to fix it to match the expected format.

Original system prompt: {system_prompt}
Original input: {input_data}
Expected format: {expected_format}
Malformed output: {raw_output}

Please fix the JSON to match the expected format. Return only the fixed JSON, no additional text."""

            self.logger.log_step("sending_fix_request", {
                "model": self.config.format_fixer_model,
                "fix_prompt_length": len(fix_prompt),
                "raw_output_preview": raw_output[:200] + "..." if len(raw_output) > 200 else raw_output
            })

            response = self.claude_client.messages.create(
                model=self.config.format_fixer_model,
                max_tokens=1000,
                messages=[{"role": "user", "content": fix_prompt}]
            )

            fixed_content = response.content[0].text

            self.logger.log_step("fix_response_received", {
                "fixed_content_length": len(fixed_content),
                "fixed_content_preview": fixed_content[:200] + "..." if len(fixed_content) > 200 else fixed_content
            })

            # Try to parse the fixed content
            try:
                parsed = json.loads(fixed_content)
                self.logger.log_step("fix_validation_successful", {
                    "parsed_fields": list(parsed.keys()) if isinstance(parsed, dict) else "non-dict"
                })

                result = {"success": True, "data": parsed, "raw": fixed_content}
                self.logger.finish_operation(True, "Format fixing successful")
                return result

            except json.JSONDecodeError as e:
                self.logger.log_error(e, "Fixed content still not valid JSON")
                result = {"success": False, "error": "Fixed content is still not valid JSON", "raw": fixed_content}
                self.logger.finish_operation(False, "Fixed content still invalid")
                return result

        except Exception as e:
            self.logger.log_error(e, "Format fixing failed")
            result = {"success": False, "error": str(e), "raw": raw_output}
            self.logger.finish_operation(False, f"Format fixing failed: {str(e)}")
            return result

# ===========================================
# KB Search Agent (Preserved for Future Use)
# ===========================================

class KBSearchAgent:
    """Calls your KB /search endpoint when needed"""
    def __init__(self, config: Config):
        self.url = config.kb_search_url
        self.logger = AgentLogger(f"{BUNDLE_ID}.KBSearchAgent", config.log_level)

    async def search(self, query: str, top_k: int = 5, resource_id: Optional[str] = None):
        """Search the knowledge base"""
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
            self.logger.finish_operation(False, result_summary="KB search failed")
            return []

    async def search_with_state(self, state: ChatGraphState) -> ChatGraphState:
        """Search KB and update state - for potential future integration"""
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
            add_step_log(state, "kb_search_failed", {
                "error": str(e),
                "query": state["user_message"]
            })
            state["search_hits"] = []

        return state

# ===========================================
# Enhanced LangGraph Agents - Working Directly with State
# ===========================================

class ClassifierAgent:
    """Classifies whether the query should be handled by our specialized model"""

    def __init__(self, config: Config):
        self.config = config
        self.logger = AgentLogger(f"{BUNDLE_ID}.ClassifierAgent", config.log_level)
        self.model_service = ThematicBotModelService(config)
        self.format_fixer = FormatFixerService(config)

    async def classify(self, state: ChatGraphState) -> ChatGraphState:
        """Classify the user query - working directly with state"""

        operation_start = self.logger.start_operation("classify_query",
            user_message=state["user_message"][:100] + "..." if len(state["user_message"]) > 100 else state["user_message"],
            execution_id=state["execution_id"]
        )

        if not self.config.has_classifier:
            # Skip classification for models that don't support it
            state["is_our_domain"] = True  # Default to processing
            state["classification_reasoning"] = f"Model {self.config.selected_model} processes all queries without classification"

            self.logger.log_step("classification_skipped", {
                "reason": "Model doesn't support classification",
                "default_domain": True
            })

            add_step_log(state, "classification", {
                "success": True,
                "skipped": True,
                "reason": "Model doesn't support classification"
            })

            self.logger.finish_operation(True, "Classification skipped")
            return state

        # Create system prompt including summary from context if available
        summary_context = ""
        if state["context"].get("running_summary"):
            summary_context = f"Conversation summary: {state['context']['running_summary']}\n\n"

        system_prompt = summary_context + """You are a **domain classifier** for a **Houseplants & Home Gardening assistant**.
Your task is to determine whether a user query should be handled by our **specialized plant-care model** or not.

Our specialized model is designed to handle:
* Houseplant care (light, watering, soil, humidity, temperature)
* Troubleshooting symptoms (yellow/brown leaves, droop, leaf drop)
* Pests & diseases common to home growing (spider mites, mealybugs, aphids, fungus gnats, powdery mildew)
* Repotting, pruning, propagation (cuttings, division, air layering)
* Fertilizing and soil mixes (potting mix components, drainage, containers)
* Home gardening basics (seed starting, herbs/veggies in pots/beds, watering schedules, seasonality, compost basics)
* Safety notes for pets/kids regarding common ornamental/edible plants (non-clinical guidance)

Classify as **NOT our domain** if the query is about:
* Large-scale agriculture, landscaping construction, or arborist/tree surgery
* Advanced botany research or plant genetics
* Non-gardening home topics (plumbing, electrical, HVAC), general small talk, or unrelated tech/business

Return your response as JSON:
{
  "is_our_domain": true/false,
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation"
}
"""

        self.logger.log_step("classification_prompt_prepared", {
            "system_prompt_length": len(system_prompt),
            "model": self.config.selected_model
        })

        with with_accounting("chat.agentic.classifier",
                             metadata={ "message": state["user_message"] }):
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

            self.logger.log_step("classification_successful", {
                "is_our_domain": state["is_our_domain"],
                "confidence": classification_data["confidence"],
                "reasoning": classification_data["reasoning"]
            })

            add_step_log(state, "classification", {
                "success": True,
                "is_our_domain": state["is_our_domain"],
                "confidence": classification_data["confidence"]
            })

            self.logger.finish_operation(True, f"Classified as {'our domain' if state['is_our_domain'] else 'not our domain'}")
        else:
            # Try format fixer
            if state["format_fix_attempts"] < 3:
                state["format_fix_attempts"] += 1

                fix_result = await self.format_fixer.fix_format(
                    result["raw"], "ClassificationOutput", state["user_message"], system_prompt
                )

                if fix_result["success"]:
                    validated = ClassificationOutput.model_validate(fix_result["data"])
                    state["is_our_domain"] = validated.is_our_domain
                    state["classification_reasoning"] = validated.reasoning

                    add_step_log(state, "classification_fixed", {
                        "success": True,
                        "fix_attempt": state["format_fix_attempts"],
                        "is_our_domain": state["is_our_domain"]
                    })

                    self.logger.finish_operation(True, "Classification successful after format fix")
                else:
                    error_msg = f"Classification failed after format fix: {fix_result['error']}"
                    state["error_message"] = error_msg
                    state["is_our_domain"] = True  # Default to processing

                    add_step_log(state, "classification_failed", {
                        "success": False,
                        "error": error_msg,
                        "fallback_used": True
                    })

                    self.logger.finish_operation(False, "Classification failed even after format fix")
            else:
                error_msg = f"Classification failed: {result['error']}"
                state["error_message"] = error_msg
                state["is_our_domain"] = True  # Default to processing

                add_step_log(state, "classification_failed", {
                    "success": False,
                    "error": error_msg,
                    "max_attempts_reached": True
                })

                self.logger.finish_operation(False, "Classification failed after max attempts")

        return state

class QueryWriterAgent:
    """Generates RAG queries"""

    def __init__(self, config: Config):
        self.config = config
        self.logger = AgentLogger(f"{BUNDLE_ID}.QueryWriterAgent", config.log_level)
        self.model_service = ThematicBotModelService(config)
        self.format_fixer = FormatFixerService(config)

    async def write_queries(self, state: ChatGraphState) -> ChatGraphState:
        """Generate weighted queries for RAG retrieval - working directly with state"""

        operation_start = self.logger.start_operation("write_queries",
            user_message=state["user_message"][:100] + "..." if len(state["user_message"]) > 100 else state["user_message"],
            execution_id=state["execution_id"]
        )

        system_prompt = """You are a query writer for a RAG system. Your task is to break down user questions into specific, weighted queries for document retrieval.

Generate 3-6 queries that will help retrieve relevant information to answer the user's question. Each query should have a weight indicating its importance (0.0 to 1.0).

IMPORTANT: Format your response as JSON:
{
    "chain_of_thought": "Your reasoning process for breaking down the question",
    "queries": [
        {
            "query": "specific search query",
            "weight": 0.8,
            "reasoning": "why this query is important"
        }
    ]
}"""

        self.logger.log_step("query_generation_prompt_prepared", {
            "system_prompt_length": len(system_prompt),
            "model": self.config.selected_model
        })

        with with_accounting("chat.agentic.query_writer",
                             metadata={ "message": state["user_message"] }):
            result = await self.model_service.call_model_with_structure(
                self.model_service.query_writer_client,
                system_prompt,
                state["user_message"],
                QueryWriterOutput,
                client_cfg=self.model_service.describe_client(self.model_service.query_writer_client, role="query_writer")
            )

        if result["success"]:
            query_data = result["data"]
            state["rag_queries"] = query_data["queries"]

            self.logger.log_step("queries_generated_successfully", {
                "query_count": len(state["rag_queries"]),
                "queries": [{"query": q["query"], "weight": q["weight"]} for q in state["rag_queries"]],
                "chain_of_thought": query_data["chain_of_thought"]
            })

            add_step_log(state, "query_generation", {
                "success": True,
                "query_count": len(state["rag_queries"]),
                "total_weight": sum(q["weight"] for q in state["rag_queries"])
            })

            self.logger.finish_operation(True, f"Generated {len(state['rag_queries'])} queries")
        else:
            # Try format fixer or provide fallback
            if state["format_fix_attempts"] < 3:
                state["format_fix_attempts"] += 1

                fix_result = await self.format_fixer.fix_format(
                    result["raw"], "QueryWriterOutput", state["user_message"], system_prompt
                )

                if fix_result["success"]:
                    validated = QueryWriterOutput.model_validate(fix_result["data"])
                    state["rag_queries"] = validated.queries

                    add_step_log(state, "query_generation_fixed", {
                        "success": True,
                        "fix_attempt": state["format_fix_attempts"],
                        "query_count": len(state["rag_queries"])
                    })

                    self.logger.finish_operation(True, "Query generation successful after format fix")
                else:
                    # Fallback to simple query based on user message
                    state["rag_queries"] = [{
                        "query": state["user_message"],
                        "weight": 1.0,
                        "reasoning": "fallback - using original user message"
                    }]

                    add_step_log(state, "query_generation_failed", {
                        "success": False,
                        "error": fix_result["error"],
                        "fallback_used": True
                    })

                    self.logger.finish_operation(False, "Query generation failed, using fallback")
            else:
                # Final fallback
                state["rag_queries"] = [{
                    "query": state["user_message"],
                    "weight": 1.0,
                    "reasoning": "fallback - using original user message"
                }]

                add_step_log(state, "query_generation_failed", {
                    "success": False,
                    "error": result["error"],
                    "max_attempts_reached": True
                })

                self.logger.finish_operation(False, "Query generation failed after max attempts")

        return state

class RAGAgent:
    """Handles document retrieval"""

    def __init__(self, config: Config):
        self.config = config
        self.logger = AgentLogger(f"{BUNDLE_ID}.RAGAgent", config.log_level)
        self.rag_service = RAGService(config)

    async def retrieve(self, state: ChatGraphState) -> ChatGraphState:
        """Retrieve relevant documents - working directly with state"""

        operation_start = self.logger.start_operation("retrieve_documents",
            execution_id=state["execution_id"],
            query_count=len(state["rag_queries"]) if state["rag_queries"] else 0
        )

        if not state["rag_queries"]:
            error_msg = "No queries available for retrieval"
            state["error_message"] = error_msg

            self.logger.log_step("no_queries_available", {"error": error_msg})
            add_step_log(state, "retrieval_failed", {"success": False, "error": error_msg})
            self.logger.finish_operation(False, error_msg)

            return state

        try:
            self.logger.log_step("starting_document_retrieval", {
                "queries": [q["query"] for q in state["rag_queries"]],
                "query_weights": [q["weight"] for q in state["rag_queries"]],
                "embedding_type": "custom" if self.config.custom_embedding_endpoint else "openai"
            })

            docs = await self.rag_service.retrieve_documents(state["rag_queries"])
            state["retrieved_docs"] = docs

            self.logger.log_step("retrieval_successful", {
                "retrieved_count": len(docs),
                "doc_sources": [doc.get("metadata", {}).get("source", "unknown") for doc in docs],
                "doc_previews": [doc["content"][:100] + "..." for doc in docs]
            })

            add_step_log(state, "retrieval", {
                "success": True,
                "document_count": len(docs),
                "unique_sources": len(set(doc.get("metadata", {}).get("source", "unknown") for doc in docs))
            })

            self.logger.finish_operation(True, f"Retrieved {len(docs)} documents")

        except Exception as e:
            error_msg = f"Document retrieval failed: {str(e)}"
            state["error_message"] = error_msg
            state["retrieved_docs"] = []

            self.logger.log_error(e, "Document retrieval failed")
            add_step_log(state, "retrieval_failed", {
                "success": False,
                "error": error_msg
            })

            self.logger.finish_operation(False, error_msg)

        return state

class RerankingAgent:
    """Reranks retrieved documents"""

    def __init__(self, config: Config):
        self.config = config
        self.logger = AgentLogger(f"{BUNDLE_ID}.RerankingAgent", config.log_level)
        self.model_service = ThematicBotModelService(config)
        self.format_fixer = FormatFixerService(config)

    async def rerank(self, state: ChatGraphState) -> ChatGraphState:
        """Rerank documents by relevance - working directly with state"""

        operation_start = self.logger.start_operation("rerank_documents",
            execution_id=state["execution_id"],
            document_count=len(state["retrieved_docs"]) if state["retrieved_docs"] else 0
        )

        if not state["retrieved_docs"]:
            state["reranked_docs"] = []
            self.logger.log_step("no_documents_to_rerank", {"message": "No documents available for reranking"})
            add_step_log(state, "reranking_skipped", {"success": True, "reason": "no_documents"})
            self.logger.finish_operation(True, "No documents to rerank")
            return state

        system_prompt = """You are a document reranking expert. Given a user question and retrieved documents, rerank them by relevance.

Provide a relevance score from 0.0 to 1.0 for each document and sort them by relevance.

Return your response as JSON:
{
    "reranked_docs": [
        {
            "content": "document content",
            "metadata": {...},
            "relevance_score": 0.95,
            "ranking_position": 1
        }
    ],
    "reasoning": "Explanation of ranking decisions"
}"""

        docs_text = json.dumps(state["retrieved_docs"], indent=2)
        user_message = f"User question: {state['user_message']}\n\nDocuments to rerank:\n{docs_text}"

        self.logger.log_step("reranking_prompt_prepared", {
            "system_prompt_length": len(system_prompt),
            "user_message_length": len(user_message),
            "documents_to_rerank": len(state["retrieved_docs"]),
            "model": self.config.selected_model
        })

        with with_accounting("chat.agentic.reranking",
                             metadata={ "message": user_message }):
            result = await self.model_service.call_model_with_structure(
                self.model_service.reranker_client,
                system_prompt,
                user_message,
                RerankingOutput,
                client_cfg=self.model_service.describe_client(self.model_service.reranker_client, role="reranker")
            )

        if result["success"]:
            reranking_data = result["data"]
            state["reranked_docs"] = reranking_data["reranked_docs"]

            self.logger.log_step("reranking_successful", {
                "reranked_count": len(state["reranked_docs"]),
                "relevance_scores": [doc.get("relevance_score", 0) for doc in state["reranked_docs"]],
                "reasoning": reranking_data["reasoning"]
            })

            add_step_log(state, "reranking", {
                "success": True,
                "reranked_count": len(state["reranked_docs"]),
                "avg_relevance_score": sum(doc.get("relevance_score", 0) for doc in state["reranked_docs"]) / len(state["reranked_docs"]) if state["reranked_docs"] else 0
            })

            self.logger.finish_operation(True, f"Reranked {len(state['reranked_docs'])} documents")
        else:
            # Fallback: keep original order with default scores
            state["reranked_docs"] = []
            for i, doc in enumerate(state["retrieved_docs"]):
                doc_copy = doc.copy()
                doc_copy["relevance_score"] = 1.0 - (i * 0.1)  # Decreasing scores
                doc_copy["ranking_position"] = i + 1
                state["reranked_docs"].append(doc_copy)

            self.logger.log_step("reranking_failed_fallback", {
                "error": result["error"],
                "using_original_order": True,
                "fallback_count": len(state["reranked_docs"])
            })

            add_step_log(state, "reranking_failed", {
                "success": False,
                "error": result["error"],
                "fallback_used": True
            })

            self.logger.finish_operation(False, "Reranking failed, using original order")

        return state

class AnswerGeneratorAgent:
    """Generates final answers"""

    def __init__(self, config: Config, delta_emitter: DeltaEmitter, streaming: bool = True):

        self.config = config
        self.logger = AgentLogger(f"{BUNDLE_ID}.AnswerGeneratorAgent", config.log_level)
        self.model_service = ThematicBotModelService(config)
        self.emit_delta = delta_emitter or (lambda *_: asyncio.sleep(0))
        self.streaming = streaming

    async def generate_answer(self, state: ChatGraphState) -> ChatGraphState:
        """Generate final answer - working directly with state"""

        operation_start = self.logger.start_operation("generate_answer",
            execution_id=state["execution_id"],
            is_our_domain=state["is_our_domain"],
            document_count=len(state["reranked_docs"]) if state["reranked_docs"] else 0
        )

        context_docs = ""
        doc_sources = []
        if state["reranked_docs"]:
            for i, doc in enumerate(state["reranked_docs"][:5]):  # Top 5 docs
                context_docs += f"Document {i+1}:\n{doc['content']}\n\n"
                if doc.get("metadata", {}).get("source"):
                    doc_sources.append(doc["metadata"]["source"])

        self.logger.log_step("context_preparation", {
            "context_length": len(context_docs),
            "documents_used": len(state["reranked_docs"][:5]) if state.get("reranked_docs") else 0,
            "sources": doc_sources
        })

        # Include summary context if available
        summary_context = ""
        if state["context"].get("running_summary"):
            summary_context = f"Previous conversation summary: {state['context']['running_summary']}\n\n"

        system_prompt = f"""{summary_context}You are a helpful assistant. Answer the user's question using the provided context documents.

Context Documents:
{context_docs}

Provide a comprehensive answer that:
1. Directly addresses the user's question
2. Uses information from the context documents when available
3. Shows clear reasoning
4. Admits when information is insufficient
5. Provides practical guidance when possible
6. Suggest followup questions if needed

Be helpful, accurate, and cite specific information from the documents when relevant."""

        self.logger.log_step("answer_generation_prompt_prepared", {
            "system_prompt_length": len(system_prompt),
            "model": self.config.selected_model
        })

        try:
            result_text = ""
            usage_out: Dict[str, Any] = {}
            with with_accounting("chat.agentic.answer_generator",
                                 metadata={ "message": state["user_message"] }):
                if self.streaming:
                    idx = -1

                    async def on_delta(txt: str):
                        nonlocal idx, result_text
                        if not txt:
                            return
                        result_text += txt
                        idx += 1
                        # fire token to client
                        await self.emit_delta(txt, idx)

                    stream_res = await self.model_service.stream_model_text_tracked(
                        self.model_service.answer_generator_client,
                        [
                            SystemMessage(content=system_prompt, id=_mid("sys")),
                            HumanMessage(content=state["user_message"], id=_mid("user")),
                        ],
                        on_delta=on_delta,
                        temperature=0.3,
                        max_tokens=2000,
                        client_cfg=self.model_service.describe_client(
                            self.model_service.answer_generator_client,
                            role="answer_generator"
                        ),
                    )
                    # fill in any trailing text/usage if provider returns them
                    result_text = stream_res.get("text", result_text)
                    usage_out = stream_res.get("usage", {}) or {}

                else:
                    # For final answer, we can use simple text generation
                    res = await self.model_service.call_model_text(
                        self.model_service.answer_generator_client,
                        [
                            SystemMessage(content=system_prompt, id=_mid("sys")),
                            HumanMessage(content=state["user_message"], id=_mid("user"))
                        ],
                        temperature=0.3,
                        max_tokens=2000,
                        client_cfg=self.model_service.describe_client(self.model_service.answer_generator_client, role="answer_generator")
                    )
                    result_text = res["text"]
                    usage_out = res.get("usage", {})

            state["final_answer"] = result_text
            add_step_log(state, "answer_generation_usage", {"usage": usage_out})

            # Add messages to state for next summarization cycle
            # state["messages"].append(HumanMessage(content=state["user_message"])
            ai_msg = AIMessage(content=state["final_answer"], id=_mid("ai"))
            state["messages"].append(ai_msg)

            self.logger.log_step("answer_generated_successfully", {
                "answer_length": len(state["final_answer"]),
                "answer_preview": state["final_answer"][:200] + "..." if len(state["final_answer"]) > 200 else state["final_answer"],
                "sources_used": doc_sources
            })

            add_step_log(state, "answer_generation", {
                "success": True,
                "answer_length": len(state["final_answer"]),
                "sources_count": len(doc_sources),
                "context_documents_used": len(state["reranked_docs"][:5]) if state.get("reranked_docs") else 0
            })

            self.logger.finish_operation(True, f"Generated answer of {len(state['final_answer'])} characters")

        except Exception as e:
            error_msg = f"Answer generation failed: {str(e)}"
            state["error_message"] = error_msg
            state["final_answer"] = "I apologize, but I encountered an error generating the response."

            self.logger.log_error(e, "Answer generation failed")
            add_step_log(state, "answer_generation_failed", {
                "success": False,
                "error": error_msg
            })

            self.logger.finish_operation(False, error_msg)

        return state

# ===========================================
# LangGraph Workflow with State Management
# ===========================================
@agentic_workflow(name=f"{BUNDLE_ID}", version="1.0.0", priority=150)
class ChatWorkflow:
    """Main workflow orchestrator with proper state management"""

    def __init__(self,
                 config: Config,
                 step_emitter: Optional[StepEmitter] = None,
                 delta_emitter: Optional[DeltaEmitter] = None):
        self.config = config
        self.logger = AgentLogger(f"{BUNDLE_ID}.ChatWorkflow", config.log_level)
        self.emit_step: StepEmitter = step_emitter or (lambda *_args, **_kw: asyncio.sleep(0))
        self.emit_delta: DeltaEmitter = delta_emitter or (lambda *_: asyncio.sleep(0))

        # Initialize services
        self.classifier = ClassifierAgent(config)
        self.query_writer = QueryWriterAgent(config)
        self.rag_agent = RAGAgent(config)
        self.reranking_agent = RerankingAgent(config)
        self.answer_generator = AnswerGeneratorAgent(config, delta_emitter=self.emit_delta)
        self.kb_search_agent = KBSearchAgent(config)  # Preserved for future use

        # Create memory for persistence
        self.memory = MemorySaver()
        self.graph = self._build_graph()

        self.logger.log_step("workflow_initialized", {
            "selected_model": config.selected_model,
            "has_classifier": config.has_classifier,
            "provider": config.provider,
            "embedding_type": "custom" if config.custom_embedding_endpoint else "openai",
            "kb_search_available": bool(config.kb_search_url)
        })

    async def run(self, session_id, conversation_id: str, state):
        result = await self.graph.ainvoke(
            state,
            config={"configurable": {"thread_id": conversation_id}},
        )
        payload = project_app_state(result)
        return payload



    def _step_start_payload(self, state: ChatGraphState, step: str) -> dict:
        if step == "query_writer":
            return {"message": "Generating RAG queries..."}
        if step == "rag_retrieval":
            return {"message": "Retrieving documents..."}
        if step == "reranking":
            return {"message": "Reranking documents..."}
        if step == "answer_generator":
            return {"message": "Generating final answer..."}
        if step == "classifier":
            return {"message": "Classifying domain..."}
        return {}

    def _step_end_payload(self, state: ChatGraphState, step: str) -> dict:
        if step == "classifier":
            return {
                "is_our_domain": state.get("is_our_domain"),
                "message": state.get("classification_reasoning"),
            }
        if step == "query_writer":
            return {
                "query_count": len(state.get("rag_queries") or []),
                "queries": [q["query"] for q in (state.get("rag_queries") or [])][:6],
            }
        if step == "rag_retrieval":
            return {
                "retrieved_count": len(state.get("retrieved_docs") or []),
                # optional: attach a lightweight preview for your KB sidebar
                "kb_search_results": {
                    "results": (state.get("retrieved_docs") or [])[:10],
                    "total_results": len(state.get("retrieved_docs") or []),
                    "query": state.get("user_message"),
                }
            }
        if step == "reranking":
            reranked = state.get("reranked_docs") or []
            avg_rel = sum(d.get("relevance_score", 0.0) for d in reranked) / len(reranked) if reranked else 0.0
            return {"avg_relevance": avg_rel, "reranked_count": len(reranked)}
        if step == "answer_generator":
            ans = (state.get("final_answer") or "")
            return {"answer_length": len(ans)}
        return {}

    def _wrap_node(self, fn, step_name: str):
        async def _wrapped(state: ChatGraphState) -> ChatGraphState:
            # started
            await self.emit_step(step_name, "started", self._step_start_payload(state, step_name))
            try:
                out = fn(state)
                if asyncio.iscoroutine(out):
                    out = await out
                # completed
                await self.emit_step(step_name, "completed", self._step_end_payload(out, step_name))
                return out
            except Exception as e:
                await self.emit_step(step_name, "error", {"error": str(e)})
                raise
        return _wrapped

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow with proper state management"""

        # Create graph with proper state schema
        workflow = StateGraph(ChatGraphState)

        # Add user message to messages list at the start
        def add_user_message(state: ChatGraphState) -> Dict[str, Any]:
            msg = HumanMessage(content=state["user_message"], id=_mid("user"))
            # IMPORTANT: return a partial update, don't mutate state
            return {"messages": [msg]}

        # workflow.add_node("add_user_message", add_user_message)

        # Add SummarizationNode - this is the correct way to use it
        if SummarizationNode:
            # Create summarization model
            chat_model = init_chat_model(f"openai:{self.config.answer_generator_model}")
            summarization_model = chat_model.bind(max_tokens=256)

            # Create SummarizationNode with proper configuration
            summarization_node = SummarizationNode(
                model=summarization_model,
                max_tokens=1000,  # Maximum tokens in output
                max_tokens_before_summary=500,  # Trigger summarization threshold
                max_summary_tokens=200,  # Budget for summary
                input_messages_key="messages",  # Read from messages
                output_messages_key="summarized_messages",  # Write to summarized_messages
                token_counter=lambda msgs: sum(len(str(msg.content)) for msg in msgs),
            )

            workflow.add_node("summarize", summarization_node)
        else:
            # Fallback if langmem not available
            def simple_summarize(state: ChatGraphState) -> ChatGraphState:
                """Simple fallback summarization"""
                state["summarized_messages"] = state["messages"][-10:]  # Keep last 10 messages
                return state

            workflow.add_node("summarize", simple_summarize)

        # Add processing nodes
        if self.config.has_classifier:
            workflow.add_node("classifier", self._wrap_node(self.classifier.classify, "classifier"))

        workflow.add_node("query_writer", self._wrap_node(self.query_writer.write_queries, "query_writer"))
        workflow.add_node("rag_retrieval", self._wrap_node(self.rag_agent.retrieve, "rag_retrieval"))
        workflow.add_node("reranking", self._wrap_node(self.reranking_agent.rerank, "reranking"))
        workflow.add_node("answer_generator", self._wrap_node(self.answer_generator.generate_answer, "answer_generator"))

        # Optional: Add KB search node (for future use)
        # workflow.add_node("kb_search", self.kb_search_agent.search_with_state)

        # Add edges
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
            await self.emit_step("workflow_complete", "completed", {"message": "Workflow complete"})
            return state
        workflow.add_node("workflow_complete", _emit_workflow_complete)
        workflow.add_edge("answer_generator", "workflow_complete")
        workflow.add_edge("workflow_complete", END)

        # Compile with memory for persistence
        return workflow.compile(checkpointer=self.memory)

    # ===========================================
    # Public Interface Methods
    # ===========================================

    async def process_message(self,
                              user_message: str,
                              thread_id: str = "default",
                              seed_messages: Optional[List[AnyMessage]] = None) -> Dict[str, Any]:
        """Process a user message through the workflow with proper state management"""

        operation_start = self.logger.start_operation("process_message",
            user_message=user_message[:100] + "..." if len(user_message) > 100 else user_message,
            thread_id=thread_id,
            selected_model=self.config.selected_model
        )

        # Create initial state
        initial_state = create_initial_state(user_message)
        if seed_messages:
            initial_state["messages"].extend(seed_messages)

        try:
            self.logger.log_step("invoking_workflow", {
                "thread_id": thread_id,
                "workflow_nodes": self._get_workflow_node_names(),
                "embedding_type": "custom" if self.config.custom_embedding_endpoint else "openai"
            })

            # Invoke the graph with thread_id for memory persistence
            result = await self.graph.ainvoke(
                initial_state,
                config={"configurable": {"thread_id": thread_id}}
            )

            self.logger.log_step("workflow_completed", {
                "final_answer_length": len(result.get("final_answer", "")) if result.get("final_answer") else 0,
                "has_error": bool(result.get("error_message")),
                "steps_completed": len(result.get("step_logs", [])),
                "is_our_domain": result.get("is_our_domain"),
                "retrieved_docs_count": len(result.get("retrieved_docs", [])),
                "reranked_docs_count": len(result.get("reranked_docs", [])),
                "conversation_length": len(result.get("messages", [])),
                "has_summary": bool(result.get("context", {}).get("running_summary"))
            })

            self.logger.finish_operation(True, f"Workflow completed successfully")

            return result

        except Exception as e:
            error_msg = f"Workflow execution failed: {str(e)}"
            self.logger.log_error(e, "Workflow execution failed")
            self.logger.finish_operation(False, error_msg)

            # Return error state
            return {
                **initial_state,
                "error_message": error_msg,
                "final_answer": "I apologize, but I encountered an error processing your request."
            }

    async def get_conversation_history(self, thread_id: str = "default") -> List[AnyMessage]:
        """Get conversation history for a thread"""
        try:
            # Get the latest state for the thread
            state = await self.graph.aget_state(config={"configurable": {"thread_id": thread_id}})
            return state.values.get("messages", []) if state.values else []
        except Exception as e:
            self.logger.log_error(e, f"Failed to get conversation history for thread {thread_id}")
            return []

    async def get_conversation_summary(self, thread_id: str = "default") -> str:
        """Get conversation summary for a thread"""
        try:
            # Get the latest state for the thread
            state = await self.graph.aget_state(config={"configurable": {"thread_id": thread_id}})
            if state.values and state.values.get("context"):
                running_summary = state.values["context"].get("running_summary")
                return str(running_summary) if running_summary else ""
            return ""
        except Exception as e:
            self.logger.log_error(e, f"Failed to get conversation summary for thread {thread_id}")
            return ""

    async def get_execution_logs(self, thread_id: str = "default") -> List[Dict[str, Any]]:
        """Get execution logs for a thread"""
        try:
            # Get the latest state for the thread
            state = await self.graph.aget_state(config={"configurable": {"thread_id": thread_id}})
            return state.values.get("step_logs", []) if state.values else []
        except Exception as e:
            self.logger.log_error(e, f"Failed to get execution logs for thread {thread_id}")
            return []

    def _get_workflow_node_names(self) -> List[str]:
        """Get list of workflow node names"""
        nodes = ["workflow_start", "summarize", "query_writer", "rag_retrieval", "reranking", "answer_generator"]
        if self.config.has_classifier:
            nodes.insert(2, "classifier")  # After summarize
        return nodes

    def suggestions(self):
        return [
            "What light, watering, and soil do my common houseplants need?",
            "Why are my leaves yellow/brown/curling, and how do I fix it?",
            "How can I prevent and treat pests like spider mites and fungus gnats?",
            "When should I repot, and what potting mix should I use?"
        ]

# ===========================================
# Usage Example
# ===========================================

async def example_usage():
    """Example of how to use the corrected workflow"""

    from kdcube_ai_app.infra.accounting.envelope import AccountingEnvelope, bind_accounting
    from kdcube_ai_app.infra.accounting import with_accounting

    import os
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
        "metadata": {
            "description": "Chat App Workflow Example",
            "tags": ["demo", "chat", "workflow"]
        }
    }

    envelope = AccountingEnvelope.from_dict(acct_dict)
    kdcube_path = os.environ.get("KDCUBE_STORAGE_PATH")
    storage_backend = create_storage_backend(kdcube_path, **{})

    # Setup
    # env
    # OPENAI_API_KEY
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    claude_api_key = os.environ.get("ANTHROPIC_API_KEY")
    embedding_model = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")

    config = Config(selected_model="gpt-4o",
                    openai_api_key=openai_api_key,
                    claude_api_key=claude_api_key,
                    embedding_model=embedding_model)

    workflow = ChatWorkflow(config)

    thread_id = "user_123_conversation"

    async with bind_accounting(envelope, storage_backend, enabled=True):
        prompt = "What are recent incidents on jail-breaking the enterprise agents?"
        async with with_accounting(COMPONENT, metadata={"prompt": prompt}):
            # First message
            result1 = await workflow.process_message(
                prompt,
                thread_id=thread_id
            )
        print(f"Response 1: {result1['final_answer']}")
        print(f"Step logs: {len(result1.get('step_logs', []))}")

        # Check conversation history and summary
        history = await workflow.get_conversation_history(thread_id)
        summary = await workflow.get_conversation_summary(thread_id)
        logs = await workflow.get_execution_logs(thread_id)

        print(f"Conversation has {len(history)} messages")
        print(f"Summary: {summary}")
        print(f"Execution logs: {len(logs)} steps")

async def example_conversation():
    """Example of how to use the corrected workflow"""

    from kdcube_ai_app.infra.accounting.envelope import AccountingEnvelope, bind_accounting
    from kdcube_ai_app.infra.accounting import with_accounting

    import os
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
        "metadata": {
            "description": "Chat App Workflow Example",
            "tags": ["demo", "chat", "workflow"]
        }
    }

    envelope = AccountingEnvelope.from_dict(acct_dict)
    kdcube_path = os.environ.get("KDCUBE_STORAGE_PATH")
    storage_backend = create_storage_backend(kdcube_path, **{})

    # Setup
    # env
    # OPENAI_API_KEY
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    claude_api_key = os.environ.get("ANTHROPIC_API_KEY")
    embedding_model = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")

    config = Config(selected_model="gpt-4o",
                    openai_api_key=openai_api_key,
                    claude_api_key=claude_api_key,
                    embedding_model=embedding_model)

    workflow = ChatWorkflow(config)

    thread_id = "user_123_conversation"

    async with bind_accounting(envelope, storage_backend, enabled=True):
        prompt = "What are recent incidents on jail-breaking the enterprise agents?"
        async with with_accounting(COMPONENT, metadata={"prompt": prompt}):
            # First message
            result1 = await workflow.process_message(
                prompt,
                thread_id=thread_id
            )
        print(f"Response 1: {result1['final_answer']}")
        print(f"Step logs: {len(result1.get('step_logs', []))}")

        prompt = "How can I mitigate the risk caused by first incident you mentioned?"
        async with with_accounting(COMPONENT, metadata={"prompt": prompt}):
            # Follow-up message - should maintain context via summarization
            result2 = await workflow.process_message(
                prompt,
                thread_id=thread_id
            )
        print(f"Response 2: {result2['final_answer']}")

        # Check conversation history and summary
        history = await workflow.get_conversation_history(thread_id)
        summary = await workflow.get_conversation_summary(thread_id)
        logs = await workflow.get_execution_logs(thread_id)

        print(f"Conversation has {len(history)} messages")
        print(f"Summary: {summary}")
        print(f"Execution logs: {len(logs)} steps")

def setup_logging():
    """Setup global logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('agent_execution.log', mode='a')
        ]
    )

# ===========================================
# Main Application Entry Point
# ===========================================

if __name__ == "__main__":
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())

    # Setup logging
    setup_logging()

    # Test the system
    asyncio.run(example_usage())