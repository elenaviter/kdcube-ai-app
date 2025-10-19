import os
from typing import List, Dict, Any

from kdcube_ai_app.apps.chat.sdk.codegen.project_retrieval import _materialize_glue_canvas, _pick_project_log_slot
from kdcube_ai_app.apps.chat.sdk.config import get_settings
from kdcube_ai_app.apps.chat.sdk.context.retrieval.ctx_rag import ContextRAGClient
from kdcube_ai_app.apps.chat.sdk.context.vector.conv_index import ConvIndex
from kdcube_ai_app.apps.chat.sdk.runtime.scratchpad import _turn_id_from_tags_safe
from kdcube_ai_app.apps.chat.sdk.storage.conversation_store import ConversationStore
from kdcube_ai_app.apps.chat.sdk.tools.citations import normalize_citation_item
from kdcube_ai_app.infra.service_hub.inventory import ConfigRequest, ModelServiceBase, create_workflow_config


async def get_pg_pool(_settings):
    global _pg_pool

    import asyncpg, json
    async def _init_conn(conn: asyncpg.Connection):
        # Encode/decode json & jsonb as Python dicts automatically
        await conn.set_type_codec('json',  encoder=json.dumps, decoder=json.loads, schema='pg_catalog')
        await conn.set_type_codec('jsonb', encoder=json.dumps, decoder=json.loads, schema='pg_catalog')

    _pg_pool = await asyncpg.create_pool(
        host=_settings.PGHOST,
        port=_settings.PGPORT,
        user=_settings.PGUSER,
        password=_settings.PGPASSWORD,
        database=_settings.PGDATABASE,
        ssl=_settings.PGSSL,
        init=_init_conn,
    )
    return _pg_pool

async def search_context(conv_store, ctx_client, model_service):
    # targets = [
    #     {
    #         "where": "user",
    #         "query": "my schema; concise; save-token; compact format; pricing table structure",
    #         "reason": "User's original efficient/compact schema definition"
    #     },
    #     {
    #         "where": "assistant_artifact",
    #         "query": "pricing entry PDF; Haiku 4.5; price table; JSON export",
    #         "reason": "The PDF artifact just created (turn_1) containing price table, which user wants replaced with their schema"
    #     }
    # ]

    targets = [
        {
            "where": "assistant_artifact",
            "query": "GateOut schema; JSON schema; PDF export"
        },
        {
            "where": "assistant_artifact",
            "query": "pricing table; Haiku 4.5 pricing entry; model pricing",
        },
        {
            "where": "assistant_artifact",
            "query": "Redis vulnerabilities; vulnerability table; CVE security",
        }
    ]

    conv = "88f56ef9-36fc-4a4f-8f27-5d53ff19dd03"
    user = "admin-user-1"
    track = "A"

    async def _search_one(where: str, query: str) -> list[dict]:
        """
        Search for turn logs via content matching.
        Returns artifact:turn.log entries for relevant turns.
        """
        conv_idx = conv_store
        if where in ("user", "assistant"):
            # Search user/assistant content, return turn logs
            [qvec] = await model_service.embed_texts([query])
            res = await conv_idx.search_turn_logs_via_content(
                user_id=user, conversation_id=conv, track_id=track,
                query_embedding=qvec,  # Use text search for simplicity; can add embedding later
                search_roles=(where,),
                top_k=5,
                days=365,
                scope="track"
            )
            return res or []

        elif where in ["assistant_artifact", "project_log"]:
            # Direct search on project logs (legacy path)
            res = await ctx_client.search(
                query=query,
                kinds=("artifact:project.log",),
                roles=("artifact",),
                days=365, top_k=5, scope="track",
                user_id=user, conversation_id=conv, track_id=track,
                include_deps=False, sort="hybrid"
            )
            return res.get("items") or []

        return []

    hits = []
    best_tid: str | None = None
    best_score: float = -1.0
    for t in targets:
        try:
            rows = await _search_one(t["where"], t["query"])
        except Exception:
            rows = []

        for r in rows:
            # Now r is a turn log artifact with relevance_score
            rec = float(r.get("score") or 0.0)
            sim = rec  # Same value for now
            comb = 0.8 * rec + 0.2 * sim
            tid = r.get("turn_id") or _turn_id_from_tags_safe(r.get("tags") or [])

            hits.append({
                "turn_id": tid,
                "role": "artifact",  # It's now the turn log artifact
                "ts": r.get("ts"),
                "score": float(r.get("score") or 0.0),
                "matched_via_role": r.get("matched_role"),  # What content matched
                "source_query": t["query"],
                "text": r.get("text", "")[:100] + "..." + r.get("text", "")[-100:],  # Preview of turn log
            })

            if tid and comb > best_score:
                best_score, best_tid = comb, tid

        print()

async def _build_program_history_from_turn_ids(
        context_rag_client,
        user_id,
        conversation_id: str,
        turn_ids: List[str],
        scope: str = "track", days: int = 365) -> List[Dict[str, Any]]:
    """
    For each turn_id, materialize: program presentation (if present), project_canvas / project_log
    from deliverables, and citations tied to the run. Returns the same shape as _build_program_history().
    """
    out = []
    seen_runs = set()

    for tid in turn_ids:
        mat = await context_rag_client.materialize_turn(
            user_id=user_id,conversation_id=conversation_id,
            turn_id=tid, scope=scope, days=days, with_payload=True
        )

        # Unpack rich envelopes (payload + ts + tags)
        prez_env = (mat.get("presentation") or {})
        dels_env = mat.get("deliverables") or {}
        assistant_env = mat.get("assistant") or {}
        user_env = mat.get("user") or {}
        solver_failure_env = mat.get("solver_failure") or {}
        citables_env = mat.get("citables") or {}
        files_env = mat.get("files") or {}

        prez = ((prez_env or {}).get("payload") or {}).get("payload") or {}
        dels = ((dels_env or {}).get("payload") or {}).get("payload") or {}
        citables = ((citables_env or {}).get("payload") or {}).get("payload") or {}
        assistant = ((((assistant_env or {}).get("payload") or {}).get("payload") or {})).get("completion") or ""
        user = (((user_env or {}).get("payload") or {}).get("payload") or {}).get("prompt") or ""
        # files = list((((files_env or {}).get("payload") or {}).get("payload") or {}).get("files_by_slot", {}).values())
        # files = [file for slot_files in files for file in slot_files.get("files")]
        files = (((files_env or {}).get("payload") or {}).get("payload") or {}).get("files_by_slot", {})
        d_items = list((dels or {}).get("items") or [])
        for de in d_items:
            de = de or {}
            artifact = de.get("value") or {}
            if artifact.get("type") == "file":
                slot_name = de.get("slot") or "default"
                file = files.get(slot_name) or {}
                artifact["path"] = file.get("path") or ""
                artifact["filename"] = file.get("filename")
                print()

        cite_items =  list((citables or {}).get("items") or [])
        round_reason = (dels or {}).get("round_reasoning") or ""

        # Prefer assistant ts, else user ts
        ts_val = assistant_env.get("ts") or user_env.get("ts") or ""

        # codegen_run_id priority: deliverables.payload -> tags -> presentation markdown
        codegen_run_id = (dels or {}).get("execution_id") or ""

        # Presentation markdown (if present)
        pres_md = (prez.get("markdown") or "") if isinstance(prez, dict) else ""

        # Citations bundle (if we have run id)
        cites = {"items": cite_items}

        # Extract canvas/log from deliverables items
        # canvas = _pick_canvas_slot(d_items) or {}
        project_log = _pick_project_log_slot(d_items) or {}

        materialized_canvas = {}
        try:
            glue = project_log.get("value","") if project_log else ""
            mat = _materialize_glue_canvas(glue, d_items)
            if mat and mat != glue:
                materialized_canvas = {"format": "markdown", "text": mat}
        except Exception as ex:
            materialized_canvas = {}

        exec_id = codegen_run_id
        if exec_id in seen_runs:
            continue
        seen_runs.add(exec_id)

        # Solver failure (markdown, if any)
        solver_failure = ((solver_failure_env or {}).get("payload") or {}).get("payload") or {}
        solver_failure_md = (solver_failure.get("markdown") or "") if isinstance(solver_failure, dict) else ""

        ret = {
            **({"program_presentation": pres_md} if pres_md else {}),
            # **({"project_canvas": {"format": canvas.get("format","markdown"), "text": canvas.get("value","")}} if canvas else {}),
            **({"project_log": {"format": project_log.get("format","markdown"), "text": project_log.get("value","")}} if project_log else {}),
            **({"project_log_materialized": materialized_canvas} if materialized_canvas else {}),
            **({"solver_failure": solver_failure_md} if solver_failure_md else {}),
            **({"web_links_citations": {"items": [normalize_citation_item(c) for c in cites["items"] if normalize_citation_item(c)]}}),
            **{"media": []},
            "ts": ts_val,
            **({"codegen_run_id": codegen_run_id} if codegen_run_id else {}),
            **({"round_reasoning": round_reason} if round_reason else {}),
            "assistant": assistant,
            "user": user,
            "deliverables": d_items if d_items else []
        }
        out.append({exec_id: ret})

    # newest first
    out.sort(key=lambda e: next(iter(e.values())).get("ts","") or "", reverse=True)
    return out

async def main():
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())

    _settings = get_settings()

    pg_pool = await get_pg_pool(_settings)
    conv_idx = ConvIndex(pool=pg_pool)
    conv_store = ConversationStore(_settings.STORAGE_PATH)


    await conv_idx.init()
    m = "claude-3-7-sonnet-20250219"
    # m = "claude-sonnet-4-20250514"
    role = "segment_enrichment"
    req = ConfigRequest(
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
        claude_api_key=os.environ.get("ANTHROPIC_API_KEY"),
        selected_model=m,
        role_models={ role: {"provider": "anthropic", "model": m}},
    )
    svc = ModelServiceBase(create_workflow_config(req))
    ctx_client = ContextRAGClient(conv_idx=conv_idx,
                                  store=conv_store,
                                  model_service=svc)
    await search_context(conv_idx, ctx_client, model_service=svc)

    user_id = "admin-user-1"
    conversation_id = "7c41b8e3-27cd-48ff-840e-9e158d1ee193"
    conversation_id = "a"

    conversation_id = ""
    tid = "turn_1759189840133_bo6bum"
    scope = "track"
    days = 365
    # mat = await ctx_client.materialize_turn(
    #     turn_id=tid, scope=scope, days=days, with_payload=True
    # )
    conversation_id = "5c0a6bb8-2fcd-4b40-ab10-a78991d1d77c"
    tid = "turn_1759582866288_mf8fjk"
    pro = await _build_program_history_from_turn_ids(ctx_client, user_id=user_id, conversation_id=conversation_id, turn_ids=[tid], scope=scope, days=days)
    conversation_id = "01c414d5-ef92-402a-ae1b-77d493961329"

    turns = await conv_idx.get_conversation_turn_ids_from_tags(user_id=user_id, conversation_id=conversation_id)
    print(f"Turns: {turns}")

    conversations = await ctx_client.list_conversations(user_id=user_id, last_n=2)

    is_new_conversation = len(turns) == 0
    c_details = await ctx_client.get_conversation_details(user_id=user_id, conversation_id=conversation_id)
    print()
    print(f"Conversations: {conversations}\nIs new: {is_new_conversation}\nDetails: {c_details}")

    conversation_artifacts = await ctx_client.fetch_conversation_artifacts(user_id=user_id,
                                                                           conversation_id=conversation_id,
                                                                           materialize=True)
    conversation_id = None
    FINGERPRINT_KIND = "artifact:turn.fingerprint.v1"
    CONV_START_FPS_TAG = "conv.start"
    data = await ctx_client.search(kinds=[FINGERPRINT_KIND],
                                   user_id=user_id,
                                   conversation_id=conversation_id,
                                   all_tags=[CONV_START_FPS_TAG],
                                   )

    conv_start = next(iter(data), None) if data else None
    conversation_title = conv_start.get("conversation_title") if conv_start else None
    print(f"Conv start: {conv_start}\nConversation title: '{conversation_title}'")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())