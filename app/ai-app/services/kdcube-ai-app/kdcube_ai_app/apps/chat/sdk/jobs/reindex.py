# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

# chat/sdk/jobs/reindex.py
import asyncio
from typing import List

from dotenv import load_dotenv, find_dotenv

from kdcube_ai_app.apps.chat.sdk.config import get_settings

from kdcube_ai_app.apps.chat.sdk.context.vector.conv_index import ConvIndex
from kdcube_ai_app.apps.chat.sdk.storage.conversation_store import ConversationStore
from models.provider import embed_texts

load_dotenv(find_dotenv())

async def reindex_conversation(tenant, project, user, conversation_id):

    # TODO: reindex also preferences
    # from kdcube_ai_app.apps.chat.sdk.context.graph.graph_ctx import GraphCtx
    # g = GraphCtx()
    v = ConvIndex()

    await v.init()
    settings = get_settings()

    conv_store = ConversationStore(settings.STORAGE_PATH)

    # somewhere in a maintenance task
    records = conv_store.list_conversation(
        tenant=tenant, project=project,
        user_type="registered", user_or_fp=user,
        conversation_id=conversation_id
    )

    def _embed(text: str) -> List[float]:
        # your sync wrapper around embed_texts
        return asyncio.get_event_loop().run_until_complete(embed_texts([text]))[0]

    n = await v.backfill_from_store(
        records=records,
        default_ttl_days=365,
        default_user_type="registered",
        embedder=_embed  # None to rely solely on persisted embeddings
    )
    print("reindexed rows:", n)