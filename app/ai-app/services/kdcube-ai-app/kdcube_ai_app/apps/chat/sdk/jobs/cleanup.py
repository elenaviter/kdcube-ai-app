# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

# chat/sdk/jobs/cleanup.py
import asyncio

from dotenv import load_dotenv, find_dotenv

from kdcube_ai_app.apps.chat.sdk.config import get_settings
from kdcube_ai_app.apps.chat.sdk.context.graph.graph_ctx import GraphCtx
from kdcube_ai_app.apps.chat.sdk.context.vector.conv_index import ConvIndex
from kdcube_ai_app.apps.chat.sdk.util import _make_project_schema

load_dotenv(find_dotenv())

async def run_cleanup(purge_anonymous_all: bool = False):
    g = GraphCtx()
    v = ConvIndex()
    await v.init()
    settings = get_settings()
    try:
        res = await g.cleanup_expired()
        print("[graph] cleanup:", res)
        schema = _make_project_schema(settings.TENANT, settings.PROJECT)
        # You can also purge expired conv messages if you want physically:
        # with asyncpg you could: DELETE FROM conv_messages WHERE ts + (ttl_days||' days')::interval < now();
        async with v._pool.acquire() as con:
            d = await con.execute(f"DELETE FROM {schema}.conv_messages WHERE ts + (ttl_days || ' days')::interval < now()")
            print("[vector] cleanup:", d)
        if purge_anonymous_all:
            n = await v.purge_user_type(user_type="anonymous", older_than_days=None)
            print("[vector] purge anonymous (all):", n)
    finally:
        await v.close()
        await g.close()

if __name__ == "__main__":
    asyncio.run(run_cleanup())
