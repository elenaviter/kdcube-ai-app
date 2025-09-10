# kdcube_ai_app/apps/chat/sdk/tools/context_rag_tool.py

from __future__ import annotations
from typing import Optional, List

import semantic_kernel as sk
try:
    from semantic_kernel.functions import kernel_function
except Exception:
    from semantic_kernel.utils.function_decorator import kernel_function

from kdcube_ai_app.apps.chat.sdk.context.retrieval.ctx_rag import ContextRAGClient

_CTX_CLIENT: Optional[ContextRAGClient] = None
def bind_integrations(integrations):
    global _CTX_CLIENT
    _CTX_CLIENT = integrations.get("ctx_client")

# These are available in your runtime:
from kdcube_ai_app.apps.chat.sdk.storage.conversation_store import ConversationStore
from kdcube_ai_app.apps.chat.sdk.context.vector.conv_index import ConvIndex

class ContextRAGTools:

    @kernel_function(
        name="ctx_search",
        description="Search in contex.... Please help document this"
    )
    async def ctx_search(self,
                         query: str,
                         kinds: Optional[List[str]] = None,
                         scope: str = "track",
                         top_k: int = 12,
                         days: int = 90,
                         include_deps: bool = True,
                         with_payload: bool = False) -> dict:
        """
        Search conversation/track context (user, assistant, artifacts).
        Scope is auto-detected from context.json but can be overridden.
        """
        return await _CTX_CLIENT.search(
            query=query, kinds=kinds, scope=scope, top_k=top_k, days=days,
            include_deps=include_deps, with_payload=with_payload
        )

    async def pull_text_artifact(self,
                                 artifact_uri: str) -> dict:
        """Fetch one stored message/artifact by artifact_uri (payload + meta + text)."""
        return await _CTX_CLIENT.pull_text_artifact(artifact_uri=artifact_uri)

    async def pull_file_artifact(self,
                                 artifact_uri: str) -> dict:
        """Fetch one stored message/artifact by artifact_uri (payload + meta + text)."""
        return await _CTX_CLIENT.pull_text_artifact(artifact_uri=artifact_uri)


    async def ctx_decide_reuse(self,
                               goal_kind: str, query: str, threshold: float = 0.78, days: int = 180, scope: str = "track") -> dict:
        """
        Decide whether to reuse an existing artifact of goal_kind for an edit/adapt task.
        Returns {reuse, candidate?, search, reason?}
        """
        return await _CTX_CLIENT.decide_reuse(goal_kind=goal_kind, query=query, threshold=threshold, days=days, scope=scope)

kernel = sk.Kernel()
tools = ContextRAGTools()
kernel.add_plugin(tools, "ctx_tools")