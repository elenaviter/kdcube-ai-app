# Conversation Memories Subsystem

This subsystem owns the conversation-level "active set" pointer used by the
codegen workflow and memory reconciler. It encapsulates where that pointer is
persisted and how it is read/written so call sites do not depend on storage
details.

Role
- Maintain the `conversation_state_v1` payload (picked buckets, selected local
  memory turn ids, cadence counters, timestamps, and optional title).
- Provide a single API for reads/writes used by the workflow and reconciler.
- Hide storage backend decisions (Postgres vs graph).

Implementation
- Primary storage: Postgres artifacts via `ContextRAGClient.upsert_artifact`,
  with kind `conversation.active_set.v1` and a stable tag
  `conv.state:conversation_state_v1`.
- Fallback storage: Graph (Neo4j) conversation blob, used only if the artifact
  backend is not configured.
- The API is implemented in `conv_memories.py` as `ConvMemoriesStore`.
