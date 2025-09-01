# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

# chatbot/storage/storage.py

import json, time
from urllib.parse import urlparse
from datetime import datetime

from kdcube_ai_app.apps.chat.sdk.config import get_settings
from kdcube_ai_app.apps.chat.sdk.inventory import _mid

try:
    from kdcube_ai_app.storage.storage import create_storage_backend
except ImportError:
    raise ImportError("Please ensure 'kdcube_ai_app.storage.storage' is importable.")

class ConversationStore:
    """
    Writes raw messages to a storage backend rooted at ${KDCUBE_STORAGE_PATH}/cb
    Layout:
      {root}/tenants/{tenant}/projects/{project}/conversation/{anonymous|registered}/{user_or_fp}/{conversation_id}/{message_id}.json
    Returns a normalized URI (file://... or s3://...).
    """

    def __init__(self, storage_uri: str | None = None):
        self._settings = get_settings()
        self.storage_uri = storage_uri or self._settings.STORAGE_PATH
        self.backend = create_storage_backend(self.storage_uri)
        parsed = urlparse(self.storage_uri)
        self.scheme = parsed.scheme or "file"
        # logical prefix for our app
        self.root_prefix = "cb"

        # For URI reconstruction
        self._file_base = parsed.path if self.scheme == "file" else ""
        self._s3_bucket = parsed.netloc if self.scheme == "s3" else ""
        self._s3_prefix = parsed.path.lstrip("/") if self.scheme == "s3" else ""

    def _join(self, *parts: str) -> str:
        return "/".join([p.strip("/").replace("//","/") for p in parts if p is not None and p != ""])

    def _uri_for_path(self, relpath: str) -> str:
        if self.scheme == "file":
            # absolute path: file://<base>/<rel>
            # base may be '' if storage_uri was like 'file:///path'
            base = self._file_base.rstrip("/")
            abs_path = self._join(base, relpath)
            return "file://" + abs_path
        if self.scheme == "s3":
            # s3://bucket/prefix/rel
            prefix = self._s3_prefix.rstrip("/")
            key = self._join(prefix, relpath)
            return f"s3://{self._s3_bucket}/{key}"
        # Fallback: opaque
        return f"{self.scheme}://{relpath}"

    def put_message(
        self,
        *,
        tenant: str,
        project: str,
        user: str | None, # user id (None => anonymous)
        fingerprint: str | None, # optional browser/device fingerprint for anon
        conversation_id: str,
        role: str,
        text: str,
        meta: dict | None = None,
        embedding: list[float] | None = None,
        user_type: str = "anonymous",
        ttl_days: int = 365
    ) -> tuple[str, str]:
        """
        Writes the JSON payload and returns (uri, message_id).
        """
        # message_id = f"{role}-{msg_ts}-{uuid.uuid4().hex[:8]}"
        msg_ts = time.strftime("%Y-%m-%dT%H-%M-%S", time.gmtime())
        message_id = _mid(role, msg_ts)
        who = "registered" if (user and user != "anonymous") else "anonymous"
        user_or_fp = user if who == "registered" else (fingerprint or "unknown")

        rel = self._join(
            self.root_prefix,
            "tenants", tenant,
            "projects", project,
            "conversation", who, user_or_fp,
            conversation_id,
            f"{message_id}.json"
        )

        payload = {
            "tenant": tenant,
            "project": project,
            "user": user,
            "conversation_id": conversation_id,
            "role": role,
            "text": text,
            "timestamp": msg_ts + "Z",
            "embedding": embedding,  # persisted for perfect reindex
            "meta": {
                "message_id": message_id,
                "user_type": user_type,
                "ttl_days": int(ttl_days),
                **(meta or {})
            }
        }
        # backend is path-relative; we just write bytes at rel
        self.backend.write_bytes(rel, json.dumps(payload, ensure_ascii=False).encode("utf-8"))
        return self._uri_for_path(rel), message_id

    def list_conversation(
        self,
        *,
        tenant: str,
        project: str,
        user_type: str,
        user_or_fp: str,
        conversation_id: str
    ) -> list[dict]:
        """
        Load all JSON message blobs for a conversation.
        """
        base = self._join(
            self.root_prefix, "tenants", tenant, "projects", project,
            "conversation", user_type, user_or_fp, conversation_id
        )
        out: list[dict] = []
        for key in self.backend.list_keys(prefix=base):
            if not key.endswith(".json"):
                continue
            try:
                raw = self.backend.read_bytes(key).decode("utf-8")
                obj = json.loads(raw)
                # carry back s3/file uri for lineage
                obj.setdefault("meta", {})["s3_uri"] = self._uri_for_path(key)
                out.append(obj)
            except Exception:
                continue
        # sort by timestamp if present
        out.sort(key=lambda x: x.get("timestamp", ""))
        return out

    async def close(self):
        return None
