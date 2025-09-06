# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

# chatbot/storage/storage.py

import json, time, os, mimetypes, pathlib
from urllib.parse import urlparse
from typing import Optional, Tuple, List, Dict, Any

from kdcube_ai_app.apps.chat.sdk.config import get_settings
from kdcube_ai_app.apps.chat.sdk.inventory import _mid
from kdcube_ai_app.apps.chat.sdk.storage.rn import (
    rn_message, rn_attachment, rn_execution_file
)

try:
    from kdcube_ai_app.storage.storage import create_storage_backend
except ImportError:
    raise ImportError("Please ensure 'kdcube_ai_app.storage.storage' is importable.")

_JSON_META = {"ContentType": "application/json"}

class ConversationStore:
    """
    Root: ${KDCUBE_STORAGE_PATH}/cb
    Messages:
      cb/tenants/{tenant}/projects/{project}/conversation/{anonymous|registered}/{user_or_fp}/{conversation_id}/{turn_id}/{message_id}.json
    Attachments:
      cb/tenants/{tenant}/projects/{project}/attachments/{anonymous|registered}/{user_or_fp}/{conversation_id}/{turn_id}/{timestamp-filename}
    Executions:
      cb/tenants/{tenant}/projects/{project}/executions/{anonymous|registered}/{user_or_fp}/{conversation_id}/{turn_id}/{out|pkg}/...
    """

    def __init__(self, storage_uri: Optional[str] = None):
        self._settings = get_settings()
        self.storage_uri = storage_uri or self._settings.STORAGE_PATH
        self.backend = create_storage_backend(self.storage_uri)
        parsed = urlparse(self.storage_uri)
        self.scheme = parsed.scheme or "file"
        self.root_prefix = "cb"
        self._file_base = parsed.path if self.scheme == "file" else ""
        self._s3_bucket = parsed.netloc if self.scheme == "s3" else ""
        self._s3_prefix = parsed.path.lstrip("/") if self.scheme == "s3" else ""

    # ---------- helpers ----------

    def _join(self, *parts: str) -> str:
        return "/".join([p.strip("/").replace("//", "/") for p in parts if p])

    def _uri_for_path(self, relpath: str) -> str:
        if self.scheme == "file":
            base = self._file_base.rstrip("/")
            abs_path = self._join(base, relpath)
            return "file://" + abs_path
        if self.scheme == "s3":
            prefix = self._s3_prefix.rstrip("/")
            key = self._join(prefix, relpath)
            return f"s3://{self._s3_bucket}/{key}"
        return f"{self.scheme}://{relpath}"

    def _who_and_id(self, user: Optional[str], fingerprint: Optional[str]) -> Tuple[str, str]:
        who = "registered" if (user and user != "anonymous") else "anonymous"
        user_or_fp = user if who == "registered" else (fingerprint or "unknown")
        return who, user_or_fp

    def _sha256_bytes(self, data: bytes) -> str:
        import hashlib
        h = hashlib.sha256(); h.update(data); return h.hexdigest()

    # ---------- messages ----------

    def put_message(
        self,
        *,
        tenant: str,
        project: str,
        user: Optional[str],
        fingerprint: Optional[str],
        conversation_id: str,
        turn_id: str,
        role: str,
        text: str,
        payload: Any | None = None,
        meta: Dict | None = None,
        embedding: List[float] | None = None,
        user_type: str = "anonymous",
        ttl_days: int = 365
    ) -> Tuple[str, str, str]:
        """
        Persist a message JSON. Returns (uri, message_id, rn).
        RN is generated HERE and written into the record.
        """
        msg_ts = time.strftime("%Y-%m-%dT%H-%M-%S", time.gmtime())
        message_id = _mid(role, msg_ts)
        who, user_or_fp = self._who_and_id(user, fingerprint)

        rel = self._join(
            self.root_prefix, "tenants", tenant, "projects", project,
            "conversation", who, user_or_fp, conversation_id, turn_id,
            f"{message_id}.json"
        )

        rn = rn_message(tenant, project, user_or_fp, conversation_id, turn_id, role, message_id)

        record = {
            "tenant": tenant,
            "project": project,
            "user": user,
            "user_id": user_or_fp,   # stable owner id used in RN
            "conversation_id": conversation_id,
            "turn_id": turn_id,
            "role": role,
            "text": text,
            "timestamp": msg_ts + "Z",
            "embedding": embedding,
            "payload": payload,
            "meta": {
                "message_id": message_id,
                "turn_id": turn_id,
                "user_type": user_type,
                "ttl_days": int(ttl_days),
                "rn": rn,
                **(meta or {})
            }
        }
        self.backend.write_bytes(rel, json.dumps(record, ensure_ascii=False, indent=2).encode("utf-8"), meta=_JSON_META)
        return self._uri_for_path(rel), message_id, rn

    def list_conversation(
        self,
        *,
        tenant: str,
        project: str,
        user_type: str,
        user_or_fp: str,
        conversation_id: str,
        turn_id: Optional[str] = None,
    ) -> List[dict]:
        """
        Traverse directories using backend.list_dir(); loads *.json.
        If turn_id is None, loads messages across all turns.
        """
        base_conv = self._join(
            self.root_prefix, "tenants", tenant, "projects", project,
            "conversation", user_type, user_or_fp, conversation_id
        )

        def _as_child(base: str, name: str) -> str:
            return name if name.startswith(base) else self._join(base, name)

        def _collect_turn(turn_path: str) -> List[dict]:
            out: List[dict] = []
            for item in self.backend.list_dir(turn_path):
                child = _as_child(turn_path, item)
                if child.endswith(".json"):
                    try:
                        raw = self.backend.read_text(child)
                        obj = json.loads(raw)
                        obj.setdefault("meta", {})["s3_uri"] = self._uri_for_path(child)
                        if "turn_id" not in obj:
                            # .../conversation/<who>/<user>/<conv>/<turn>/<message>.json
                            parts = child.strip("/").split("/")
                            try:
                                i = parts.index("conversation")
                                obj["turn_id"] = parts[i+4]
                                obj.setdefault("meta", {})["turn_id"] = obj["turn_id"]
                            except Exception:
                                pass
                        out.append(obj)
                    except Exception:
                        continue
            out.sort(key=lambda x: x.get("timestamp", ""))
            return out

        if turn_id:
            return _collect_turn(self._join(base_conv, turn_id))

        out_all: List[dict] = []
        for item in self.backend.list_dir(base_conv):
            child = _as_child(base_conv, item)
            if child.endswith(".json"):
                try:
                    raw = self.backend.read_text(child)
                    obj = json.loads(raw)
                    obj.setdefault("meta", {})["s3_uri"] = self._uri_for_path(child)
                    out_all.append(obj)
                except Exception:
                    pass
            else:
                out_all.extend(_collect_turn(child))
        out_all.sort(key=lambda x: x.get("timestamp", ""))
        return out_all

    # ---------- attachments (role-aware, turn in path) ----------

    def put_attachment(
        self,
        *,
        tenant: str,
        project: str,
        user: Optional[str],
        fingerprint: Optional[str],
        conversation_id: str,
        turn_id: str,
        role: str = "artifact",
        filename: str,
        data: bytes,
        mime: Optional[str] = None,
        user_type: Optional[str] = None,
        ttl_days: int = 365,
        request_id: Optional[str] = None,
    ) -> Tuple[str, str, str]:
        """
        Save a binary/text file under /attachments/.../{conversation_id}/{turn_id}/.
        Returns (uri, key, rn). RN includes user_id and role.
        """
        if not turn_id:
            raise ValueError("turn_id is required for attachments")

        ts = time.strftime("%Y%m%d%H%M%S", time.gmtime())
        who, user_or_fp = self._who_and_id(user, fingerprint)
        base = self._join(
            self.root_prefix, "tenants", tenant, "projects", project,
            "attachments", who, user_or_fp, conversation_id, turn_id
        )
        safe_name = os.path.basename(filename) or "file.bin"
        rel_name = f"{ts}-{safe_name}"
        rel = self._join(base, rel_name)

        meta = {"ContentType": mime} if mime else None
        self.backend.write_bytes(rel, data, meta=meta)

        # RN is the logical filename (without timestamp) OR the actual stored name?
        # To keep dereferencing simple, we use the stored name.
        rn = rn_attachment(tenant, project, user_or_fp, conversation_id, turn_id, role, rel_name)
        return self._uri_for_path(rel), rel, rn

    # ---------- execution snapshot (role-aware RNs in manifest) ----------

    def put_execution_snapshot(
        self,
        *,
        tenant: str,
        project: str,
        user: Optional[str],
        fingerprint: Optional[str],
        conversation_id: str,
        turn_id: str,
        role: str = "artifact",
        out_dir: Optional[str] = None,
        pkg_dir: Optional[str] = None,
    ) -> dict:
        """
        Copy /out and /pkg trees under /executions/.../{turn_id}/.
        Returns a manifest with RN per file.
        """
        who, user_or_fp = self._who_and_id(user, fingerprint)
        base = self._join(
            self.root_prefix, "tenants", tenant, "projects", project,
            "executions", who, user_or_fp, conversation_id, turn_id
        )

        def _copy_tree(src: Optional[str], kind: str) -> Tuple[Optional[str], List[dict]]:
            if not src:
                return None, []
            srcp = pathlib.Path(src)
            if not srcp.exists():
                return None, []
            root_rel = self._join(base, kind)
            files_meta: List[dict] = []
            for p in srcp.rglob("*"):
                if not p.is_file():
                    continue
                rel_under = str(p.relative_to(srcp)).replace("\\", "/")
                key = self._join(root_rel, rel_under)
                data = p.read_bytes()
                ctype = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
                self.backend.write_bytes(key, data, meta={"ContentType": ctype})
                url = self._uri_for_path(key)

                files_meta.append({
                    "key": key,
                    "url": url,
                    "size": len(data),
                    "sha256": self._sha256_bytes(data),
                    "mime": ctype,
                    "kind": kind,
                    "rn": rn_execution_file(tenant, project, user_or_fp, conversation_id, turn_id, role, kind, rel_under)
                })
            return self._uri_for_path(root_rel), files_meta

        out_root, out_files = _copy_tree(out_dir, "out")
        pkg_root, pkg_files = _copy_tree(pkg_dir, "pkg")
        return {"roots": {"out": out_root, "pkg": pkg_root}, "files": out_files + pkg_files}

    async def close(self):
        return None
