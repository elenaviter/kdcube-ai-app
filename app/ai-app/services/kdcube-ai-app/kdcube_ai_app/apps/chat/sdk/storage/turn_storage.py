# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

# chat/sdk/storage/turn_storage.py

from typing import Optional, Dict, Any
import os, json, mimetypes
import hashlib, shutil, datetime, pathlib

# ---------- External storage (turn-wide) ----------

class _LocalTurnStore:
    """
    Very small storage: copies files into a configured root under {request_id}/...
    and builds URLs using an optional BASE_URL. Good enough for dev/test.
    """
    def __init__(self, root_dir: Optional[str] = None, base_url: Optional[str] = None):
        self.root = pathlib.Path(root_dir or os.getenv("TOOLMGR_STORE_DIR", "./var/chat_store")).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.base_url = (base_url or os.getenv("TOOLMGR_STORE_BASEURL") or "").rstrip("/")

    def _dst(self, request_id: str, rel: str) -> pathlib.Path:
        safe = rel.lstrip("/").replace("\\", "/")
        return (self.root / request_id / safe).resolve()

    def _url_for(self, request_id: str, rel: str, dst: pathlib.Path) -> str:
        if self.base_url:
            return f"{self.base_url}/{request_id}/{rel.lstrip('/')}"
        return f"file://{dst}"

    def _sha256(self, p: pathlib.Path) -> str:
        h = hashlib.sha256()
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    async def save_file(self, *, request_id: str, src_path: pathlib.Path, dest_rel: str) -> Dict[str, Any]:
        dst = self._dst(request_id, dest_rel)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dst)
        size = dst.stat().st_size
        meta = {
            "key": f"{request_id}/{dest_rel}",
            "url": self._url_for(request_id, dest_rel, dst),
            "size": size,
            "sha256": self._sha256(dst),
            "mime": mimetypes.guess_type(str(dst))[0] or "application/octet-stream",
        }
        return meta

    async def save_text(self, *, request_id: str, text: str, dest_rel: str, mime: str = "text/plain") -> Dict[str, Any]:
        dst = self._dst(request_id, dest_rel)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(text, encoding="utf-8")
        size = dst.stat().st_size
        meta = {
            "key": f"{request_id}/{dest_rel}",
            "url": self._url_for(request_id, dest_rel, dst),
            "size": size,
            "sha256": self._sha256(dst),
            "mime": mime,
        }
        return meta

    async def save_json(self, *, request_id: str, obj: Dict[str, Any], dest_rel: str) -> Dict[str, Any]:
        text = json.dumps(obj, ensure_ascii=False, indent=2)
        return await self.save_text(request_id=request_id, text=text, dest_rel=dest_rel, mime="application/json")
