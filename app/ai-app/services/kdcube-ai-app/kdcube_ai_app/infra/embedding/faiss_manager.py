# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

# faiss_cache.py
import os
import threading
import gc
import fcntl
from collections import OrderedDict
from datetime import datetime

import faiss
import redis

from kdcube_ai_app.storage.storage import IStorageBackend, create_storage_backend
from kdcube_ai_app.apps.chat.sdk.config import get_settings
from kdcube_ai_app.infra.redis.client import get_sync_redis_client


class FaissProjectCache:
    """
    Cross-process, file-lock protected LRU cache of FAISS indices.
    Supports pluggable storage via IStorageBackend (S3 or local FS).
    """

    def __init__(
            self,
            storage: IStorageBackend,
            max_loaded: int = 3,
            redis_url: str | None = None,
    ):
        if not redis_url:
            redis_url = get_settings().REDIS_URL
        self._lock       = threading.RLock()
        self._cond       = threading.Condition(self._lock)
        self._cache      = OrderedDict()   # project → { idx, ref_count, lock_fd }
        self._max_loaded = max_loaded

        self.storage = storage
        self._redis  = get_sync_redis_client(redis_url)

        # Start watcher for external rebuilds
        threading.Thread(target=self._watch_updates, daemon=True).start()

    def _pointer_key(self, project: str) -> str:
        return f"kdcube:faiss:{project}:index_s3_key"

    def _pubsub_channel(self, project: str) -> str:
        return f"kdcube:faiss:{project}:updates"

    def _lockfile_path(self, project: str) -> str:
        return os.path.join("/tmp", f"faiss-{project}.lock")

    def _download_lockfile_path(self, project: str) -> str:
        return os.path.join("/tmp", f"faiss-{project}.download.lock")

    def publish_new_index(self, project: str, idx: faiss.Index):
        """
        1) write idx → /tmp
        2) persist via storage.write_bytes()
        3) set Redis pointer + PUBLISH update
        4) evict any in-memory copy
        """
        # 1) write locally
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        tmp_path  = f"/tmp/{project}-{timestamp}.faiss"
        faiss.write_index(idx, tmp_path)

        # 2) read bytes + persist
        with open(tmp_path, "rb") as f:
            data = f.read()
        storage_key = f"{project}/faiss-indexes/{timestamp}.faiss"
        self.storage.write_bytes(storage_key, data)

        # 3) update pointer & notify
        self._redis.set(self._pointer_key(project), storage_key)
        self._redis.publish(self._pubsub_channel(project), storage_key)

        # 4) drop any in-memory cache so next load picks up new file
        with self._lock:
            if project in self._cache:
                ent = self._cache.pop(project)
                os.close(ent["lock_fd"])
                del ent["idx"]
                gc.collect()

    class _Usage:
        def __init__(self, parent, project: str):
            self._p    = parent
            self._proj = project

        def __enter__(self) -> faiss.Index:
            with self._p._lock:
                # already cached?
                if self._proj in self._p._cache:
                    ent = self._p._cache.pop(self._proj)
                    ent["ref_count"] += 1
                    self._p._cache[self._proj] = ent  # bump LRU
                    return ent["idx"]

                # load pointer
                skey = self._p._redis.get(self._p._pointer_key(self._proj))
                if not skey:
                    raise RuntimeError(f"No FAISS index for project '{self._proj}'")
                storage_key = skey.decode()

                # decide local tmp path
                local_path = f"/tmp/{self._proj}-{os.path.basename(storage_key)}"

                # —— DOWNLOAD LOCK ——
                dlock_fd = os.open(
                    self._p._download_lockfile_path(self._proj),
                    os.O_CREAT | os.O_RDWR
                )
                fcntl.flock(dlock_fd, fcntl.LOCK_EX)
                try:
                    if not os.path.exists(local_path):
                        # fetch from storage backend
                        data = self._p.storage.read_bytes(storage_key)
                        with open(local_path, "wb") as f:
                            f.write(data)
                finally:
                    fcntl.flock(dlock_fd, fcntl.LOCK_UN)
                    os.close(dlock_fd)
                # —— end download lock ——

                # mmap the index
                print("About to read FAISS index from:", local_path)
                idx = faiss.read_index(
                    local_path,
                    faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY
                )

                # acquire shared lock so eviction sees it in use
                lock_fd = os.open(
                    self._p._lockfile_path(self._proj),
                    os.O_CREAT | os.O_RDWR
                )
                fcntl.flock(lock_fd, fcntl.LOCK_SH)

                ent = {"idx": idx, "ref_count": 1, "lock_fd": lock_fd}
                self._p._cache[self._proj] = ent
                self._p._evict_if_needed()
                return idx

        def __exit__(self, exc_type, exc, tb):
            with self._p._lock:
                ent = self._p._cache[self._proj]
                ent["ref_count"] -= 1
                if ent["ref_count"] == 0:
                    self._p._cond.notify_all()

    def get(self, project: str):
        """
        Use as:
            with cache.get("myproj") as idx:
                D,I = idx.search(...)
        """
        return FaissProjectCache._Usage(self, project)

    def _evict_if_needed(self):
        while len(self._cache) > self._max_loaded:
            evicted = False
            for proj, ent in list(self._cache.items()):
                if ent["ref_count"] != 0:
                    continue

                # try exclusive eviction lock
                ex_fd = os.open(self._lockfile_path(proj), os.O_RDWR)
                try:
                    fcntl.flock(ex_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    os.close(ex_fd)
                    continue

                # safe to evict
                os.close(ex_fd)
                popped = self._cache.pop(proj)
                os.close(popped["lock_fd"])
                del popped["idx"]
                gc.collect()
                evicted = True
                break

            if not evicted:
                self._cond.wait()

    def _watch_updates(self):
        ps = self._redis.pubsub(ignore_subscribe_messages=True)
        ps.psubscribe("kdcube:faiss:*:updates")
        for msg in ps.listen():
            if msg["type"] == "pmessage":
                chan    = msg["channel"].decode()
                project= chan.split(":",2)[1]
                with self._lock:
                    if project in self._cache:
                        ent = self._cache.pop(project)
                        os.close(ent["lock_fd"])
                        del ent["idx"]
                        gc.collect()
