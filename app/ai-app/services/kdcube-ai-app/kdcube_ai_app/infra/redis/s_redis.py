# SPDX-License-Identifier: MIT
# SSH tunnel + Redis client: list or delete keys on a remote dockerized Redis.
from __future__ import annotations
import os, socket, time, subprocess, json
from contextlib import contextmanager
from typing import Tuple, List, Dict, Any, Optional

import redis
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# ── SSH TUNNEL HELPERS ─────────────────────────────────────────────────────────
def _get_free_port(host: str = "127.0.0.1") -> int:
    s = socket.socket()
    try:
        s.bind((host, 0))
        return s.getsockname()[1]
    finally:
        s.close()

@contextmanager
def ssh_local_forward(
        *,
        ssh_host: str,
        ssh_user: str,
        ssh_key_path: str,
        remote_host: str,
        remote_port: int,
        ssh_port: int = 22,
        local_host: str = "127.0.0.1",
        wait_timeout_s: float = 10.0,
) -> Tuple[str, int]:
    """
    Starts: ssh -N -L <local_port>:<remote_host>:<remote_port> ssh_user@ssh_host
    Yields (local_host, local_port). Kills the tunnel on exit.
    """
    local_port = _get_free_port(local_host)
    cmd = [
        "ssh",
        "-N",
        "-L", f"{local_port}:{remote_host}:{remote_port}",
        "-o", "BatchMode=yes",
        "-o", "ExitOnForwardFailure=yes",
        "-o", "ServerAliveInterval=15",
        "-o", "ServerAliveCountMax=3",
        "-o", "StrictHostKeyChecking=accept-new",
        "-i", ssh_key_path,
        "-p", str(ssh_port),
        f"{ssh_user}@{ssh_host}",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Wait until local side accepts connections
    deadline = time.time() + wait_timeout_s
    last_err = None
    while time.time() < deadline:
        try:
            with socket.create_connection((local_host, local_port), timeout=0.5):
                break
        except Exception as e:
            last_err = e
            if proc.poll() is not None:  # ssh died early
                _, err = proc.communicate(timeout=1)
                raise RuntimeError(f"SSH tunnel failed to start:\n{err.decode(errors='ignore')}")
            time.sleep(0.2)
    else:
        proc.terminate()
        raise TimeoutError(f"Tunnel did not come up on {local_host}:{local_port}: {last_err}")

    try:
        yield (local_host, local_port)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

# ── REDIS OVER SSH ─────────────────────────────────────────────────────────────
def _connect_redis(host: str, port: int, db: int, password: str) -> redis.Redis:
    return redis.Redis(
        host=host,
        port=port,
        db=db,
        password=password,
        decode_responses=True,     # return str, not bytes
        socket_timeout=5,
        socket_connect_timeout=5,
    )

def _get_envs():
    # SSH
    ssh_host = os.environ["BASTION_HOST"]
    ssh_user = os.environ["BASTION_USER"]
    ssh_key  = os.environ["SSH_KEYPATH"]
    ssh_port = int(os.environ.get("BASTION_PORT", "22"))
    # Redis exposed on remote host
    remote_host = os.environ.get("REMOTE_REDIS_HOST", "127.0.0.1")
    remote_port = int(os.environ.get("REMOTE_REDIS_PORT", "5445"))  # because "5445:6379"
    # Redis auth/db
    redis_password = os.environ["REDIS_PASSWORD"]
    redis_db = int(os.environ.get("REDIS_DB", "0"))
    return ssh_host, ssh_user, ssh_key, ssh_port, remote_host, remote_port, redis_password, redis_db

# ---- LIST KEYS ----
def list_keys_by_pattern_via_ssh(
        pattern: str,
        *,
        limit: Optional[int] = None,
        with_meta: bool = False,
        as_json: bool = False,
) -> List[str] | List[Dict[str, Any]]:
    """
    Lists keys matching 'pattern' via SSH tunnel.
    If with_meta=True, returns dicts with type and size where possible.
    """
    (ssh_host, ssh_user, ssh_key, ssh_port,
     remote_host, remote_port, redis_password, redis_db) = _get_envs()

    with ssh_local_forward(
            ssh_host=ssh_host,
            ssh_user=ssh_user,
            ssh_key_path=ssh_key,
            ssh_port=ssh_port,
            remote_host=remote_host,
            remote_port=remote_port,
    ) as (lh, lp):
        r = _connect_redis(lh, lp, redis_db, redis_password)
        r.ping()

        keys: List[str] = []
        for k in r.scan_iter(match=pattern, count=1000):
            keys.append(k)
            if limit is not None and len(keys) >= limit:
                break

        if not with_meta:
            if as_json:
                print(json.dumps(keys, indent=2))
            else:
                print("\n".join(keys) if keys else f"<no keys match {pattern!r}>")
            r.connection_pool.disconnect()
            return keys

        # Meta: type + size where we can in one or two pipeline passes
        meta: List[Dict[str, Any]] = []
        if not keys:
            if as_json:
                print("[]")
            else:
                print(f"<no keys match {pattern!r}>")
            r.connection_pool.disconnect()
            return meta

        # Pass 1: TYPE for all
        p = r.pipeline()
        for k in keys:
            p.type(k)  # returns "string","list","set","zset","hash","stream"...
        types = p.execute()

        # Pass 2: size per key type
        p = r.pipeline()
        for k, t in zip(keys, types):
            if t == "string":
                p.strlen(k)
            elif t == "list":
                p.llen(k)
            elif t == "set":
                p.scard(k)
            elif t == "zset":
                p.zcard(k)
            elif t == "hash":
                p.hlen(k)
            elif t == "stream":
                # xinfo_stream → dict with 'length'
                p.xinfo_stream(k)
            else:
                p.exists(k)  # fallback; 1 or 0
        sizes = p.execute()

        for k, t, sz in zip(keys, types, sizes):
            if t == "stream" and isinstance(sz, dict) and "length" in sz:
                size_val = sz.get("length")
            else:
                size_val = sz
            meta.append({"key": k, "type": t, "size": size_val})

        if as_json:
            print(json.dumps(meta, indent=2))
        else:
            for row in meta:
                print(f"{row['key']}  type={row['type']}  size={row['size']}")

        r.connection_pool.disconnect()
        return meta

# ---- DELETE KEYS ----
def delete_keys_by_pattern_via_ssh(pattern: str) -> int:
    """
    Opens an SSH tunnel, deletes all keys matching 'pattern', returns count deleted.
    Uses UNLINK when supported; fallback to DEL.
    """
    (ssh_host, ssh_user, ssh_key, ssh_port,
     remote_host, remote_port, redis_password, redis_db) = _get_envs()

    with ssh_local_forward(
            ssh_host=ssh_host,
            ssh_user=ssh_user,
            ssh_key_path=ssh_key,
            ssh_port=ssh_port,
            remote_host=remote_host,
            remote_port=remote_port,
    ) as (lh, lp):
        r = _connect_redis(lh, lp, redis_db, redis_password)
        r.ping()

        to_delete: List[str] = list(r.scan_iter(match=pattern, count=1000))
        if not to_delete:
            print(f"No keys match: {pattern!r}")
            r.connection_pool.disconnect()
            return 0

        deleted = 0
        try:
            for i in range(0, len(to_delete), 1000):
                chunk = to_delete[i:i+1000]
                deleted += r.unlink(*chunk)
        except redis.exceptions.ResponseError:
            for i in range(0, len(to_delete), 1000):
                chunk = to_delete[i:i+1000]
                deleted += r.delete(*chunk)

        print(f"Deleted {deleted} keys matching {pattern!r}")
        r.connection_pool.disconnect()
        return deleted

# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    op = os.environ.get("REDIS_OP", "list").strip().lower()
    op = "delete"
    if op == "delete":
        # "kdcube:config:bundles:mapping:*"
        pattern = os.environ.get("REDIS_DELETE_PATTERN") or os.environ.get("REDIS_PATTERN") or "*"
        pattern = "kdcube:config:bundles:mapping:*"
        delete_keys_by_pattern_via_ssh(pattern)
    else:
        # list
        pattern = os.environ.get("REDIS_LIST_PATTERN") or os.environ.get("REDIS_PATTERN") or "*"
        limit = os.environ.get("REDIS_LIST_LIMIT")
        limit_int = int(limit) if (limit and limit.isdigit()) else None
        with_meta = os.environ.get("REDIS_LIST_META", "0") in ("1", "true", "yes", "on")
        as_json = os.environ.get("REDIS_LIST_JSON", "0") in ("1", "true", "yes", "on")
        list_keys_by_pattern_via_ssh(pattern, limit=limit_int, with_meta=with_meta, as_json=as_json)

