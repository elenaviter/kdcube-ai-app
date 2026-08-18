# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter
"""A branch clone must be able to stop trusting its own cache marker.

LIVE: the press app's store clones sat 114 commits behind origin while every
"Sync now" reported success. `git_bundle_cache_status` calls a clone current
when the marker's commit equals local HEAD — always true after any
materialization — so for a BRANCH source `ensure_git_bundle` returned before
fetching, every time, and the clone froze at whatever the remote held on the
day it was created. `invalidate_git_bundle_cache` is the named front door a
caller uses (after ITS OWN dirty/ahead guard) to make the next ensure fetch.
"""
from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path

import contextlib

import pytest

import kdcube_ai_app.infra.plugin.git_bundle as git_bundle
from kdcube_ai_app.infra.plugin.git_bundle import (
    compute_git_bundle_paths,
    git_bundle_cache_status,
)


@pytest.fixture()
def invalidate(monkeypatch):
    """The helper with its distributed locks faked out, as the locking suite
    does — a unit test has no Redis to wait two minutes on."""

    @contextlib.asynccontextmanager
    async def _no_lock(**_kwargs):
        yield

    monkeypatch.setattr(git_bundle, "_async_redis_bundle_lock", _no_lock)
    monkeypatch.setattr(git_bundle, "_async_bundle_lock", _no_lock)
    return git_bundle.invalidate_git_bundle_cache


def _init_repo(path: Path) -> str:
    path.mkdir(parents=True, exist_ok=True)
    for args in (
        ["git", "init", "--initial-branch", "main"],
        ["git", "config", "user.email", "test@example.com"],
        ["git", "config", "user.name", "Test"],
    ):
        subprocess.run(args, cwd=path, check=True, capture_output=True)
    (path / "README.md").write_text("# store\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=path, check=True, capture_output=True)
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=path, check=True, capture_output=True, text=True
    )
    return head.stdout.strip()


def _materialized_clone(tmp_path: Path, *, bundle_id: str, url: str):
    """A clone in the exact shape ensure_git_bundle leaves behind: repo on
    disk, marker recording its HEAD."""
    paths = compute_git_bundle_paths(
        bundle_id=bundle_id, git_url=url, git_ref="main",
        git_subdir=None, bundles_root=tmp_path / "bundles",
    )
    paths.repo_root.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", url, str(paths.repo_root)], check=True, capture_output=True
    )
    head = subprocess.run(
        ["git", "-C", str(paths.repo_root), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    (paths.repo_root / ".kdcube.git-bundle.json").write_text(
        json.dumps({
            "schema": 1, "bundle_id": bundle_id, "git_url": url,
            "normalized_git_url": url, "git_ref": "main", "git_subdir": "",
            "commit": head,
        }),
        encoding="utf-8",
    )
    return paths


def test_a_branch_clone_reads_current_even_when_the_remote_moved(tmp_path):
    """The trap itself, pinned: origin advances, the cache still says current.

    This is the behaviour that froze the live clones — kept as a test so that
    if the platform semantics ever change, the invalidation callers get told.
    """
    origin = tmp_path / "origin"
    _init_repo(origin)
    url = str(origin)
    paths = _materialized_clone(tmp_path, bundle_id="press.store.test", url=url)

    (origin / "new-entry.md").write_text("new\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=origin, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "advance"], cwd=origin, check=True, capture_output=True)

    status = asyncio.run(git_bundle_cache_status(
        bundle_id="press.store.test", git_url=url, git_ref="main",
        git_subdir=None, bundles_root=tmp_path / "bundles",
    ))
    assert status.current is True  # the remote is never consulted


def test_invalidate_makes_the_cache_stale_so_the_next_ensure_fetches(tmp_path, invalidate):
    origin = tmp_path / "origin"
    _init_repo(origin)
    url = str(origin)
    _materialized_clone(tmp_path, bundle_id="press.store.test", url=url)

    removed = asyncio.run(invalidate(
        bundle_id="press.store.test", git_url=url, git_ref="main",
        git_subdir=None, bundles_root=tmp_path / "bundles",
    ))
    assert removed is True

    status = asyncio.run(git_bundle_cache_status(
        bundle_id="press.store.test", git_url=url, git_ref="main",
        git_subdir=None, bundles_root=tmp_path / "bundles",
    ))
    assert status.current is False
    assert status.reason == "missing_marker"


def test_invalidating_a_clone_that_was_never_materialized_is_a_no_op(tmp_path, invalidate):
    origin = tmp_path / "origin"
    _init_repo(origin)
    removed = asyncio.run(invalidate(
        bundle_id="press.store.test", git_url=str(origin), git_ref="main",
        git_subdir=None, bundles_root=tmp_path / "bundles",
    ))
    assert removed is False
