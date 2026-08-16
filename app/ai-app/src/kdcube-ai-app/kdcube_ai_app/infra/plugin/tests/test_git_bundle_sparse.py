"""Sparse cloning of a bundle store: only the folder the bundle uses.

A store that lives in one directory of a large repository used to cost the
whole repository, once per bundle that pointed at it. These exercise the clone
against a real local repository — no network, no mocking of git itself — and
they exist mainly to prove the SAFETY properties: off by default, and a plain
clone whenever anything about sparse does not work.
"""

from __future__ import annotations

import asyncio
import os
import pathlib
import subprocess

import pytest

from kdcube_ai_app.infra.plugin import git_bundle


class _Log:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def log(self, message: str, level: str = "INFO") -> None:
        self.lines.append(f"{level}:{message}")


def _git(*args: str, cwd: pathlib.Path) -> None:
    subprocess.run(
        ["git", *args], cwd=str(cwd), check=True,
        capture_output=True, text=True,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
    )


@pytest.fixture()
def source_repo(tmp_path: pathlib.Path) -> pathlib.Path:
    """A repository with two content folders, one of which a bundle wants."""
    repo = tmp_path / "source"
    (repo / "publications" / "wanted").mkdir(parents=True)
    (repo / "publications" / "unwanted").mkdir(parents=True)
    (repo / "publications" / "wanted" / "entry.md").write_text("the store\n")
    (repo / "publications" / "unwanted" / "big.md").write_text("x" * 4096 + "\n")
    _git("init", "-q", "-b", "main", cwd=repo)
    _git("config", "user.email", "t@example.com", cwd=repo)
    _git("config", "user.name", "T", cwd=repo)
    # Local transport refuses a partial clone unless the source allows it —
    # which is exactly the condition the fallback exists for, so it is set
    # here deliberately and unset in the fallback test.
    _git("config", "uploadpack.allowFilter", "true", cwd=repo)
    _git("add", "-A", cwd=repo)
    _git("commit", "-qm", "first", cwd=repo)
    return repo


def _clone(source: pathlib.Path, dest: pathlib.Path, subdir: str | None,
           sparse: bool | None = None) -> _Log:
    log = _Log()
    asyncio.run(git_bundle._clone_maybe_sparse(
        git_url=str(source), dest=dest, git_subdir=subdir,
        depth=None, logger=log, env=None, sparse=sparse,
    ))
    return log


def test_off_by_default_clones_everything(source_repo, tmp_path, monkeypatch):
    """Nothing changes for a deployment that has not asked for this."""
    monkeypatch.delenv("BUNDLE_GIT_SPARSE", raising=False)
    dest = tmp_path / "clone"
    _clone(source_repo, dest, "publications/wanted")
    assert (dest / "publications" / "wanted" / "entry.md").is_file()
    assert (dest / "publications" / "unwanted" / "big.md").is_file()


def test_sparse_fetches_only_the_subdir(source_repo, tmp_path, monkeypatch):
    monkeypatch.setenv("BUNDLE_GIT_SPARSE", "1")
    dest = tmp_path / "clone"
    log = _clone(source_repo, dest, "publications/wanted")
    # The store is there in full…
    assert (dest / "publications" / "wanted" / "entry.md").read_text() == "the store\n"
    # …and the rest of the repository is not checked out at all.
    assert not (dest / "publications" / "unwanted").exists()
    assert any("sparse clone" in line for line in log.lines), log.lines


def test_a_repo_that_refuses_filtering_still_clones(tmp_path, monkeypatch):
    """The fallback is the point: a bundle that starts beats one that does not."""
    repo = tmp_path / "plain-source"
    (repo / "publications" / "wanted").mkdir(parents=True)
    (repo / "publications" / "wanted" / "entry.md").write_text("the store\n")
    (repo / "elsewhere").mkdir()
    (repo / "elsewhere" / "other.md").write_text("other\n")
    _git("init", "-q", "-b", "main", cwd=repo)
    _git("config", "user.email", "t@example.com", cwd=repo)
    _git("config", "user.name", "T", cwd=repo)
    _git("config", "uploadpack.allowFilter", "false", cwd=repo)
    _git("add", "-A", cwd=repo)
    _git("commit", "-qm", "first", cwd=repo)

    monkeypatch.setenv("BUNDLE_GIT_SPARSE", "1")
    dest = tmp_path / "clone"
    log = _clone(repo, dest, "publications/wanted")
    # Whatever git said about filtering, the bundle has its store.
    assert (dest / "publications" / "wanted" / "entry.md").is_file()
    if any("unavailable" in line for line in log.lines):
        # It fell back, so it fell back to a COMPLETE clone.
        assert (dest / "elsewhere" / "other.md").is_file()


def test_no_subdir_is_never_sparse(source_repo, tmp_path, monkeypatch):
    """A bundle whose store IS the repository has nothing to narrow to."""
    monkeypatch.setenv("BUNDLE_GIT_SPARSE", "1")
    dest = tmp_path / "clone"
    log = _clone(source_repo, dest, None)
    assert (dest / "publications" / "unwanted" / "big.md").is_file()
    assert not any("sparse clone" in line for line in log.lines), log.lines


def test_a_sparse_clone_still_fetches_and_resets(source_repo, tmp_path, monkeypatch):
    """The refresh lane runs on it unchanged — that is what makes this safe.

    A sparse checkout that could not be updated would trade disk for a store
    frozen at its first commit, which is not a trade worth making.
    """
    monkeypatch.setenv("BUNDLE_GIT_SPARSE", "1")
    dest = tmp_path / "clone"
    _clone(source_repo, dest, "publications/wanted")

    (source_repo / "publications" / "wanted" / "entry.md").write_text("moved on\n")
    _git("add", "-A", cwd=source_repo)
    _git("commit", "-qm", "second", cwd=source_repo)

    _git("fetch", "origin", "main", cwd=dest)
    _git("reset", "--hard", "origin/main", cwd=dest)
    assert (dest / "publications" / "wanted" / "entry.md").read_text() == "moved on\n"
    assert not (dest / "publications" / "unwanted").exists()


def test_the_caller_decides_not_the_environment(source_repo, tmp_path, monkeypatch):
    """`sparse=` is the contract; the env is only a fallback for the loader.

    Every knob in this system is a descriptor property read by the thing it
    configures. A caller that passes the parameter must win over whatever the
    environment happens to say, in both directions.
    """
    monkeypatch.setenv("BUNDLE_GIT_SPARSE", "0")
    on = tmp_path / "caller-says-yes"
    _clone(source_repo, on, "publications/wanted", sparse=True)
    assert not (on / "publications" / "unwanted").exists()

    monkeypatch.setenv("BUNDLE_GIT_SPARSE", "1")
    off = tmp_path / "caller-says-no"
    _clone(source_repo, off, "publications/wanted", sparse=False)
    assert (off / "publications" / "unwanted" / "big.md").is_file()


def test_no_caller_and_no_env_is_a_full_clone(source_repo, tmp_path, monkeypatch):
    monkeypatch.delenv("BUNDLE_GIT_SPARSE", raising=False)
    dest = tmp_path / "clone"
    _clone(source_repo, dest, "publications/wanted", sparse=None)
    assert (dest / "publications" / "unwanted" / "big.md").is_file()
