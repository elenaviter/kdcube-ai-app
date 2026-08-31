# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


_BUILD_CONTEXT_ROOT = Path("app/ai-app/deployment/docker/local-python-packages")
_CONTAINER_ROOT = Path("/tmp/kdcube-local-python-packages")
_IGNORED_SOURCE_NAMES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "venv",
}


@dataclass(frozen=True)
class LocalPythonPackageSource:
    distribution: str
    source: Path


def _canonical_distribution(value: str) -> str:
    return re.sub(r"[-_.]+", "-", str(value or "").strip().lower())


def parse_local_python_package_sources(
    values: Iterable[str] | None,
) -> tuple[LocalPythonPackageSource, ...]:
    """Parse trusted maintainer overrides in ``distribution=source`` form."""
    parsed: list[LocalPythonPackageSource] = []
    seen: set[str] = set()
    for raw_value in values or ():
        raw = str(raw_value or "").strip()
        if "=" not in raw:
            raise ValueError(
                "Local Python package overrides use DIST=SOURCE_DIR, "
                f"got: {raw or '<empty>'}"
            )
        distribution_raw, source_raw = raw.split("=", 1)
        distribution = _canonical_distribution(distribution_raw)
        if not distribution or not re.fullmatch(r"[a-z0-9][a-z0-9-]*", distribution):
            raise ValueError(f"Invalid Python distribution name: {distribution_raw!r}")
        if distribution in seen:
            raise ValueError(f"Duplicate local Python package override: {distribution}")
        source = Path(source_raw).expanduser().resolve()
        if not source.is_dir():
            raise ValueError(f"Local Python package source is not a directory: {source}")
        if not ((source / "pyproject.toml").is_file() or (source / "setup.py").is_file()):
            raise ValueError(
                "Local Python package source needs pyproject.toml or setup.py: "
                f"{source}"
            )
        seen.add(distribution)
        parsed.append(LocalPythonPackageSource(distribution=distribution, source=source))
    return tuple(parsed)


def _ignore_source_entry(_directory: str, names: Sequence[str]) -> set[str]:
    ignored = {name for name in names if name in _IGNORED_SOURCE_NAMES}
    ignored.update(name for name in names if name.endswith((".egg-info", ".pyc")))
    return ignored


def stage_local_python_package_sources(
    repo_root: Path,
    packages: Sequence[LocalPythonPackageSource],
) -> Path:
    """Copy local package sources into the staged platform Docker context."""
    stage_root = Path(repo_root).resolve() / _BUILD_CONTEXT_ROOT
    clear_local_python_package_sources(repo_root)
    stage_root.mkdir(parents=True, exist_ok=True)
    sources_root = stage_root / "sources"
    sources_root.mkdir()

    requirements: list[str] = []
    manifest: list[dict[str, str]] = []
    for package in packages:
        destination = sources_root / package.distribution
        shutil.copytree(
            package.source,
            destination,
            ignore=_ignore_source_entry,
        )
        requirements.append(
            str(_CONTAINER_ROOT / "sources" / package.distribution)
        )
        manifest.append(
            {
                "distribution": package.distribution,
                "source": str(package.source),
            }
        )

    (stage_root / "requirements.txt").write_text(
        "".join(f"{requirement}\n" for requirement in requirements),
        encoding="utf-8",
    )
    (stage_root / "manifest.json").write_text(
        json.dumps({"packages": manifest}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return stage_root


def clear_local_python_package_sources(repo_root: Path) -> None:
    """Remove transient local-package material while preserving the hook."""
    stage_root = Path(repo_root).resolve() / _BUILD_CONTEXT_ROOT
    shutil.rmtree(stage_root / "sources", ignore_errors=True)
    for filename in ("requirements.txt", "manifest.json"):
        (stage_root / filename).unlink(missing_ok=True)
