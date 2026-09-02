# SPDX-License-Identifier: MIT
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DockerBuildStoragePolicy:
    max_cache_size: str = "12GB"
    min_free_space: str = "8GB"
    legacy_cache_age: str = "24h"


DEFAULT_DOCKER_BUILD_STORAGE_POLICY = DockerBuildStoragePolicy()


def builder_cache_prune_command(
    help_text: str,
    *,
    policy: DockerBuildStoragePolicy = DEFAULT_DOCKER_BUILD_STORAGE_POLICY,
) -> tuple[str, ...]:
    """Select bounded cache cleanup flags supported by the local Docker CLI."""
    base = ("docker", "builder", "prune", "-f")
    help_text = str(help_text or "")
    if "--max-used-space" in help_text and "--min-free-space" in help_text:
        return (
            *base,
            "--max-used-space",
            policy.max_cache_size,
            "--min-free-space",
            policy.min_free_space,
        )
    if "--keep-storage" in help_text:
        return (*base, "--keep-storage", policy.max_cache_size)
    return (*base, "--filter", f"until={policy.legacy_cache_age}")


def build_storage_maintenance_commands(
    builder_help_text: str,
    *,
    policy: DockerBuildStoragePolicy = DEFAULT_DOCKER_BUILD_STORAGE_POLICY,
) -> tuple[tuple[str, ...], ...]:
    """Return safe cleanup commands; named images and volumes remain untouched."""
    return (
        ("docker", "image", "prune", "-f"),
        builder_cache_prune_command(builder_help_text, policy=policy),
    )
