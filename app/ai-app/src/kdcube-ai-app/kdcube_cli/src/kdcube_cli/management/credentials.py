from __future__ import annotations

from typing import Any

from kdcube_cli.management.errors import ManagementCliError


def normalize_bearer(value: Any) -> str:
    """Return one opaque bearer value without accepting header syntax."""

    if not isinstance(value, str):
        candidate = ""
    else:
        candidate = value.strip()
    if (
        not candidate
        or len(candidate) > 65536
        or candidate.lower().startswith("bearer ")
        or any(not 0x21 <= ord(character) <= 0x7E for character in candidate)
    ):
        raise ManagementCliError(
            "management_credential_invalid",
            "The delegated KDCube credential is invalid.",
        )
    return candidate


__all__ = ["normalize_bearer"]
