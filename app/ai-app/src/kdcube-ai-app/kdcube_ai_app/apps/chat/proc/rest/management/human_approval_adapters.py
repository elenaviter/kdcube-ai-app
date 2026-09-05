# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""Descriptor-selected stronger human-approval adapters."""

from __future__ import annotations

from fastapi import Request
from kdcube_ai_app.apps.chat.proc.rest.management.human_approval import (
    USER_VERIFICATION,
    HumanApprovalVerifier,
)
from kdcube_ai_app.apps.chat.proc.rest.management.human_approval_oidc import (
    OidcFreshAuthenticationVerifier,
)


def descriptor_human_approval_verifier(
    request: Request,
    *,
    required_assurance: str,
) -> HumanApprovalVerifier:
    del request
    if required_assurance == USER_VERIFICATION:
        from kdcube_ai_app.apps.chat.proc.rest.management.human_approval_webauthn import (
            WebAuthnHumanApprovalVerifier,
        )

        return WebAuthnHumanApprovalVerifier()
    return OidcFreshAuthenticationVerifier()


__all__ = ["descriptor_human_approval_verifier"]
