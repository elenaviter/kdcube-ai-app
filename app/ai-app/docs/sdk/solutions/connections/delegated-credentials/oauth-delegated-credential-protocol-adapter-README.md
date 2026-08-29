---
id: repo:kdcube-ai-app/app/ai-app/docs/sdk/solutions/connections/delegated-credentials/oauth-delegated-credential-protocol-adapter-README.md
title: "OAuth Delegated Credential Protocol Adapter"
summary: "Points from KDCube's OAuth host routes to Prokura's canonical delegated-credential protocol contract."
status: active
tags: ["sdk", "connections", "prokura", "oauth", "delegated-credentials"]
keywords: ["OAuth2 authorization server", "PKCE", "CIMD", "dynamic client registration", "Prokura"]
updated_at: 2026-08-29
see_also:
  - https://github.com/elenaviter/app-ecosystem/blob/main/docs/prokura/package/oauth-delegated-credential-protocol.md
  - repo:kdcube-ai-app/app/ai-app/docs/service/auth/auth-README.md
---

# OAuth Delegated Credential Protocol Adapter

Prokura owns the canonical client resolution, PKCE, authorization, consent,
credential issuance, card lookup, and revocation contract. Read
[OAuth Delegated Credential Protocol](https://github.com/elenaviter/app-ecosystem/blob/main/docs/prokura/package/oauth-delegated-credential-protocol.md).

KDCube remains the first protocol host. Its authentication, descriptor, and MCP
recipes describe how the Prokura state machine is mounted on KDCube public
operations and guarded application surfaces.
