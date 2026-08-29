---
id: repo:kdcube-ai-app/app/ai-app/docs/arch/delegated-authority-and-admission-README.md
title: "Delegated Authority And Admission"
summary: "Points from KDCube's host architecture to Prokura's canonical delegated-authority and admission contract."
status: current
tags: ["arch", "security", "admission", "prokura", "delegated-access"]
keywords: ["delegated authority", "managed surface guard", "delegated access card", "Prokura"]
updated_at: 2026-08-29
see_also:
  - https://github.com/elenaviter/app-ecosystem/blob/main/docs/prokura/package/delegated-authority-and-admission.md
  - repo:kdcube-ai-app/app/ai-app/docs/arch/security-and-trust-model-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/solutions/connections/authenticated-mcp/authenticated-mcp-README.md
---

# Delegated Authority And Admission

Prokura owns the canonical delegated-card authority and per-call admission
contract. Read
[Delegated Authority And Admission](https://github.com/elenaviter/app-ecosystem/blob/main/docs/prokura/package/delegated-authority-and-admission.md)
for card/catalog resolution, managed-surface guards, connected-account claims,
and direct or relayed named-service invocation.

KDCube remains the host runtime. Its authenticated MCP, Data Bus, named-service,
request-session, and application-surface documents describe how KDCube supplies
the transport and runtime adapters around that authority.
