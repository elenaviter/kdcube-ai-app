---
id: repo:kdcube-ai-app/app/ai-app/docs/arch/delegated-authority-and-admission-README.md
title: "Delegated Authority And Admission"
summary: "Points from KDCube's host architecture to Connection Hub's canonical delegated-authority and admission contract."
status: current
tags: ["arch", "security", "admission", "connection-hub", "delegated-access"]
keywords: ["delegated authority", "managed surface guard", "delegated access card", "Connection Hub"]
updated_at: 2026-08-30
see_also:
  - https://github.com/elenaviter/app-ecosystem/blob/main/docs/connection-hub/package/delegated-authority-and-admission.md
  - https://github.com/elenaviter/app-ecosystem/blob/main/docs/connection-hub/connection-hub-architecture.md
  - repo:kdcube-ai-app/app/ai-app/docs/recipes/connections/protect-external-service-with-connection-hub-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/arch/security-and-trust-model-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/solutions/connections/authenticated-mcp/authenticated-mcp-README.md
---

# Delegated Authority And Admission

Connection Hub owns the canonical delegated-card authority and per-call admission
contract. Read
[Delegated Authority And Admission](https://github.com/elenaviter/app-ecosystem/blob/main/docs/connection-hub/package/delegated-authority-and-admission.md)
for card/catalog resolution, managed-surface guards, connected-account claims,
direct protected-service admission, and direct or relayed named-service
invocation.

KDCube remains the host runtime. Its authenticated MCP, Data Bus, named-service,
request-session, and application-surface documents describe how KDCube supplies
the transport and runtime adapters around that authority.
