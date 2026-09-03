---
id: repo:kdcube-ai-app/app/ai-app/docs/arch/delegated-authority-and-admission-README.md
title: "Delegated Authority And Admission"
summary: "Points from KDCube's host architecture to Connection Hub's delegated authority, invocation policy, external MCP proxy, and direct-admission contract."
status: current
tags: ["arch", "security", "admission", "connection-hub", "delegated-access"]
keywords: ["delegated authority", "managed surface guard", "delegated access card", "invocation policy", "external MCP proxy", "Connection Hub"]
updated_at: 2026-09-02
see_also:
  - https://github.com/elenaviter/app-ecosystem/blob/main/docs/connection-hub/package/delegated-authority-and-admission.md
  - https://github.com/elenaviter/app-ecosystem/blob/main/docs/connection-hub/connection-hub-architecture.md
  - repo:kdcube-ai-app/app/ai-app/docs/recipes/connections/protect-external-service-with-connection-hub-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/service/cicd/delegated-management-service-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/arch/security-and-trust-model-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/solutions/connections/authenticated-mcp/authenticated-mcp-README.md
---

# Delegated Authority And Admission

Connection Hub owns the canonical delegated-card authority and per-call admission
contract. Read
[Delegated Authority And Admission](https://github.com/elenaviter/app-ecosystem/blob/main/docs/connection-hub/package/delegated-authority-and-admission.md)
for card/catalog resolution, once-or-always invocation policy, user-owned
external MCP proxying, managed-surface guards, connected-account claims,
direct protected-service admission, and direct or relayed named-service
invocation.

KDCube remains the host runtime. Its authenticated MCP, Data Bus, named-service,
request-session, and application-surface documents describe how KDCube supplies
the transport and runtime adapters around that authority.

The [Delegated KDCube Management Service](../service/cicd/delegated-management-service-README.md)
is the platform reference for a state-changing protected service. It combines
live Connection Hub admission, an exact browser-approved request permit, and a
separate effect ledger before reloading one declared application.
