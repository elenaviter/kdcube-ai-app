---
id: repo:kdcube-ai-app/app/ai-app/docs/sdk/solutions/connections/delegated-cards/delegated-cards-README.md
title: "Delegated Access Cards: Storage, Rendering, And Enforcement"
summary: "Points from KDCube's Connection Hub integration to Prokura's canonical delegated-card lifecycle."
status: active
tags: ["sdk", "connections", "prokura", "delegated-access", "cards"]
keywords: ["Delegated by KDCube", "resource grants", "catalog drift", "Prokura"]
updated_at: 2026-08-29
see_also:
  - https://github.com/elenaviter/app-ecosystem/blob/main/docs/prokura/package/delegated-cards.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/solutions/connections/connection-hub-solution-README.md
---

# Delegated Access Cards

Prokura owns the canonical card, catalog, persistence, rendering, mutation,
revocation, drift, and per-call enforcement lifecycle. Read
[Delegated Access Cards](https://github.com/elenaviter/app-ecosystem/blob/main/docs/prokura/package/delegated-cards.md).

KDCube's Connection Hub app is the first frontend over this contract. KDCube
configuration and integration documents continue to own the host descriptor,
transport, consent-event, and application-registration details.
