---
id: kdcube-services@1-0/docs/journal/2026-07-27-named-service-complete-delivery-and-exact-actions.md
title: "Complete Named-Service Delivery And Exact Actions"
summary: "Aligns Mail, Slack, and Sheets around pagination, complete external/harness retrieval, and independently governed object actions."
status: active
tags: ["kdcube-services", "named-services", "mail", "slack", "sheets", "mcp", "streaming", "delegated-access"]
---

# Complete Named-Service Delivery And Exact Actions

Named-service tool results stay bounded for model use, while authorized data
remains completely retrievable:

- Mail search accepts/returns provider cursors. A Mail message can be streamed
  as `kdcube.mail.message.snapshot.v1`; attachments retain separate refs.
- Slack list/search accepts/returns cursors, exact message refs are readable,
  and provider file bytes stream without a KDCube model-context size cap.
- Sheets retains selected-range reads and its complete
  `kdcube.sheets.snapshot.v1` stream.

Turnless MCP clients receive signed KDCube URLs. Those routes verify the bound
identity and exact ref, resolve the current connected-account credential,
enforce current consent, then fetch and stream current provider data. They are
live provider proxies, not pre-hosted immutable artifacts. The delegated
caller grant is checked before minting; the signed URL is then the short-lived
GET capability. Connected-account revocation is effective on its next use,
while caller-grant revocation prevents a replacement URL. Harness clients use
`object.get(response_mode=stream)` through `react.pull`, which writes a stable
copy into the current turn workspace.

The generic bridge now forwards provider cursors and authorizes bounded actions
as `object.action.<action>` while dispatching them to provider
`object.action`. Connection Hub descriptors declare those exact variants so
read, send/post, upload, download, and provider-specific actions can be granted
independently. Older parent-only `object.action` descriptors remain compatible;
once exact variants are declared, unknown actions fail closed.
