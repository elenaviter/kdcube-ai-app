---
id: kdcube-services@1-0/docs/journal/2026-08-01-docs-explicit-tab-selection.md
title: "Google Docs Explicit Tab Selection"
summary: "Native document reads now expose the complete tab inventory, and multi-tab mutations require an explicit target before Google receives a write."
status: active
tags: ["kdcube-services", "productivity", "google", "docs", "tabs", "named-services", "mcp"]
keywords: ["Google Docs tabs", "tab id", "multi-tab document", "batchUpdate", "tab selection"]
see_also:
  - repo:kdcube-ai-app/app/ai-app/docs/recipes/connections/integrations/google-service-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/integrations/google/google-README.md
  - repo:kdcube-ai-app/app/ai-app/src/kdcube-ai-app/kdcube_ai_app/apps/chat/sdk/examples/bundles/kdcube-services@1-0/interface/README.md
---

# Google Docs Explicit Tab Selection

## Symptom

Google Docs can contain several tabs, including nested tabs. Search only returns
Drive metadata, so a search result cannot tell an agent which tabs exist. Once a
native document is opened, the Docs API also has different defaults for an
omitted tab scope: many location-based edits affect the first tab, while an
unscoped text replacement can affect every tab.

Allowing those defaults through the SDK would make an ambiguous agent action a
real provider write. The user could ask to update one tab and receive a change
in another, or a replacement could spread farther than intended.

## Contract

A native document read requests complete tab content and returns the extracted
text from every tab. It also returns `tab_count` and a flattened tab inventory.
Each tab record includes its stable `tab_id`, title, order, parent, nesting
level, and body end index.

Single-tab documents remain concise: the SDK can infer the only target. A
multi-tab mutation must carry its scope:

- insert, append, styling, page-break, and image operations use one `tab_id`;
- replacement uses selected `tab_ids` or explicit `all_tabs=true`;
- flexible batches use one `tab_id`; an all-tabs batch is allowed only when
  every request is `replaceAllText`.

The SDK validates the selected ids against a fresh document read and stamps the
provider request with `tabId` or `tabsCriteria`. Unknown and conflicting scopes
are rejected before the mutation request.

The provider-neutral `docs` named service adds a human-addressable layer over
that exact SDK contract. A caller can supply one `tab_selector` by title,
literal title fragment, 1-based position, or full hierarchy; replacement also
accepts several `tab_selectors`. The provider reads current tab metadata,
resolves exactly one candidate per selector, and then calls the same typed SDK
operation with `tab_id` or `tab_ids`. Existing clients may continue to send the
exact ids directly.

## Agent-Visible Recovery

An omitted multi-tab scope returns `docs_tab_selection_required`. The error
includes `tab_count`, the available tab records, and a direct next action. The
named-service bridge preserves those details, and the typed productivity MCP
tools describe the same parameters in `tools/list`.

The agent therefore sees what happened. It can select a tab whose title matches
the user's request, or ask the user when no intended tab is clear. The harness
does not silently pick the first tab or expand the write to every tab.

Named-service ambiguity is also explicit. For example, a `Notes` title fragment
can match distinct tabs named `Internal Notes` and `Invoice Notes`. The provider
returns `docs_tab_selector_ambiguous` with bounded position and hierarchy
candidates; no mutation follows. A successful natural selection records its
resolved tab in `selector_resolution`.

## Regression Coverage

Focused tests cover complete tab inventory, nested-tab order, selected-tab end
indices, every typed one-tab mutation, selected and all-tab replacement,
unknown tabs, conflicting scopes, flexible batch stamping, named-service error
propagation, and typed MCP parameter descriptions. They also assert that an
ambiguous mutation performs only the preflight read and sends no provider
write. Named-service tests additionally cover title fragments, hierarchy,
multiple selectors, overlapping-fragment ambiguity, and exact-id compatibility.
