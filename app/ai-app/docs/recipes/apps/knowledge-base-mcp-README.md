---
id: repo:kdcube-ai-app/app/ai-app/docs/recipes/apps/knowledge-base-mcp-README.md
title: "Connect The KDCube Knowledge Base To Claude (MCP)"
summary: "One-minute procedure to add a KDCube app's public knowledge MCP as a custom connector in Claude — using the kdcube.tech/mcp vanity alias, the anonymous streamable-http endpoint, a verify step, and how to point the same recipe at your own runtime and bundle."
status: active
tags: ["recipes", "kdcube-for-agents", "mcp", "knowledge", "claude", "connector", "public", "demo"]
updated_at: 2026-07-12
see_also:
  - repo:kdcube-ai-app/app/ai-app/docs/recipes/apps/named-services-mcp-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/recipes/resource_sharing/publish-discoverable-content-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/tools/mcp-README.md
---
# Connect The KDCube Knowledge Base To Claude (MCP)

A KDCube app can expose its knowledge base as a **public MCP endpoint** —
anonymous, no login — so any MCP client can search it and read the cited
sources. This recipe adds it to Claude as a custom connector in about a
minute, ready for a demo.

## The endpoint

Two equivalent URLs point at the same server:

| Form | URL |
| --- | --- |
| **Vanity alias** (use this for a demo) | `https://kdcube.tech/mcp/docs` — also `https://kdcube.tech/mcp` |
| Canonical | `https://dev.kdcube.tech/api/integrations/bundles/demo/demo-march/knowledge@1-0/public/mcp/knowledge` |

The alias is a `308` permanent redirect to the canonical URL (served by the
site's CloudFront function, so it also works on every preview host, e.g.
`https://pr123.kdcube.tech/mcp/docs`). The endpoint is **anonymous** (the
`/public/` path segment) and speaks **streamable HTTP** MCP — exactly what a
remote custom connector needs. Nothing to configure on the KDCube side; it is
already live.

## Add it to Claude

In Claude (Desktop or claude.ai):

1. **Settings → Connectors** → **Add custom connector**.
2. Name it, e.g. `KDCube Knowledge`.
3. Paste the URL: **`https://kdcube.tech/mcp/docs`**.
4. **Add** / **Connect**. Because the endpoint is public, it connects
   straight away — no OAuth, no key.

The connector appears with its tools enabled. In a chat, make sure the
`KDCube Knowledge` connector is toggled on for the conversation.

> Custom (remote) connectors are a paid-plan feature. If your Claude Desktop
> only offers a local JSON config, add the same URL as a remote MCP server
> there instead; the URL is the only thing that matters.

## Verify

Ask Claude, with the connector on:

> Using the KDCube Knowledge connector, search for how KDCube handles
> subagents, and cite the sources.

Claude calls the connector's search tool, gets back matching documents, reads
the cited sources, and answers with links. If it returns KDCube docs, the
connection works. (The connector's own tool list — search plus source
reading — is visible in the connector's detail panel once added.)

## Point it at your own runtime and bundle

The canonical URL is a fixed shape — swap the host, tenant, project, and
bundle for your own:

```
https://<runtime-host>/api/integrations/bundles/<tenant>/<project>/<knowledge-bundle>/public/mcp/knowledge
```

For example, a locally running knowledge bundle under the
`demo-tenant / demo-project` runtime is:

```
https://<your-runtime-host>/api/integrations/bundles/demo-tenant/demo-project/knowledge@1-0/public/mcp/knowledge
```

To give your own knowledge endpoint a clean vanity alias like
`yoursite.tech/mcp`, add a `308` redirect for the `/mcp*` paths to the
canonical URL. The reference is the site's CloudFront function
(`website/scripts/preview-infra/cloudfront-function.js`): a small
`VANITY_REDIRECTS` map that matches `/mcp`, `/mcp/docs`, etc. before any other
rewriting and returns a `308` to the runtime MCP URL.

## Notes

- The endpoint is read-only knowledge retrieval — search and source reading;
  it does not write or act. For agent-actionable KDCube surfaces (conversation,
  mail, slack, tasks, memory) over MCP, see
  [Make A Named Service Agent-Friendly (MCP)](named-services-mcp-README.md).
- The public knowledge surface is the same discoverability content the app
  publishes to the web; see
  [Publish Discoverable Content From An App](../resource_sharing/publish-discoverable-content-README.md).
- If a client refuses to follow the `308`, point it at the canonical URL
  directly — same server.
