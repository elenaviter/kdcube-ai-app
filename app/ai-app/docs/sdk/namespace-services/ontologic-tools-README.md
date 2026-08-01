---
id: repo:kdcube-ai-app/app/ai-app/docs/sdk/namespace-services/ontologic-tools-README.md
title: "Namespace Services: Ontologic Tools"
summary: "The named-service ontologic tools as one coherent model-facing surface for operating any realm by satisfying schemas."
status: design
tags: ["sdk", "namespace-services", "ontologic-tools", "agents", "schema", "affordance"]
updated_at: 2026-08-01
keywords:
  [
    "ontologic tools",
    "schema satisfaction",
    "affordance",
    "object schema",
    "provider about",
    "domain selector",
    "provider-bounded search",
    "provider translation",
    "upsert_object",
    "object_action",
    "update strategy",
    "realm",
  ]
see_also:
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/namespace-services/README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/namespace-services/providers-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/namespace-services/clients-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/tools/named-services-tools-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/namespace-services/react-object-materialization-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/solutions/ecosystem-component/ecosystem-component-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/namespace-services/object-ref-presentation-and-actions-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/recipes/connections/integrations/google-service-README.md
---
# Namespace Services: Ontologic Tools

The named-service **ontologic tools** are the single model-facing interface for
operating any realm. A realm owner explicitly publishes a provider; an
integrating app configures which operations its agents may call; the agent then
works one realm through the same small, generic tool set regardless of domain.
Publication lifecycle details belong to
[Discovery Registry](discovery-README.md), not this tool-model page.

This page is the conceptual surface: how the tools compose and why "operate a
realm" reduces to "satisfy a schema". It does not restate per-operation detail
or consumer wiring:

- per-operation provider contract (request/response fields, search scopes,
  streamed reads, block production): [Providers](providers-README.md);
- how operations become model-callable tools, `surfaces.as_consumer`
  allow-lists, `tool_traits`, catalog rendering, and the config→tool mapping
  table: [Named Service Tools](../tools/named-services-tools-README.md);
- the runtime context boundary (who owns what across the call):
  [Clients](clients-README.md);
- the realm participation contract:
  [Ecosystem Component](../solutions/ecosystem-component/ecosystem-component-README.md).

## The Ontologic Model

A realm is **schema-bearing objects**. The agent does not learn a bespoke API
per realm. It operates by **satisfying schemas**:

```text
to act in a realm = to create or modify an object that satisfies its schema
```

There is no generic "call" and no free-floating "use case". A use case ("file a
bug", "attach a report", "open the editor") is always one of: create an object,
modify an object, or run a schema-declared action on an object. The schema is
the contract; satisfying it is the operation.

Satisfaction is **recursive**. A schema may require a typed link to another
kind. To provide that link the agent reads the linked kind's schema, resolves or
creates an object of that kind, and supplies its `object_ref`. Required-input
resolution therefore walks the kind graph by reading schemas, not by guessing
URIs.

The **affordance** of a realm — everything the agent can do — is exactly:

```text
affordance =
    the catalog of kinds (what objects exist)
  + each schema's typed links (how kinds connect)
  + each schema's `tools` block (which tool + args satisfy each op per kind)
```

Nothing else is callable. If it is not a kind, a link, or a declared op on a
kind, it is not part of the realm's surface. This keeps the model's world small:
it only ever inspects schemas and satisfies them.

### A Schema Is the Realm's Language

An object schema is not a copy of the provider's endpoint catalog. It is the
small language an agent uses to understand and operate the realm:

- **object kinds** name things a user recognizes;
- **relations and selectors** explain how to find the intended thing;
- **actions** say what can happen to it;
- **fields and claims** define the exact bounded request;
- **results and errors** make the outcome and the next valid step visible.

Here, meaning is **domain meaning**, not a promise of semantic search. A realm
can expose semantic search when its provider supplies it. Otherwise it exposes
only the provider-backed search and explicit predicates it can evaluate over a
bounded provider response. KDCube does not build an index over the provider's
object space.

Provider endpoint names, raw request bodies, and call ordering stay behind the
provider. Opaque provider ids can be returned as handles, but the user should
not have to supply one. The adapter resolves a request such as "the second tab"
from document structure. It can resolve "my comment containing payment terms"
from provider-returned comment fields. A broader meaning-based match is
available only when the provider declares semantic search.

A complete document realm can therefore present this ontology:

```text
docs
  documents
  tabs
  comment threads
  replies
  exports

operations
  find / read / copy / export a document
  add / rename / delete / read a tab
  edit a selected tab
  list / create / reply / edit / delete comments
  resolve / reopen a comment thread
```

That does not imply one generic provider call per line. A request such as
"reply to my unresolved comment about payment terms" may require the adapter
to page through the selected document's comments, resolve the matching thread,
perform the provider mutation, and return the updated thread. A separate
request such as "append this to the second tab" resolves that tab from the
document structure. The agent sees one bounded domain action; the provider
owns the call sequence.

```text
user language
  "my unresolved comment about payment terms"
        |
        v
schema-declared selector + action
  docs.comment_thread + author=me + resolved=false + text_contains + reply
        |
        v
provider adapter
  resolve provider ids -> perform supported calls -> normalize result
        |
        v
provider API
```

The `selector` shape in this example is conceptual. A provider can declare
supported selection through search filters, an action payload, or another
schema-owned field. Named services do not impose one universal selector JSON
envelope; they require the provider to describe the supported shape and its
ambiguity behavior.

The schema is also a capability boundary. It advertises only behavior the
configured provider can perform. A preview-only or unavailable provider
feature is omitted or marked unavailable, and an attempted unsupported action
returns a precise capability error. The adapter must not silently replace a
more specific operation with a weaker one.

**Google Docs status:** the current named-service adapter resolves tabs by
title, literal title fragment, 1-based position, or hierarchy. It resolves one
document-level Drive comment by literal text, quoted text, author (`me` is
supported), resolved state, or position. Exact provider ids remain available
for callers that already hold them. Ambiguous selectors return bounded
candidates, and a tab-scoped comment request returns
`tab_anchored_comments_unavailable`. Native tab anchors remain a preview-only
provider capability. Provider-side translation and capability reporting are
specified in
[Providers](providers-README.md#domain-operations-and-provider-translation).

## The Tool Surface

The shipped ontologic tools are nine generic operations. They are domain-free;
the realm fills in meaning through `about` and `object.schema`.

| Tool | Role in the surface |
| --- | --- |
| `provider_about` | The realm catalog: kinds, scopes, action vocabulary, and a query playbook. Read first when the rendered scope hints are not enough. |
| `object_schema` | The exact contract for one kind/scope: fields, typed links, search filters, and the `tools` block that names the tool + args per op. |
| `list_objects` | Browse a collection with pagination. |
| `search_objects` | Find objects by query within a provider-declared scope. |
| `get_object` | Fetch one object by ref (live realm state). |
| `upsert_object` | Create or modify one object that satisfies a kind's schema. On update, a collection field accepts either a bare list (set/append per `update_strategy`) or a `{add, remove}` delta; objects use `patch`/`replace`; scalars are replace (set if provided, preserved if omitted). |
| `delete_object` | Destroy one object itself (the file/record everywhere it is used). Not a list-editing tool — to take an item off a list use that field's `{remove}` delta. |
| `object_action` | Run a schema-declared bounded action on an object (`preview`, `open`, `download`, or a provider-defined action). |
| `host_file` | Host a runtime file/ref into the realm and get back a realm-owned file ref to cite via `upsert_object`. |

These nine tools and the schema `tools` block are **shipped**. For their exact
parameters and the config that exposes each one, see
[Named Service Tools](../tools/named-services-tools-README.md). The
`object_ref` opacity rule and provider-owned actions are in
[Object Refs, Presentation, And Actions](object-ref-presentation-and-actions-README.md).

Provider results stay client-neutral. They describe the object, its size and
shape, available inline data, cursors/ranges, and any signed artifact URL. They
do not prescribe client-local tools. A ReAct consumer may inspect a local
artifact with ReAct tools; an external MCP client may download the signed URL
and use its own file facilities. Mail, Slack, and Sheets follow this boundary.

### How They Compose

The tools form one navigation path from "I know nothing about this realm" to "I
mutated it correctly". Some client surfaces also expose
`provider_capabilities` between `provider_about` and `object_schema`; it reports
the provider-declared operations available in that deployment without changing
the three-level intro/about/schema model:

```text
provider_about            discover the realm: kinds, scopes, action vocabulary
      |                   (catalog — what is here)
      v
object_schema(kind/scope) get the exact contract for the kind I need to satisfy
      |                   (by part — fields, typed links, the `tools` block)
      v
list_objects /            find or fetch the concrete objects I will read or link
search_objects /
get_object
      |
      v
upsert_object /           satisfy the schema: create/modify, delete, or run an
delete_object /           action. The schema told me which tool and which args.
object_action
```

The path is **catalog → schema (by part) → find → satisfy**. The agent reads
the realm down to the kind it must satisfy, never the whole realm at once.

The same returned `object_ref` can also become a local agent artifact when the
ReAct consumer declares a provider-backed pull policy for that namespace. The
tools still discover the object; `react.pull` then asks the owner for exact
bytes and returns a local `conv:fi:` ref for `react.read`. The `sheets`
namespace is a worked example: search returns a `sheets:` ref, and pull
materializes a JSON spreadsheet snapshot. External MCP clients do not receive
these `react.*` tools. A turnless get can return a signed URL for the complete
snapshot, including beside selected A1 values returned inline. See
[ReAct Object Materialization](react-object-materialization-README.md).

### The Schema's `tools` Block Is the Affordance Surface

The schema does not just describe fields — it literally names, per op, **which
tool to call and which args are required/optional**. This is the worked example
from the real tasks realm (`task.issue`):

```python
"tools": {
    "list":   {"tool": "named_services.list_objects",
               "required": {"namespace": "task"}},
    "search": {"tool": "named_services.search_objects",
               "required": {"namespace": "task:issue", "query": "<text>"}},
    "get":    {"tool": "named_services.get_object",
               "required": {"namespace": "task", "object_ref": "task:issue:<issue_id>"}},
    "create": {"tool": "named_services.upsert_object",
               "required": {"namespace": "task", "object_json": {"title": "<title>"}},
               "optional_object_json": ["description", "state", "assignee",
                                        "tags", "attrs", "attachments", "attachment_refs"]},
    "update": {"tool": "named_services.upsert_object",
               "required": {"namespace": "task", "object_ref": "task:issue:<issue_id>",
                            "object_json": {"title": "<new title>"}},
               "optional_object_json": ["description", "state", "assignee",
                                        "tags", "attrs", "attachments", "attachment_refs"]},
    "delete": {"tool": "named_services.delete_object",
               "required": {"namespace": "task", "object_ref": "task:issue:<issue_id>"}},
    "host_file": {"tool": "named_services.host_file",
                  "required": {"namespace": "task", "object_ref": "task:issue:<issue_id>",
                               "file_ref": "conv:fi:turn_<id>.files/<path> or a local runtime file path"},
                  "optional": ["filename", "mime", "description"],
                  "returns": "task:issue:attachment:<issue_id>/attachments/<attachment_id>/v<version>/<filename>"},
    "attach_hosted_refs": {"tool": "named_services.upsert_object",
                           "description": "Cite task-owned hosted attachment refs on the issue after host_file returns them.",
                           "required": {"namespace": "task", "object_ref": "task:issue:<issue_id>",
                                        "object_json": {"attachment_refs": [{
                                            "ref": "task:issue:attachment:<issue_id>/attachments/<attachment_id>/v<version>/<filename>",
                                            "filename": "<filename>", "mime": "<mime>"}]}}},
},
```

Reading this top to bottom is the affordance: the agent learns that creating an
issue is `upsert_object(namespace="task", object_json={title})`, that updating
adds `object_ref`, that a file becomes a citation in two steps (`host_file` then
`upsert_object` with the returned ref), and that `attachment_refs` is a typed
link to the `task.attachment` kind. No domain-specific tool was invented; the
generic tools plus the schema's `tools` block are the whole surface.

### Per-Field `update_strategy`

A collection (array) field in `upsert_object` accepts **either** a bare list
**or** a `{ "add": [...], "remove": [...] }` delta. `update_strategy` tells the
agent what the bare-list form does:

- **Arrays:** `append` (the bare list is added to the existing list) or
  `replace` (the bare list swaps the whole list). The `{add, remove}` delta is
  always an incremental edit regardless of strategy.
- **Objects:** `patch` (set the provided keys, keep the rest) or `replace`
  (overwrite the whole object).
- **Scalars** have no strategy — a provided value is set (replace), an omitted
  one is preserved.

Read the strategy before a bare-list update: it is the difference between
adding to a field and silently overwriting it. Omit a field to leave it
unchanged.

#### The `{add, remove}` delta

The delta form edits a list incrementally: removes are applied first, then adds.

- `add` appends the listed item(s).
- `remove` removes matching item(s) — by value for value-lists, by ref or
  `dedup_key` for ref-lists.

| Goal | What to send |
| --- | --- |
| Set / overwrite the whole list | a bare list (field is `replace`, or send the full intended value) |
| Add or remove some items | a `{add, remove}` delta |
| Replace one item | `add` it with a matching `dedup_key` (see below) |

#### `dedup_key` (per-parent supersede)

An `append` collection field may also declare a `dedup_key`. Adding an item
(bare or via `add`) whose key matches an existing one **within the same parent
object** **replaces/supersedes** it. So "replace one item" is just "add it again
with the same key" — there is no add-then-delete dance. Example: a task's
`attachments` keyed by `filename` — re-host the same `filename` and the new
version supersedes the old one on that issue.

#### Removing a collection item

Removal is **one way: the field's `{remove: [...]}` delta** — list the item(s)
to drop (by value for value-lists; by ref or `dedup_key` for ref-lists). For a
ref-list (e.g. `attachments`), `{remove}` **detaches** the item from this parent
object; a shared underlying object is preserved.

`delete_object` is **not** a list-editing tool. It **destroys the object
itself** — the underlying file/record everywhere it is used — and is a separate,
rarer operation, never the way to take an item off a list.

These bare-list / `{add, remove}` / `dedup_key` / removal defaults are also
part of the shared named-service agent guidance. The ReAct harness injects that
guidance directly; other agent integrations can expose the same provider
contract and tools without changing their semantics.

`update_strategy`, `dedup_key`, the `{add, remove}` delta, and these removal
rules are **shipped** on the task realm. Other surface improvements below remain
proposed.

## Teach It To Humans Too

The schema teaches the MODEL; the same spec now also teaches the USER: the
capability picker renders the realm's declared presentation (purpose,
works-with, human labels per operation/action, object-kind one-liners) as a
service card the user understands and narrows. A realm is well-modeled when
both readers pass: the agent can work it from `about`/`object_schema`, and a
user can read the card and control it. Declarations in
[Providers — The Presentation Layer](providers-README.md); rendering in
[Per-User Agent Capabilities](../solutions/user-settings/capabilities-README.md).

## Improvements (PROPOSED — not shipped)

These tighten the surface without adding domain tools. They are conventions and
extensions, not currently shipped behavior.

### `about` as a navigable top-level catalog + query playbook

`provider.about` is realm-filled, so the realm owner can make it the agent's
entry point. **Recommended convention** (content guidance, not a new op):

- a **top-level catalog**: namespaces, a shallow list of kinds/scopes, and the
  action vocabulary — the kinds/scopes it lists are exactly the selectors the
  agent passes to a focused `object_schema`;
- a **query playbook**: per common intent, a scope + filter template + example
  query, and a short "how to query this realm" note.

This makes `about → object_schema` a deterministic drill-down: the catalog names
the parts, and the agent fetches one part at a time. The same convention is
stated for the consumer side in
[Named Service Tools](../tools/named-services-tools-README.md) and
[Clients](clients-README.md); it lives in `about` content, not in generic code.

### `object_schema` projection selectors

For a large realm, reading a whole schema is wasteful. **Proposed extension**:
projection selectors on `object_schema` — `kind` / `scope` / a field subset /
traversal `depth` — so the agent fetches exactly the slice it needs (e.g. just
the create-required fields of one kind, or one level of typed links). These are
proposed params, **not current params**; today the agent fetches by kind/scope
and reads the returned contract.
