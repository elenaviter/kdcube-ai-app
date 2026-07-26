# Two-Level Consent: a plain MCP client reading a KDCube governed door

A runnable, reference MCP client that shows what a KDCube governed MCP door
feels like from the outside: **every policy denial arrives as a structured,
actionable consent envelope - never a bare HTTP 403.** The client names which of
the two authorization levels denied the call and prints the exact human fix.

- [`two_level.py`](two_level.py) - the reusable classifier (the reference
  logic). Dependency-light, importable, testable. **This is the piece other
  clients copy.**
- [`client.py`](client.py) - a thin runnable driver over the official `mcp`
  SDK that connects, lists tools, calls one, and narrates the outcome.
- [`test_two_level.py`](test_two_level.py) - the classifier verified against
  the real envelope shapes as literal dicts (no live door needed).

## What it proves

A single tool call crosses **two sequential authorization levels**, and a
KDCube door answers each denial with a structured envelope returned AS THE TOOL
RESULT:

- **Level 1 - the caller's own grant.** The bearer's delegated credential lacks
  a grant the operation needs. The door names the exact missing grants and a
  Connection Hub deep link where the user approves them for this caller. A
  level-1 miss is never fixed by replaying the call.

- **Level 2 - the connected account plus the per-account binding.** The caller
  holds the MCP grant, but the user-to-provider side cannot satisfy the call. A
  `reason` says precisely why, and `retry_hint` says whether replaying the same
  call works after the user acts:

  | reason | the user's fix |
  | --- | --- |
  | `connect_required` | connect an account on the backing provider at the hub URL |
  | `claim_upgrade_required` | approve the listed claims for an existing account |
  | `reconnect_required` | reconnect an account whose stored credential stopped working |
  | `account_required` | resend the **same** call with `account_id` from `candidates` |
  | `agent_grant_required` | tick the claim for an account on this caller's grant card |

The door's contract is **relay, do not retry blindly**: only `account_required`
resolves by resending (with an `account_id`); every other reason needs a human
action at the Connection Hub URL first. The classifier surfaces this as
`ConsentOutcome.resend_with_account_id`, and the demo never loops or hammers the
door.

Each denial is distinct and self-describing, so a consuming client always knows
whose action clears it - the caller's own grant (level 1) or the user's account
and per-account binding (level 2).

## How to run

From `app/ai-app/src/kdcube-ai-app`:

```bash
python3 -m kdcube_ai_app.apps.chat.sdk.examples.mcp.two_level_client.client \
    --url    "$KDCUBE_MCP_URL" \
    --bearer "$KDCUBE_MCP_BEARER" \
    --tool   productivity_slack_search \
    --query  "quarterly planning"
```

| flag | env fallback | meaning |
| --- | --- | --- |
| `--url` | `KDCUBE_MCP_URL` | the governed door URL |
| `--bearer` | `KDCUBE_MCP_BEARER` | a delegated bearer token for that door |
| `--tool` | - | tool id (default `productivity_slack_search`) |
| `--query` | - | the `query` param for search tools |
| `--account-id` | - | resend a level-2 `account_required` call with this account |

A door URL looks like:

```
<base>/api/integrations/bundles/<tenant>/<project>/kdcube-services@1-0/public/mcp/productivity
```

(the `productivity` pure-MCP door; `.../mcp/named_services` is the generic
named-services door).

The client prints the tool roster, then a narration block:

- **SUCCESS** - "level 1 passed (grant) - level 2 passed (account + binding)"
  plus a short result preview.
- **LEVEL 1** - the code, the missing grants, and the one-line fix (and it does
  not retry).
- **LEVEL 2** - the reason, `retry_hint`, and the fix; for `account_required` it
  lists the candidate accounts and tells you to resend with an `account_id`.

Missing `--url`/`--bearer` exits with a helpful message (exit code 2); a missing
`mcp` SDK or a transport/auth error is reported, not retried.

### A live capture

This is the real tool result returned by the `productivity` door for a bearer
whose caller holds the grant and whose account is bound (both levels pass) -
note `error` is `null`, not absent, and the envelope is `{ok, error, ret}`:

```json
{"ok": true, "error": null,
 "ret": {"messages": [], "count": 0, "account_id": "slack_411cbb1ecf4ef353"}}
```

`classify_tool_result` reads it as `ok=True, level=0`, which the client narrates:

```
========================================================================
SUCCESS  'productivity_slack_search'
  level 1 passed (the caller's own grant)
  level 2 passed (the connected account + per-account binding)
  result: {"messages": [], "count": 0, "account_id": "slack_411cbb1ecf4ef353"}
========================================================================
```

The level-1 and level-2 *denial* envelopes are exercised as literal fixtures in
`test_two_level.py` (a fully-bound bearer never sees them live); each fixture is
copied from the door's own denial paths.

### Obtaining a delegated bearer

The bearer is a KDCube-issued **delegated credential** for the door. Two ways to
get one:

- **Connect via OAuth** (dynamic client registration) - an MCP-speaking app
  probes the door, registers, and completes the authorize flow; KDCube issues
  the delegated credential.
- **Mint a bounded automation token** in Connection Hub (Delegated by KDCube ->
  Create automation access), narrowed to selected resources, grants, and
  named-service operations, with a TTL.

Both journeys, the door configuration, and the full error vocabulary are in the
canonical reference:
[Authenticated MCP: The Full Configuration Chain](../../../../../../../../../docs/sdk/solutions/connections/authenticated-mcp/authenticated-mcp-README.md).

## How it pairs with the productivity door (the server half)

This example is the **client** side. The **server** side - how you BUILD a
governed door that emits exactly these envelopes - is the `productivity` MCP
surface of the `kdcube-services@1-0` example bundle:
[`surfaces/mcp/productivity.py`](../../bundles/kdcube-services@1-0/surfaces/mcp/productivity.py),
a pure-MCP door wrapping Slack search plus mail search/read, each tool declaring
the connected-account claims it needs and enforcing them with one call.

One page is how you build a governed MCP server; this is how a client consumes
it. Read them together with the configuration chain above, which owns the full
concept and links both.

## Verify the classifier without a door

```bash
python3 -m pytest kdcube_ai_app/apps/chat/sdk/examples/mcp/two_level_client/ -q
```

The tests feed `classify_tool_result` the real envelope shapes (level 1, level-2
`connect_required`, level-2 `account_required`, `agent_grant_required`, success,
and a malformed payload) as literal dicts and assert the level, reason, and fix.

## Cite this from a recipe or article

- The **classifier** ([`two_level.py`](two_level.py)) is the canonical
  reference for "given a KDCube tool result, which of the two levels denied me
  and how does the user fix it." Cite `classify_tool_result` /
  `ConsentOutcome` when an article needs the client-side reading of the
  two-level model.
- The **narration** in [`client.py`](client.py) is the copy-ready
  demonstration that a plain MCP client experiences every denial as an
  actionable two-level lifecycle rather than a bare 403.
- Always pair a citation with the server half
  ([the configuration chain](../../../../../../../../../docs/sdk/solutions/connections/authenticated-mcp/authenticated-mcp-README.md)),
  which owns the concept in depth.
