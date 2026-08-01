---
id: google-docs-discovery-debugger
title: Google Docs Discovery Debugger
summary: Reproduce Docs named-service title lookup and distinguish account, title, and Drive MIME mismatches.
tags: [sdk, examples, named-services, google-docs, diagnostics]
keywords: [Google Docs, Google Drive, title search, MIME type, MCP, connected account]
see_also:
  - repo:kdcube-ai-app/app/ai-app/docs/recipes/connections/integrations/google-service-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/integrations/google/google-README.md
---

# Google Docs Discovery Debugger

[`debug_search.py`](debug_search.py) is a read-only reproduction for a document
that KDCube's `docs` namespace cannot find by title. It separates two paths:

1. **Governed path:** calls `named_services_schema` and
   `named_services_search` over the same MCP endpoint an external agent uses.
2. **Provider path:** calls the SDK's
   `execute_google_docs_operation(operation="search")`, then asks Drive for the
   exact title without a MIME filter and with the native-Google-Docs filter.

This distinction matters because Drive can open a Microsoft Word file in the
Google Docs UI while retaining its DOCX MIME type and filename. The `docs`
namespace treats that file as an import source: a query for `26_006` can exactly
match `26_006.docx`, and `copy` converts it into a new editable native Google Doc
without changing the source file.

[`sdk_search.py`](sdk_search.py) is the focused SDK-only version. It calls
`execute_google_docs_operation(operation="search")`, selects the exact match
when one exists, and calls the SDK's `get_source` operation for its metadata. It
does not construct an HTTP request or call the Drive API directly. Use it to
verify the behavior an SDK consumer receives; use `debug_search.py` only when
that behavior must be compared with raw provider responses.

Both samples stop at Drive discovery. Search results do not contain a native
document's tab structure. After selecting or converting a source, call the
SDK `get` operation on the native document ref before editing. That read returns
`tab_count` and the complete tab inventory. A multi-tab mutation must then name
one `tab_id`, selected `tab_ids`, or explicit `all_tabs=true`, according to the
operation.

## IntelliJ Run Configuration

- Script: `debug_search.py`
- Interpreter: `app/venvs/ai-app/chat-processor/bin/python`
- Working directory: `app/ai-app/src/kdcube-ai-app`
- Add `app/ai-app/src/kdcube-ai-app` as a source root.

Set the Script field to `sdk_search.py` for the SDK-only run. Both scripts read
the same local `.env` and IntelliJ environment values.

Environment values can be entered in the Run Configuration or placed in a
local `.env` beside the script. Git ignores that file. `example.env` lists the
accepted names without carrying credentials.

For the governed path set:

```text
KDCUBE_DOCS_QUERY=26_006
KDCUBE_DOCS_ACCOUNT_ID=google_...
KDCUBE_MCP_URL=https://.../kdcube-services@1-0/public/mcp/named_services
KDCUBE_MCP_BEARER=<delegated bearer for that door>
```

For the provider comparison, additionally set a short-lived token for the same
Google account:

```text
GOOGLE_ACCESS_TOKEN=<short-lived test token>
```

### Obtain A Short-Lived Read-Only Token

Use Google's official [OAuth 2.0 Playground](https://developers.google.com/oauthplayground/):

1. In **Step 1**, enter this scope under **Input your own scopes**:
   `https://www.googleapis.com/auth/drive.readonly`.
2. Select **Authorize APIs** and sign in as the Google account being diagnosed.
3. Confirm the account on Google's consent screen before approving read-only
   Drive access.
4. In **Step 2**, select **Exchange authorization code for tokens**.
5. Put only the resulting **Access token** in the IntelliJ Run Configuration as
   `GOOGLE_ACCESS_TOKEN`. Do not use or retain the refresh token.

The Playground shows the access token's expiry. This sample needs only Drive
read access because it compares title and MIME metadata; it does not read or
modify document content. When using your own OAuth client in the Playground,
Google requires `https://developers.google.com/oauthplayground` as an authorized
redirect URI for that client.

The connected-account token remains server-side in normal KDCube operation.
The optional direct stage deliberately uses a separate test token so this
standalone process can isolate Google Drive behavior without weakening that
boundary. The script does not print either token.

Run as a module from the source root if preferred:

```bash
app/venvs/ai-app/chat-processor/bin/python -m \
  kdcube_ai_app.apps.chat.sdk.examples.named_services.productivity.docs.debug_search

app/venvs/ai-app/chat-processor/bin/python -m \
  kdcube_ai_app.apps.chat.sdk.examples.named_services.productivity.docs.sdk_search
```

## Reading The Result

| Observation | Meaning |
| --- | --- |
| MCP and SDK both return the item with `exact_title_match: true` | Named-service logical-title discovery works. |
| The result has `object_kind: docs.import_source` | Copy it first; the returned `docs.document` ref is the editable native copy. |
| Raw prefix finds `26_006.docx`, while raw exact `26_006` and native Docs do not | Drive stores the extension and the file is an import source; this is expected. |
| Prefix finds another name without `exact_title_match` | The actual or logical title differs from the requested title. |
| Drive identity is the wrong email | KDCube or the test token selected another Google account. |
| Direct SDK finds it but MCP does not | Investigate connected-account selection, grants, or the staged runtime version. |
| No stage finds it | The file is not visible to that account, is trashed, or has another title. |

`incompleteSearch: false` means Drive completed that query against its selected
corpus. It does not mean that a separately paginated recent-document listing
has enumerated every document.
