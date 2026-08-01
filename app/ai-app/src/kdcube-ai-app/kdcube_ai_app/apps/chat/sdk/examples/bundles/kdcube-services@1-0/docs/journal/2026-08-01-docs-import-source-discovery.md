---
id: kdcube-services@1-0/docs/journal/2026-08-01-docs-import-source-discovery.md
title: "Google Docs Import-Source Discovery"
summary: "Docs title search now discovers compatible Drive document files and converts a copied source into an editable native Google Doc."
status: active
tags: ["kdcube-services", "productivity", "google", "docs", "named-services", "docx", "conversion"]
keywords: ["Google Docs", "Google Drive", "DOCX", "logical title", "document conversion", "import source"]
see_also:
  - repo:kdcube-ai-app/app/ai-app/docs/recipes/connections/integrations/google-service-README.md
  - repo:kdcube-ai-app/app/ai-app/src/kdcube-ai-app/kdcube_ai_app/apps/chat/sdk/examples/bundles/kdcube-services@1-0/interface/README.md
---

# Google Docs Import-Source Discovery

## Symptom

A title search returned no Google Docs result even though the connected account
could open the requested document in the Google Docs UI. Raw Drive metadata
showed that the file was a DOCX whose provider filename included `.docx`; it was
not a native Google Doc.

The original search filtered Drive to
`application/vnd.google-apps.document`. It therefore excluded compatible files
that Google can open and convert. Pagination could not fix that mismatch because
the excluded file was outside every page of the filtered query.

## Contract

Docs discovery now returns native Google Docs plus compatible DOCX, ODT, and RTF
files. For an import source, the response preserves the provider filename and
also reports a logical title with the known extension removed. A query such as
`26_006` can therefore be an exact logical-title match for `26_006.docx`.

The named-service object kinds make the next step explicit:

```text
docs.import_source
  docs:<provider>:<account_id>:source:<file_id>
    |
    | object.action copy
    v
docs.document
  docs:<provider>:<account_id>:document:<document_id>
```

`copy` uses provider-native copy for native Google Docs. For an import source,
it downloads the approved source and uploads it with the native Google Docs MIME
type. The source stays unchanged. The returned native document ref supports the
existing read, replacement, comment, and export operations.

Import-source reads return metadata and conversion guidance rather than
pretending that the Docs API can read or edit the source directly. Other
document mutations reject a source ref and tell the caller to copy it first.

## Authority And Failure Semantics

Discovery and source metadata use `docs:read`. Copy/conversion resolves both
`docs:read` and `docs:write` on the connected account, while the caller still
needs the exact copy-operation grant. The Google credential remains in the
trusted service.

Copy remains a non-idempotent provider operation. If the conversion upload has
an unknown outcome, the client searches for the target title before retrying.
The conversion input is capped by the existing 10 MiB document import limit.

## Regression Coverage

Focused tests cover logical exact matching for a DOCX filename, import-source
metadata resolution, source-ref projection, native copy, DOCX-to-native copy,
read-plus-write credential claims, and rejection of edits against an unconverted
source. A live read-only provider check also confirmed that an extensionless
query returns the existing DOCX as one exact logical-title import source while
the literal and native-only control queries remain empty.
