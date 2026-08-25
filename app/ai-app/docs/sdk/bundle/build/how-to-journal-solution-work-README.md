---
id: repo:kdcube-ai-app/app/ai-app/docs/sdk/bundle/build/how-to-journal-solution-work-README.md
title: "How To Journal Solution Work Across Repositories"
summary: "Maintenance procedure for the two journal layers of a KDCube solution: each app's own docs/journal plus one central solution journal, with reciprocal Related Journals links, pointer indexes, and clock-true timestamps."
tags: ["sdk", "bundle", "maintenance", "journal", "documentation", "process"]
keywords: ["solution journal", "central journal", "feature journal", "chronicles index", "related journals", "reciprocal journal links", "journal pointer index", "bundle maintenance journaling", "dated journal entry", "journal timestamps"]
updated_at: 2026-08-25
see_also:
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/bundle/build/how-to-write-bundle-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/bundle/build/how-to-navigate-kdcube-docs-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/bundle/build/how-to-release-bundle-content-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/bundle/build/sync-tier1-bundle-docs-to-build-with-kdcube-plugins-README.md
---
# How To Journal Solution Work Across Repositories

Every KDCube app keeps its own memory of WHY in `docs/journal/` (the
contract is in
[how-to-write-bundle-README.md](how-to-write-bundle-README.md)). A real
solution is usually bigger than one app: several bundles, a content
repository, descriptors across environments, procedures, a website. This
page is the maintenance procedure for that larger memory: one central
solution journal beside the per-app journals, with links that work in
both directions.

Agents maintaining a solution should follow this procedure whenever a
change spans more than one app or repository, and should read the
relevant journals before touching in-flight work.

## The two layers

**The app layer.** Each bundle keeps `docs/journal/` with dated entries
(`YYYY-MM-DD-<topic>.md`), a `journal.md` chronological index, and a
`README.md` human index. This layer answers "why is this app the way it
is". It ships with the app and travels with the app's repository.

**The solution layer.** The operator designates one repository as the
solution home (typically the bundle/content repository the workspace
already uses). That repository carries the central journal:

```
<solution-repo>/docs/journal/
  YY/MM/<feature>/journal/
    chronicles.md                      # the index: one row per entry
    <yyyy.mm.dd.HHMM>-<topic>.md       # one entry per significant move
```

One folder per feature or workstream, entries accumulated under it,
`chronicles.md` as the index table (date, entry link, one-line what
happened). This layer answers "why is the solution the way it is":
decisions that cross apps, descriptor rollouts, procedure changes,
investigations whose evidence spans repositories.

## Which layer gets the entry

- A decision local to one app: the app's `docs/journal/`, per the app
  contract. Nothing else.
- A move that crosses apps, environments, descriptors, or procedures:
  one entry in the central journal. Do not write the same entry twice;
  the app side links to it instead (next section).
- When one change produces both (an app-local mechanism plus a
  solution-level rollout), each journal records its own scope and the
  two entries link to each other. Neither restates the other's content.

## Pointer indexes: when the history lives centrally

Some maintained things are not apps with their own `docs/journal/`: a
procedure package, a content folder, a script suite. Their entries live
in the central journal only, but the thing itself must not become
historyless. Give it a `journal/README.md` pointer index inside its own
folder: a relative link to each central entry about it plus one line
saying what that entry holds for this package. Add the pointer row in
the same change as the central entry and its chronicles row, and let the
central chronicles name the pointer folder back. Both directions exist
from the first entry on.

## Reciprocal Related Journals

When work in one feature journal crosses a concern owned by another
(an investigation reuses another feature's benchmark, a fix lands in a
flow another journal documents), connect the two indexes from BOTH
sides. Each `chronicles.md` (and, for an app, its `docs/journal/`
index) gains a `## Related Journals` section with this exact table
shape:

| Journal | Connection | Boundary |
| --- | --- | --- |
| `[Other Journal](../../other-feature/journal/chronicles.md)` | Name the shared flow, evidence, change, or operational dependency. Link the exact entry when it is the evidence being referenced. | State what each journal owns and what the referenced conclusions do **not** establish. |

Keep `Connection` concrete and `Boundary` explicit. For example, a
widget request benchmark may inform a page-loading investigation while
not measuring the static shell: say so in `Boundary`. The backlink in
the other journal describes the same relationship from its own scope.
Never duplicate the other journal's incident text or measurements.

## Timestamps come from the clock

Run `date` before stamping an entry or an index row; never extrapolate
a plausible time from the narrative. A wrong stamp that already shipped
gets a one-line correction inside the entry; the filename stays,
because it is an address other records cite.

## Hygiene

- No secrets, tokens, or user credentials in any journal, ever. Journals
  are tracked files; treat them as readable by every collaborator.
- Journals are working memory, not documentation. Public product docs
  carry the canonical conclusions and link canonical documentation;
  they should not depend on working-journal evidence.
- Update the journal in the same change as the work it records, not as
  a later batch. An entry written while the context is live is the
  entry a future maintainer can actually use.
