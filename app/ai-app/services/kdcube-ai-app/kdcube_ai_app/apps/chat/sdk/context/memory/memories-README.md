# Memory Subsystem (Chat SDK)

**Last Updated:** 2026-01-25

---

## Overview

The memory subsystem captures per-turn preferences and facts, assistant-originated signals, and user feedback.
It then presents the most relevant memory slices to downstream agents without reloading full turn logs.

Core goals:
- Preserve per-turn preferences and facts as compact fingerprints.
- Prevent repeated assistant promotions via assistant signals.
- Keep feedback visible as a memory-like signal.
- Keep retrieval fast (index-only where possible).

---

## Terminology (stable names)

1) Assistant Signals (per-turn, assistant-originated)
   - Source: turn summarizer via a dynamic assistant-signal spec.
   - Stored in turn log. Such entry is tagged with `assistant_signal`.
   - Represents: what the assistant already promoted/claimed (promo guard).

2) User Feedback (per-turn)
   - Source: gate feedback + manual feedback injection.
   - Stored as its own artifact (`artifact:turn.log.reaction`).
   - Represents: user corrections / satisfaction / issues with prior output.

---

## Artifacts and Storage

Below are the memory-related artifacts and where their content lives.

- `artifact:turn.fingerprint.v1`
  - Storage: index-only (no S3 blob).
  - Stored in `conv_messages.text` as JSON for fast retrieval.
  - Tags:
    - `artifact:turn.fingerprint.v1`
    - `assistant_signal` (when assistant_signals exist)
    - `assistant_signal:<key>` (normalized key)

- `artifact:turn.log.reaction`
  - Storage: index-only (no S3 blob).
  - Stored in `conv_messages.text` as a serialized JSON block prefixed with `[turn.log.reaction]`.
  - Tag `origin:user` vs `origin:machine` distinguishes source.

---