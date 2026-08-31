# Telegram notify: text and images to the caller's connected account

2026-08-31

## What was added

Four operations under `route="operations"` (registered/paid/privileged),
implemented in `services/telegram/notify.py` with a thin entrypoint facade:

- `telegram_status` — integration state, the bot's username and t.me deep
  link, and whether the caller's Connection Hub identity family carries a
  Telegram edge.
- `telegram_send` — one text message to the caller's own connected account.
  Text-only by contract: file-shaped keys are refused with a pointer to the
  images operation (the Slack named service's `post_message` rule, mirrored).
- `telegram_send_images` — images or documents with optional captions.
  Lanes in the named-services preference order: single-use `staged:` refs
  (bytes were PUT to a signed slot), public `url`, capped inline
  `content_base64` (10MB). Non-image mimes go as documents. Staged refs are
  consumed only after a successful send.
- `telegram_request_upload` — a signed single-use upload slot, the same
  staging plumbing the named services use (`_integration_upload_slot`).

## The model

No chat id in any descriptor and no per-app registry. The user links
Telegram to their KDCube account once through the deployment bot's Mini App
(Connect tab embeds the Connection Hub widget); this bundle resolves the
edge via `identity_family_resolve` and uses the Telegram user id as the
private chat id. The recipient is always the authenticated caller —
identity from the session, never the payload.

The bot is the deployment bot the workspace app runs on. The integration
row (`config/bundles.template.yaml`, `telegram.kdcube_ref`, disabled by
default) points `secret_refs.bot_token` at the Connection Hub authenticator
secret and declares NO webhook: one webhook exists per bot and the
workspace bundle owns it. This bundle only sends.

## Files

- `services/telegram/notify.py` (+ package `__init__.py`)
- `entrypoint.py` — four `@api` facades
- `config/bundles.template.yaml` — `connections.connection_hub` +
  `integrations.telegram.kdcube_ref` (enabled: false)
- `interface/kdcube-services.openapi.yaml` — four operations under the
  `Telegram Notify` tag
- `tests/test_telegram_notify.py` — 9 tests: family unwrap, deep link,
  file-key refusal, happy path to the caller's chat id, not-connected,
  staged lane with single-use consumption, inline cap, document kind,
  empty-images refusal. Suite result: 9 passed; the 3 pre-existing
  failures in named_services/productivity surface tests are unrelated
  (verified against the unmodified tree).

## Next

A `telegram` named-service realm (send as `object.action`, same staged
contract, LinkedIn-provider shape: write-only, `upload_slot_factory`, no
`file_url_factory`) so external agents reach it through the NS grammar.
