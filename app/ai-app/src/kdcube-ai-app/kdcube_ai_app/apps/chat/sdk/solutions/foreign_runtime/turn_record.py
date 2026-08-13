# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter
#
# ── turn_record.py ── run-to-completion turn recording (timing, artifacts, title) ──
#
# A run-to-completion foreign-runtime turn writes no React timeline, so the
# platform pieces React authors as it goes — the turn-timing summary, the
# recorded-events artifact reload replays, the persisted stream aggregates, the
# first-turn conversation title — must be produced explicitly around the run.
# These are FREE ASYNC FUNCTIONS that take the ``entrypoint`` (a
# ``BaseEntrypoint`` derivative) explicitly, so any foreign-runtime app calls
# them from its own ``execute_core`` / ``post_run_hook`` without inheriting a
# mixin. All are best-effort by construction: a recording failure never affects
# the turn.

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from kdcube_ai_app.apps.chat.sdk.runtime import comm_ctx
from kdcube_ai_app.apps.chat.sdk.util import _now_ms

LOGGER = logging.getLogger("kdcube.foreign_runtime.turn_record")


async def emit_turn_timing(entrypoint: Any, *, started_ms: int, total_ms: int) -> None:
    """Emit the turn-timing summary event (mirrors ``BaseWorkflow.report_timings``).

    A run-to-completion turn has no React timeline to author ``chat.turn.summary``,
    so the turn's elapsed time never reaches the recorded-events artifact the
    conversation reload replays — a reloaded turn would lose its duration. Emitting
    the SAME event here, on the recorded comm (the one the economics door's
    ``accounting.usage`` cost badge also rides), threads ``elapsed_ms`` into that
    artifact so reload restores the time exactly like React. Field name/shape match
    BaseWorkflow so the same reload reader surfaces it. Best-effort: a timing-emit
    failure never affects the turn."""
    try:
        comm = entrypoint.comm
        if comm is None:
            return
        await comm.event(
            agent="turn_controller",
            type="chat.turn.summary",
            route="chat.step",
            title="Turn Summary (Timings)",
            step="turn.summary",
            data={"elapsed_ms": int(total_ms), "started_ms": int(started_ms), "ended_ms": int(_now_ms())},
            status="completed",
        )
    except Exception:
        LOGGER.warning("[foreign-runtime] turn-timing emit failed", exc_info=True)


async def persist_turn_artifacts(
    entrypoint: Any,
    state: Dict[str, Any],
    result: Optional[Dict[str, Any]] = None,
) -> None:
    """Persist this turn's recorded chat events as the ``conv.artifacts.events``
    artifact the conversation reload replays, plus the ``conv.artifacts.stream``
    delta aggregates (subsystem/canvas replay, e.g. the code-exec panel).

    The economics door emits the turn's cost live (``accounting.usage`` — the $
    badge) and the app emits the turn's elapsed time (``chat.turn.summary``);
    both are recorded on the turn's comm (recording started by the base
    ``pre_run_hook``). But a run-to-completion app writes no React timeline, so
    without this the recorded events are never saved — the turn streamed cost +
    time live yet a reloaded turn showed neither. Persisting the SAME events
    artifact the React/workspace path persists makes reload restore both, via
    the shared SDK mechanism (no hand-rolled economics format). The stream
    fallback is inert on rich-log turns. Best-effort by construction (the base
    persistence methods swallow their own failures): a persistence failure
    never affects the turn.

    The saved-artifact user is threaded from the economics/authority projection
    when the raw ``user`` state key is empty (a foreign-runtime turn can carry
    the user on the authority projection — mirrors ``conversation_is_new``'s
    user resolution), so the artifact is scoped to the same
    (user, conversation) the reload reads. The caller's ``state`` is never
    mutated. Call this from the app's ``post_run_hook`` AFTER
    ``super().post_run_hook(...)``."""
    del result  # parity slot with the post-run hook signature; unused today
    # Scope the artifact to the record user. `_save_events_artifact` reads
    # `state["user"]`; a foreign-runtime turn can carry the user on the
    # authority projection, so fall back to it without mutating the caller's
    # state.
    save_state = state
    if not str(state.get("user") or "").strip():
        record_user = str(
            state.get("economics_user")
            or state.get("authority_user")
            or state.get("actor_user")
            or state.get("fingerprint")
            or ""
        ).strip()
        if record_user:
            save_state = dict(state)
            save_state["user"] = record_user
    await entrypoint._save_events_artifact(state=save_state)
    # Subsystem/canvas stream replay on reload (the code-exec panel): persist
    # this turn's delta aggregates as conv.artifacts.stream — the same artifact
    # React saves itself; the fallback is inert on rich-log turns.
    await entrypoint._persist_stream_artifacts_fallback(state=save_state)


async def finalize_conversation_title(
    entrypoint: Any,
    state: Dict[str, Any],
    *,
    conversation_id: str,
    question: str,
    answer: Optional[str] = None,
    title_role: Optional[str] = None,
) -> None:
    """Propose a short conversation title on the FIRST turn, stream it to the
    client, and stash it on ``state`` for the turn recorder to persist.

    Generated from the user's QUESTION (an optional ``answer`` adds signal when
    available) so it can run BEFORE the agent — the title then appears even if
    the agent's turn later errors, and never depends on a successful answer. The
    first-turn signal is framework-neutral: a new conversation has no prior
    recorded turn log. Uses the reusable SDK utility directly (no ctx_browser /
    no thinking stream — this is a run-to-completion turn). ``title_role`` names
    the accounted model role the one small title call bills under — pass the
    agent's own answer role (a known-good modern model that follows the
    two-channel protocol), not the unconfigured ``gate.simple`` default; None
    falls back to the utility default. Fail-open by construction: any failure
    leaves the turn untouched."""
    try:
        # Deferred import: the title utility transitively imports model-service
        # machinery; keep this module import-light (and framework-import-free).
        from kdcube_ai_app.apps.chat.sdk.tools.backends.summary.conversation_title import (
            emit_conversation_title_event,
            generate_conversation_title,
        )

        question = (question or "").strip()
        answer = (answer or "").strip()
        conversation_id = str(
            conversation_id or state.get("conversation_id") or state.get("session_id") or ""
        ).strip()
        svc = getattr(entrypoint, "models_service", None)
        LOGGER.info(
            "[foreign-runtime] title check conversation=%s question_len=%d svc=%s",
            conversation_id, len(question), "set" if svc is not None else "NONE",
        )
        if not question or not conversation_id or svc is None:
            LOGGER.info("[foreign-runtime] title SKIP: missing question/conversation/model-service")
            return
        if not await conversation_is_new(entrypoint, state, conversation_id=conversation_id):
            LOGGER.info("[foreign-runtime] title SKIP: conversation not new conversation=%s", conversation_id)
            return
        title_kwargs = {"role": title_role} if title_role else {}
        title = (await generate_conversation_title(
            svc, user_message=question, answer=answer or None, **title_kwargs,
        ) or "").strip()
        if not title:
            LOGGER.info("[foreign-runtime] title SKIP: model returned an empty title")
            return
        # Persist seam: the framework-neutral recorder reads this off `result`.
        state["conversation_title"] = title
        _comm = comm_ctx.get_comm()
        try:
            LOGGER.info(
                "[foreign-runtime] conversation-title generated conversation=%s title=%r "
                "comm=%s turn=%s — emitting",
                conversation_id, title, ("set" if _comm is not None else "NONE"),
                str(state.get("turn_id") or ""),
            )
        except Exception:
            pass
        # Emit seam: the SAME chat event the React workflow emits, streamed via
        # this turn's comm (the one the app already streams through), so the
        # chat component updates the conversation header live.
        await emit_conversation_title_event(
            _comm,
            conversation_id=conversation_id,
            turn_id=str(state.get("turn_id") or "").strip(),
            title=title,
        )
        try:
            LOGGER.info(
                "[foreign-runtime] conversation-title emitted conversation=%s", conversation_id
            )
        except Exception:
            pass
    except Exception:
        LOGGER.warning(
            "[foreign-runtime] conversation-title generation/emit FAILED", exc_info=True
        )


async def conversation_is_new(
    entrypoint: Any, state: Dict[str, Any], *, conversation_id: str
) -> bool:
    """A conversation is new when it has no prior recorded turn (the current
    turn's log is written after ``execute_core``). Read the platform conversation
    record — the same store the conversation list reads — so the signal matches
    what the user sees. Fail-safe to "not new" (skip the title) on any error."""
    try:
        client = await entrypoint.get_ctx_client()
        if client is None:
            LOGGER.info("[foreign-runtime] is_new: NO ctx client (pg_pool missing?) -> not new")
            return False
        # Read under the SAME user the door records the turn log under: the
        # economics door writes the minimal turn log under its `user_id`
        # (== state["economics_user"], the projected-authority record user),
        # NOT the raw `actor_user`/`user`/`fingerprint` state keys — those can
        # be empty when the user is carried only on the authority projection
        # (comm user_obj / identity_authority). Preferring `economics_user`
        # keeps record, list, and this probe agreed on (user, conversation);
        # the raw keys stay as fallbacks for a non-economics run().
        user_id = str(
            state.get("economics_user")
            or state.get("authority_user")
            or state.get("actor_user")
            or state.get("user")
            or state.get("fingerprint")
            or ""
        ).strip()
        if not user_id or not conversation_id:
            LOGGER.info(
                "[foreign-runtime] is_new: empty user_id=%r or conversation_id=%r -> not new",
                user_id, conversation_id,
            )
            return False
        res = await client.recent(
            kinds=["artifact:turn.log"],
            roles=("artifact",),
            limit=1,
            days=365,
            user_id=user_id,
            conversation_id=conversation_id,
        )
        items = res.get("items") or []
        LOGGER.info(
            "[foreign-runtime] is_new probe user=%s conversation=%s prior_turn_logs=%d -> new=%s",
            user_id, conversation_id, len(items), not items,
        )
        return not items
    except Exception:
        LOGGER.warning("[foreign-runtime] is_new probe FAILED -> not new", exc_info=True)
        return False
