# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

# kdcube_ai_app/apps/chat/sdk/runtime/solution/solution_workspace.py

import traceback, pathlib, logging

logger = logging.getLogger(__name__)

_TEXT_MIMES = {
    "text/plain", "text/markdown", "text/x-markdown", "text/html", "text/css",
    "text/csv", "text/tab-separated-values", "text/xml",
    "application/json", "application/xml",
    "application/yaml", "application/x-yaml",
    "application/javascript", "application/x-javascript",
    "application/x-python", "text/x-python",
    "application/sql", "text/x-sql",
}

def _is_text_mime(m: str | None) -> bool:
    m = (m or "").lower().strip()
    if m in _TEXT_MIMES:
        return True
    return m.startswith("text/")

def _unique_target(base_dir: pathlib.Path, basename: str) -> pathlib.Path:
    """
    Ensure we don't overwrite duplicates; add -1, -2, ... if needed.
    """
    candidate = base_dir / basename
    if not candidate.exists():
        return candidate
    stem = pathlib.Path(basename).stem
    suf  = pathlib.Path(basename).suffix
    i = 1
    while True:
        c = base_dir / f"{stem}-{i}{suf}"
        if not c.exists():
            return c
        i += 1

async def rehost_previous_files(
        prev_files: list[dict],
        workdir: pathlib.Path,
        turn_id: str  # ← NEW: turn_id for this deliverable
) -> list[dict]:
    """
    Copy files from conversation storage into workdir/<turn_id>/,
    organizing historical files by their source turn.

    Structure:
      workdir/
        turn_1765841825124_s1lw9s/
          report.pdf
          data.xlsx
        turn_1765841834567_a2bc3d/
          chart.png

    Updated paths in artifacts reflect this structure: "turn_<id>/filename"
    """
    from kdcube_ai_app.apps.chat.sdk.storage.conversation_store import ConversationStore
    from kdcube_ai_app.apps.chat.sdk.config import get_settings

    out: list[dict] = []
    if not prev_files:
        return out

    # Create turn-specific subdirectory
    turn_dir = workdir / turn_id
    turn_dir.mkdir(parents=True, exist_ok=True)

    store = ConversationStore(get_settings().STORAGE_PATH)

    for file in prev_files:
        try:
            artifact = file.get("value") or {}
            mime = artifact.get("mime") or ""
            src_path = artifact.get("path") or ""

            if not src_path:
                # Nothing to rehost; pass through
                out.append({**artifact, "rehosted": False})
                continue

            # Extract just the filename (no parent dirs from storage)
            filename = pathlib.Path(src_path).name

            # Target: workdir/<turn_id>/filename
            target = turn_dir / filename

            if _is_text_mime(mime):
                try:
                    content = await store.backend.read_text_a(src_path)
                    target.write_text(content, encoding="utf-8")
                except Exception:
                    out.append({**artifact, "rehosted": False, "rehost_error": "read_failed"})
                    continue
            else:
                try:
                    content = await store.backend.read_bytes_a(src_path)
                    target.write_bytes(content)
                except Exception:
                    out.append({**artifact, "rehosted": False, "rehost_error": "read_failed"})
                    continue

            # Update artifact with new path (relative to workdir)
            artifact["source_path"] = src_path
            artifact["path"] = f"{turn_id}/{filename}"  # ← KEY: turn-namespaced path
            artifact["rehosted"] = True
            out.append(artifact)

        except Exception as e:
            logger.error(f"[rehost] Failed to process file: {e}\n{traceback.format_exc()}")

    return out