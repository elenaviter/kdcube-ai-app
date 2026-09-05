"""Per-turn filesystem layout for directly hosted Agent Harness adapters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from kdcube_ai_app.apps.chat.sdk.runtime.harness.workspace.layout import (
    artifact_outdir_for,
    resolve_artifact_path,
)
from kdcube_ai_app.apps.chat.sdk.runtime.harness.workspace.references import (
    is_turn_id,
)


@dataclass(frozen=True)
class DirectTurnWorkspace:
    """Canonical runtime, artifact, and package roots for one direct turn."""

    run_root: Path
    turn_id: str

    def __post_init__(self) -> None:
        root = Path(self.run_root).expanduser().resolve()
        turn_id = str(self.turn_id or "").strip()
        if not is_turn_id(turn_id):
            raise ValueError(
                "direct turn_id must use the canonical turn_... form so produced "
                "files receive valid conv:fi: references"
            )
        object.__setattr__(self, "run_root", root)
        object.__setattr__(self, "turn_id", turn_id)
        self.runtime_outdir.mkdir(parents=True, exist_ok=True)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.artifact_root.mkdir(parents=True, exist_ok=True)

    @property
    def turn_root(self) -> Path:
        return self.run_root / self.turn_id

    @property
    def runtime_outdir(self) -> Path:
        return self.turn_root / "out"

    @property
    def workdir(self) -> Path:
        return self.turn_root / "work"

    @property
    def artifact_root(self) -> Path:
        return artifact_outdir_for(self.runtime_outdir)

    def artifact_path(self, relative_path: str) -> Path:
        return resolve_artifact_path(self.runtime_outdir, relative_path)

    def current_file(self, relative_path: str) -> Path:
        clean = str(relative_path or "").strip().strip("/")
        if not clean or any(part in {"", ".", ".."} for part in Path(clean).parts):
            raise ValueError("current-turn file path must be a safe relative path")
        return self.artifact_path(f"{self.turn_id}/files/{clean}")

    def current_attachment(self, filename: str) -> Path:
        clean = Path(str(filename or "").strip()).name
        if not clean or clean in {".", ".."}:
            raise ValueError("attachment filename is required")
        return self.artifact_path(f"{self.turn_id}/attachments/{clean}")


__all__ = ["DirectTurnWorkspace"]
