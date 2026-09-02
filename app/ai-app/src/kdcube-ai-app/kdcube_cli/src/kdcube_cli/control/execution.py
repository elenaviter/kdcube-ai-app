# SPDX-License-Identifier: MIT
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Protocol, Sequence


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str = ""
    stderr: str = ""


class CommandRunner(Protocol):
    def run(
        self,
        command: Sequence[str],
        *,
        cwd: Optional[Path] = None,
        env: Optional[Mapping[str, str]] = None,
        timeout: Optional[float] = None,
        capture_output: bool = False,
    ) -> CommandResult:
        ...


class SubprocessCommandRunner:
    def run(
        self,
        command: Sequence[str],
        *,
        cwd: Optional[Path] = None,
        env: Optional[Mapping[str, str]] = None,
        timeout: Optional[float] = None,
        capture_output: bool = False,
    ) -> CommandResult:
        completed = subprocess.run(
            list(command),
            cwd=str(cwd) if cwd is not None else None,
            env=dict(env) if env is not None else None,
            timeout=timeout,
            capture_output=capture_output,
            text=capture_output,
            check=False,
        )
        return CommandResult(
            returncode=int(completed.returncode),
            stdout=str(completed.stdout or "") if capture_output else "",
            stderr=str(completed.stderr or "") if capture_output else "",
        )
