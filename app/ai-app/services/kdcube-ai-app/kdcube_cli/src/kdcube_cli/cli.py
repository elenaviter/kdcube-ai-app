# SPDX-License-Identifier: MIT
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.text import Text

from kdcube_cli.banner import print_cli_banner
from kdcube_cli import installer as installer_mod


DEFAULT_REPO = "https://github.com/kdcube/kdcube-ai-app.git"
DEFAULT_DIR = Path.home() / ".kdcube" / "kdcube-ai-app"
DEFAULT_WORKDIR = Path.home() / ".kdcube" / "kdcube-runtime"


def run(cmd: list[str], cwd: Path | None = None) -> None:
    subprocess.run(cmd, cwd=cwd, check=True)


def _select_option(console: Console, title: str, options: list[str], default_index: int = 0) -> str:
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        return Prompt.ask(title, choices=options, default=options[default_index])
    try:
        from readchar import readkey, key
    except Exception:
        return Prompt.ask(title, choices=options, default=options[default_index])

    idx = max(0, min(default_index, len(options) - 1))

    def _render() -> Panel:
        text = Text()
        text.append(title + "\n\n", style="bold")
        for i, option in enumerate(options):
            if i == idx:
                text.append("➤ ", style="bold cyan")
                text.append(option, style="bold cyan")
            else:
                text.append("  " + option)
            text.append("\n")
        text.append("\nUse ↑/↓ and Enter.", style="dim")
        return Panel(text, title="Select")

    with Live(_render(), console=console, refresh_per_second=30, transient=True) as live:
        while True:
            k = readkey()
            if k in (key.UP, "k"):
                idx = (idx - 1) % len(options)
            elif k in (key.DOWN, "j"):
                idx = (idx + 1) % len(options)
            elif k in (key.ENTER, "\r"):
                return options[idx]
            elif k in (key.CTRL_C, "\x03"):
                raise KeyboardInterrupt
            live.update(_render())

def _extract_platform_ref(text: str) -> str | None:
    in_platform = False
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if not line.startswith(" "):
            in_platform = line.strip().startswith("platform:")
            continue
        if in_platform and "ref:" in line:
            _, value = line.split("ref:", 1)
            value = value.strip().strip('"').strip("'")
            return value or None
    return None


def _read_local_ref(repo_root: Path) -> str | None:
    path = repo_root / "release.yaml"
    if not path.exists():
        return None
    return _extract_platform_ref(path.read_text())


def _read_remote_ref(repo_root: Path) -> str | None:
    try:
        subprocess.run(["git", "fetch", "origin", "main"], cwd=repo_root, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError:
        return None
    try:
        proc = subprocess.run(
            ["git", "show", "origin/main:release.yaml"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError:
        return None
    return _extract_platform_ref(proc.stdout)


def ensure_repo(console: Console, repo: str, target: Path) -> None:
    if target.exists() and (target / ".git").is_dir():
        console.print(f"Repo already exists at {target}")
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    console.print(f"Cloning {repo} to {target}")
    run(["git", "clone", repo, str(target)])


def _read_install_meta(workdir: Path) -> dict | None:
    meta_path = workdir / "config" / "install-meta.json"
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text())
    except Exception:
        return None


def _git_status(repo_root: Path) -> tuple[str | None, str | None, str | None]:
    """Return (local_head, remote_head, status) where status is 'up-to-date'|'behind'|'diverged'|'ahead'|None."""
    try:
        subprocess.run(["git", "fetch", "origin", "main"], cwd=repo_root, check=True, capture_output=True, text=True)
        local_head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        remote_head = subprocess.run(
            ["git", "rev-parse", "origin/main"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        counts = subprocess.run(
            ["git", "rev-list", "--left-right", "--count", "HEAD...origin/main"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        ahead, behind = (int(x) for x in counts.split())
        if ahead == 0 and behind == 0:
            status = "up-to-date"
        elif behind > 0 and ahead == 0:
            status = f"behind ({behind} commits)"
        elif ahead > 0 and behind == 0:
            status = f"ahead ({ahead} commits)"
        else:
            status = f"diverged (ahead {ahead}, behind {behind})"
        return local_head, remote_head, status
    except Exception:
        return None, None, None


def run_installer(
    console: Console,
    repo_root: Path,
    workdir: Path,
    mode: str,
    release_ref: str | None,
    docker_namespace: str | None,
) -> None:
    console.print("Launching setup wizard...")
    installer_mod.run_setup(
        console,
        repo_root=repo_root,
        workdir=workdir,
        install_mode=mode,
        release_ref=release_ref,
        docker_namespace=docker_namespace,
    )


def main() -> None:
    console = Console()
    print_cli_banner()
    parser = argparse.ArgumentParser(description="KDCube Apps bootstrap CLI")
    parser.add_argument("--repo", default=DEFAULT_REPO, help="Git repo URL")
    parser.add_argument(
        "--path",
        default=str(DEFAULT_DIR),
        help="Install directory for the repo",
    )
    parser.add_argument(
        "--reset-config",
        action="store_true",
        help="Re-run config prompts and allow editing existing values",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Alias for --reset-config",
    )
    args = parser.parse_args()

    repo_path = Path(os.path.expanduser(args.path)).resolve()
    try:
        workdir = Path(
            Prompt.ask("Compose workdir (config+data root)", default=str(DEFAULT_WORKDIR))
        ).expanduser().resolve()

        install_meta = _read_install_meta(workdir)
        if install_meta and install_meta.get("platform_ref"):
            console.print(f"[dim]Installed release (workdir):[/dim] {install_meta.get('platform_ref')}")
        elif (workdir / "config").exists():
            console.print("[dim]Installed release (workdir):[/dim] unknown (no metadata)")

        ensure_repo(console, args.repo, repo_path)
        local_ref = _read_local_ref(repo_path)
        remote_ref = _read_remote_ref(repo_path)
        if remote_ref:
            console.print(f"[dim]Latest release (remote):[/dim] {remote_ref}")
        if local_ref:
            console.print(f"[dim]Repo release.yaml:[/dim] {local_ref}")

        _, _, status = _git_status(repo_path)
        if status:
            console.print(f"[dim]Repo status:[/dim] {status}")

        choices = ["release-latest", "release-tag", "upstream", "skip"]
        if install_meta and install_meta.get("platform_ref"):
            choices.insert(1, "release-installed")
        choice = _select_option(
            console,
            "Install source",
            options=choices,
            default_index=0,
        )
        docker_namespace = None
        if choice == "upstream":
            mode = "upstream"
            release_ref = None
            run(["git", "pull"], cwd=repo_path)
        elif choice == "skip":
            mode = "skip"
            release_ref = None
        else:
            mode = "release"
            if choice == "release-installed":
                release_ref = install_meta.get("platform_ref") if install_meta else None
                if not release_ref:
                    release_ref = Prompt.ask("Release version (platform.ref)")
            elif choice == "release-tag":
                release_ref = Prompt.ask("Release version (platform.ref)")
            else:
                release_ref = remote_ref or Prompt.ask("Release version (platform.ref)")

        if args.reset_config or args.reset:
            os.environ["KDCUBE_RESET_CONFIG"] = "1"
        run_installer(console, repo_path, workdir, mode, release_ref, docker_namespace)
    except FileNotFoundError as exc:
        raise SystemExit("Missing dependency. Please install Git and Python.") from exc
    except KeyboardInterrupt:
        console.print("\n[yellow]Setup cancelled.[/yellow]")
        raise SystemExit(130)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"Command failed with exit code {exc.returncode}.") from exc


if __name__ == "__main__":
    main()
