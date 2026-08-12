from __future__ import annotations

import re
import subprocess
from pathlib import Path


WEB_APP_ROOT = Path(__file__).resolve().parents[1]


ROOT_FRONTEND_FILE_PATTERNS = [
    re.compile(r"""(?:src|href)=["']/(?:assets|img)/"""),
    re.compile(r"""(?:src|href)=["']/config\.json["']"""),
    re.compile(r"""["'`]\/(?:assets|img)\/"""),
    re.compile(r"""["'`]\/config\.json["'`]"""),
    re.compile(r"""url\(\s*["']?/(?:assets|img)/"""),
]


def _build_artifact(tmp_path: Path) -> Path:
    out_dir = tmp_path / "chat-web-app-dist"
    subprocess.run(
        ["npm", "run", "build_no_lint", "--", "--outDir", str(out_dir), "--emptyOutDir"],
        cwd=WEB_APP_ROOT,
        check=True,
    )
    return out_dir


def test_built_control_plane_artifact_has_no_root_frontend_file_urls(tmp_path: Path) -> None:
    out_dir = _build_artifact(tmp_path)
    index_html = (out_dir / "index.html").read_text(encoding="utf-8")

    assert "data-kdcube-control-plane-bootstrap" in index_html
    assert "assets/" in index_html
    assert "img/favicon.svg" in index_html

    checked_files = [out_dir / "index.html"]
    checked_files.extend((out_dir / "assets").glob("*.js"))
    checked_files.extend((out_dir / "assets").glob("*.css"))

    for path in checked_files:
        content = path.read_text(encoding="utf-8")
        for pattern in ROOT_FRONTEND_FILE_PATTERNS:
            assert not pattern.search(content), f"{path.relative_to(out_dir)} contains root frontend URL"


def test_built_control_plane_artifact_contains_runtime_mount_logic(tmp_path: Path) -> None:
    out_dir = _build_artifact(tmp_path)
    index_html = (out_dir / "index.html").read_text(encoding="utf-8")

    assert '"chat", "callback", "dummy"' in index_html
    assert "window.__KDCUBE_CONTROL_PLANE_MOUNT__ = mount" in index_html
    assert "import(publicUrl(src))" in index_html
    assert "/platform" not in index_html
    assert "/control/ui" not in index_html
