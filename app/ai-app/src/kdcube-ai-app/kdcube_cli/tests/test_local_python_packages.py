from pathlib import Path

import pytest

from kdcube_cli.local_python_packages import (
    clear_local_python_package_sources,
    parse_local_python_package_sources,
    stage_local_python_package_sources,
)


def _package_source(root: Path, name: str = "connection-hub") -> Path:
    source = root / name
    (source / "src" / "connection_hub").mkdir(parents=True)
    (source / "pyproject.toml").write_text(
        "[build-system]\nrequires = ['setuptools>=68']\n",
        encoding="utf-8",
    )
    (source / "src" / "connection_hub" / "__init__.py").write_text(
        "VERSION = 'local'\n",
        encoding="utf-8",
    )
    (source / "build").mkdir()
    (source / "build" / "stale.txt").write_text("stale", encoding="utf-8")
    return source


def test_parse_local_python_package_source_validates_shape(tmp_path: Path):
    source = _package_source(tmp_path)

    parsed = parse_local_python_package_sources(
        [f"Connection_Hub={source}"]
    )

    assert len(parsed) == 1
    assert parsed[0].distribution == "connection-hub"
    assert parsed[0].source == source.resolve()

    with pytest.raises(ValueError, match="DIST=SOURCE_DIR"):
        parse_local_python_package_sources([str(source)])
    with pytest.raises(ValueError, match="Duplicate"):
        parse_local_python_package_sources(
            [f"connection-hub={source}", f"connection_hub={source}"]
        )


def test_stage_local_python_package_source_is_transient(tmp_path: Path):
    repo_root = tmp_path / "repo"
    stage_root = (
        repo_root
        / "app"
        / "ai-app"
        / "deployment"
        / "docker"
        / "local-python-packages"
    )
    stage_root.mkdir(parents=True)
    (stage_root / ".gitkeep").write_text("", encoding="utf-8")
    source = _package_source(tmp_path / "source")
    package = parse_local_python_package_sources(
        [f"connection-hub={source}"]
    )[0]

    staged = stage_local_python_package_sources(repo_root, [package])

    assert staged == stage_root
    assert (
        stage_root / "sources" / "connection-hub" / "src" / "connection_hub" / "__init__.py"
    ).is_file()
    assert not (stage_root / "sources" / "connection-hub" / "build").exists()
    assert (stage_root / "requirements.txt").read_text(encoding="utf-8") == (
        "/tmp/kdcube-local-python-packages/sources/connection-hub\n"
    )

    clear_local_python_package_sources(repo_root)

    assert (stage_root / ".gitkeep").is_file()
    assert not (stage_root / "sources").exists()
    assert not (stage_root / "requirements.txt").exists()
    assert not (stage_root / "manifest.json").exists()
