import os
from pathlib import Path

import pytest

from kdcube_cli.descriptor_files import copy_descriptor_file, write_descriptor_text


pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX file modes are required")


def _mode(path: Path) -> int:
    return path.stat().st_mode & 0o777


def test_write_secret_descriptor_repairs_mode_and_keeps_atomic_temp_private(tmp_path: Path):
    target = tmp_path / "secrets.yaml"
    target.write_text("services: {}\n", encoding="utf-8")
    target.chmod(0o644)

    write_descriptor_text(target, "services:\n  demo: value\n")

    assert _mode(target) == 0o600
    assert target.read_text(encoding="utf-8") == "services:\n  demo: value\n"
    assert list(tmp_path.glob(".secrets.yaml.tmp-*")) == []


def test_copy_bundle_secret_descriptor_creates_owner_only_target(tmp_path: Path):
    source = tmp_path / "source.yaml"
    source.write_text("bundles:\n  items: []\n", encoding="utf-8")
    target = tmp_path / "bundles.secrets.yaml"

    copy_descriptor_file(source, target)

    assert _mode(target) == 0o600
    assert target.read_bytes() == source.read_bytes()
