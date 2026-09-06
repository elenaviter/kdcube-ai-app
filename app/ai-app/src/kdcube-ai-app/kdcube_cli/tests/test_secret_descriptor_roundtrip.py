from __future__ import annotations

import os
import stat
import time
from pathlib import Path
from typing import Any

import pytest
import yaml

from kdcube_cli.management.errors import ManagementCliError
from kdcube_cli.management.models import ManagementSecretTarget, ManagementTarget
from kdcube_cli.management.secret_descriptors import (
    load_secret_descriptors,
    write_secret_descriptors,
    write_secret_descriptors_into_directory,
)
from kdcube_cli.management.secret_export import (
    SECRET_EXPORT_RESULT_SCHEMA,
    SECRET_EXPORT_START_SCHEMA,
    ExportedSecret,
    SecretExportClient,
    SecretExportRequest,
)


def _values() -> tuple[ExportedSecret, ...]:
    return (
        ExportedSecret(
            target=ManagementSecretTarget.create(
                scope="platform",
                key="platform.services.fixture.token",
            ),
            value="platform-service-canary",
        ),
        ExportedSecret(
            target=ManagementSecretTarget.create(
                scope="platform",
                key="platform.infra.redis.password",
            ),
            value="platform-infra-canary",
        ),
        ExportedSecret(
            target=ManagementSecretTarget.create(
                scope="bundle",
                bundle_id="fixture.bundle@1.0.0",
                key="provider.token",
            ),
            value="bundle-canary",
        ),
        ExportedSecret(
            target=ManagementSecretTarget.create(
                scope="user",
                user_id="user-1",
                key="personal.token",
            ),
            value="user-canary",
        ),
        ExportedSecret(
            target=ManagementSecretTarget.create(
                scope="user",
                user_id="user-1",
                bundle_id="fixture.bundle@1.0.0",
                key="provider.token",
            ),
            value="user-bundle-canary",
        ),
    )


def test_literal_descriptor_pair_round_trips_every_secret_scope(tmp_path: Path) -> None:
    output = tmp_path / "private-descriptors"
    exported = write_secret_descriptors(output, _values())

    assert exported.platform_count == 2
    assert exported.bundle_count == 1
    assert exported.user_count == 2
    if os.name == "posix":
        assert stat.S_IMODE(output.stat().st_mode) == 0o700
        assert stat.S_IMODE(exported.platform_path.stat().st_mode) == 0o600
        assert stat.S_IMODE(exported.bundles_path.stat().st_mode) == 0o600

    platform = yaml.safe_load(exported.platform_path.read_text(encoding="utf-8"))
    bundles = yaml.safe_load(exported.bundles_path.read_text(encoding="utf-8"))
    assert set(platform) == {"platform", "users"}
    assert platform["platform"]["infra"]["redis"]["password"] == (
        "platform-infra-canary"
    )
    assert platform["users"]["user-1"]["bundles"][
        "fixture.bundle@1.0.0"
    ]["secrets"]["provider"]["token"] == "user-bundle-canary"
    assert bundles == {
        "bundles": {
            "version": "1",
            "items": [
                {
                    "id": "fixture.bundle@1.0.0",
                    "secrets": {"provider": {"token": "bundle-canary"}},
                }
            ],
        }
    }

    imported = load_secret_descriptors(output)
    assert imported.platform_count == 2
    assert imported.bundle_count == 1
    assert imported.user_count == 2
    assert {
        item.target.provider_key: item.value for item in imported.values
    } == {item.target.provider_key: item.value for item in _values()}


def test_literal_pair_can_join_an_existing_descriptor_directory(tmp_path: Path) -> None:
    output = tmp_path / "complete-descriptors"
    output.mkdir()
    (output / "assembly.yaml").write_text("context: {}\n", encoding="utf-8")

    exported = write_secret_descriptors_into_directory(output, _values())

    assert exported.directory == output
    assert (output / "assembly.yaml").read_text(encoding="utf-8") == "context: {}\n"
    assert load_secret_descriptors(output).total_count == len(_values())
    if os.name == "posix":
        assert stat.S_IMODE(output.stat().st_mode) == 0o700
        assert stat.S_IMODE(exported.platform_path.stat().st_mode) == 0o600
        assert stat.S_IMODE(exported.bundles_path.stat().st_mode) == 0o600


def test_user_only_export_keeps_the_canonical_platform_root(tmp_path: Path) -> None:
    output = tmp_path / "user-only-descriptors"
    value = ExportedSecret(
        target=ManagementSecretTarget.create(
            scope="user",
            user_id="user-1",
            key="provider.token",
        ),
        value="user-only-canary",
    )

    exported = write_secret_descriptors(output, (value,))

    platform = yaml.safe_load(exported.platform_path.read_text(encoding="utf-8"))
    assert platform == {
        "platform": {},
        "users": {
            "user-1": {"secrets": {"provider": {"token": "user-only-canary"}}}
        },
    }


def test_existing_descriptor_pair_requires_explicit_replace(tmp_path: Path) -> None:
    output = tmp_path / "complete-descriptors"
    output.mkdir()
    (output / "secrets.yaml").write_text("platform: {}\n", encoding="utf-8")
    (output / "bundles.secrets.yaml").write_text(
        "bundles:\n  version: '1'\n  items: []\n",
        encoding="utf-8",
    )

    with pytest.raises(ManagementCliError) as exists:
        write_secret_descriptors_into_directory(output, _values())
    assert exists.value.code == "secret_export_output_exists"

    exported = write_secret_descriptors_into_directory(
        output,
        _values(),
        replace=True,
    )
    assert load_secret_descriptors(exported.directory).total_count == len(_values())


def test_existing_descriptor_pair_rolls_back_a_partial_replace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "complete-descriptors"
    output.mkdir(mode=0o700)
    platform_path = output / "secrets.yaml"
    bundles_path = output / "bundles.secrets.yaml"
    old_platform = b"platform:\n  retained: old-platform\n"
    old_bundles = b"bundles:\n  version: '1'\n  items: []\n"
    platform_path.write_bytes(old_platform)
    bundles_path.write_bytes(old_bundles)
    if os.name == "posix":
        platform_path.chmod(0o600)
        bundles_path.chmod(0o600)
    real_replace = os.replace
    injected = False

    def fail_second_new_file(source: str | Path, destination: str | Path) -> None:
        nonlocal injected
        source_path = Path(source)
        if source_path.name == "bundles.secrets.yaml" and not injected:
            injected = True
            raise OSError("injected pair replacement failure")
        real_replace(source, destination)

    monkeypatch.setattr(os, "replace", fail_second_new_file)

    with pytest.raises(ManagementCliError) as failed:
        write_secret_descriptors_into_directory(output, _values(), replace=True)

    assert failed.value.code == "secret_export_output_write_failed"
    assert platform_path.read_bytes() == old_platform
    assert bundles_path.read_bytes() == old_bundles


def test_literal_import_rejects_legacy_and_ambiguous_descriptor_shapes(
    tmp_path: Path,
) -> None:
    output = tmp_path / "private-descriptors"
    output.mkdir(mode=0o700)
    platform_path = output / "secrets.yaml"
    bundles_path = output / "bundles.secrets.yaml"
    platform_path.write_text("services:\n  fixture:\n    token: legacy\n")
    bundles_path.write_text("bundles:\n  version: '1'\n  items: []\n")
    if os.name == "posix":
        platform_path.chmod(0o600)
        bundles_path.chmod(0o600)

    with pytest.raises(ManagementCliError) as legacy:
        load_secret_descriptors(output)
    assert legacy.value.code == "secret_import_namespace_invalid"

    platform_path.write_text("platform: {}\n")
    bundles_path.write_text(
        "bundles:\n"
        "  version: '1'\n"
        "  items:\n"
        "  - id: duplicate@1-0\n"
        "    secrets: {one: first}\n"
        "  - id: duplicate@1-0\n"
        "    secrets: {two: second}\n"
    )
    if os.name == "posix":
        platform_path.chmod(0o600)
        bundles_path.chmod(0o600)
    with pytest.raises(ManagementCliError) as duplicate:
        load_secret_descriptors(output)
    assert duplicate.value.code == "secret_import_descriptor_invalid"


def test_literal_import_rejects_placeholder_values(tmp_path: Path) -> None:
    output = tmp_path / "private-descriptors"
    output.mkdir(mode=0o700)
    platform_path = output / "secrets.yaml"
    bundles_path = output / "bundles.secrets.yaml"
    platform_path.write_text(
        "platform:\n  services:\n    fixture:\n      token: <FILL_ME>\n",
        encoding="utf-8",
    )
    bundles_path.write_text(
        "bundles:\n  version: '1'\n  items: []\n",
        encoding="utf-8",
    )
    if os.name == "posix":
        platform_path.chmod(0o600)
        bundles_path.chmod(0o600)

    with pytest.raises(ManagementCliError) as placeholder:
        load_secret_descriptors(output)
    assert placeholder.value.code == "secret_import_placeholder_value"


def test_bundle_only_import_does_not_require_platform_descriptor(tmp_path: Path) -> None:
    output = tmp_path / "bundle-descriptors"
    output.mkdir(mode=0o700)
    bundles_path = output / "bundles.secrets.yaml"
    bundles_path.write_text(
        "bundles:\n"
        "  version: '1'\n"
        "  items:\n"
        "  - id: fixture@1-0\n"
        "    secrets:\n"
        "      provider:\n"
        "        token: bundle-canary\n",
        encoding="utf-8",
    )
    if os.name == "posix":
        bundles_path.chmod(0o600)

    imported = load_secret_descriptors(output, include_platform=False)

    assert imported.platform_count == 0
    assert imported.bundle_count == 1
    assert imported.values[0].target.provider_key == (
        "bundles.fixture@1-0.secrets.provider.token"
    )


def test_literal_import_rejects_more_than_the_bounded_inventory(
    tmp_path: Path,
) -> None:
    output = tmp_path / "too-many-secrets"
    output.mkdir(mode=0o700)
    platform_path = output / "secrets.yaml"
    bundles_path = output / "bundles.secrets.yaml"
    platform_path.write_text(
        yaml.safe_dump(
            {
                "platform": {
                    "acceptance": {
                        f"key_{index}": f"value-{index}"
                        for index in range(4097)
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    bundles_path.write_text(
        "bundles:\n  version: '1'\n  items: []\n",
        encoding="utf-8",
    )
    if os.name == "posix":
        platform_path.chmod(0o600)
        bundles_path.chmod(0o600)

    with pytest.raises(ManagementCliError) as too_large:
        load_secret_descriptors(output)
    assert too_large.value.code == "secret_import_inventory_too_large"


class _QueuedTransport:
    def __init__(self, responses: list[tuple[int, dict[str, Any]]]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def post(self, *, url: str, payload: dict[str, Any]):
        self.calls.append((url, dict(payload)))
        return self.responses.pop(0)


@pytest.mark.asyncio
async def test_whole_export_freezes_server_inventory_for_exchange() -> None:
    target = ManagementTarget.create(
        public_base_url="https://kdcube.example.test",
        tenant="tenant-a",
        project="project-a",
    )
    frozen = tuple(sorted((item.target for item in _values()), key=lambda item: item.identity))
    request = SecretExportRequest.create(
        target=target,
        callback_uri="http://127.0.0.1:54321/callback",
        state="s" * 43,
        code_challenge="c" * 43,
        selection="all",
    )
    digest = request.request_digest_for(frozen)
    now = int(time.time())
    transaction_id = "t" * 43
    start_payload = {
        "schema": SECRET_EXPORT_START_SCHEMA,
        "ok": True,
        "transaction_id": transaction_id,
        "request_digest": digest,
        "authorization_url": (
            "https://kdcube.example.test/api/integrations/management/v1/"
            f"secrets/export/authorize?transaction={transaction_id}"
        ),
        "required_assurance": "session_confirmation",
        "expires_at": now + 120,
        "target_count": len(frozen),
        "targets": [],
    }
    by_key = {item.target.provider_key: item for item in _values()}
    result_payload = {
        "schema": SECRET_EXPORT_RESULT_SCHEMA,
        "ok": True,
        "transaction_id": transaction_id,
        "request_digest": digest,
        "target": {"tenant": "tenant-a", "project": "project-a"},
        "approval": {
            "assurance": "session_confirmation",
            "method": "browser_session",
            "verified_at": now,
        },
        "values": [
            {**item.to_dict(), "value": by_key[item.provider_key].value}
            for item in frozen
        ],
    }
    transport = _QueuedTransport([(200, start_payload), (200, result_payload)])
    client = SecretExportClient(transport=transport)

    started = await client.start(request)
    result = await client.exchange(
        request,
        started,
        code="x" * 43,
        code_verifier="v" * 64,
    )

    assert transport.calls[0][1]["selection"] == "all"
    assert "targets" not in transport.calls[0][1]
    assert started.target_count == len(frozen)
    assert started.targets == ()
    assert result.request_digest == digest
    assert [item.target for item in result.values] == list(frozen)
