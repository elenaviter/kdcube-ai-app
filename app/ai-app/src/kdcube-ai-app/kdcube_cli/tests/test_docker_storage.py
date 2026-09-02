from kdcube_cli import cli as cli_mod
from kdcube_cli.docker_storage import (
    DockerBuildStoragePolicy,
    build_storage_maintenance_commands,
    builder_cache_prune_command,
)


def test_modern_builder_cleanup_bounds_cache_and_reserves_free_space():
    command = builder_cache_prune_command(
        "--max-used-space bytes\n--min-free-space bytes\n--reserved-space bytes",
    )

    assert command == (
        "docker",
        "builder",
        "prune",
        "-f",
        "--max-used-space",
        "12GB",
        "--min-free-space",
        "8GB",
    )


def test_legacy_builder_cleanup_uses_supported_storage_limit():
    command = builder_cache_prune_command("--keep-storage bytes")

    assert command == (
        "docker",
        "builder",
        "prune",
        "-f",
        "--keep-storage",
        "12GB",
    )


def test_old_builder_cleanup_expires_stale_cache():
    command = builder_cache_prune_command(
        "Usage: docker builder prune",
        policy=DockerBuildStoragePolicy(legacy_cache_age="36h"),
    )

    assert command == (
        "docker",
        "builder",
        "prune",
        "-f",
        "--filter",
        "until=36h",
    )


def test_build_cleanup_never_prunes_named_images_or_volumes():
    commands = build_storage_maintenance_commands(
        "--max-used-space bytes\n--min-free-space bytes",
    )

    assert commands[0] == ("docker", "image", "prune", "-f")
    flattened = " ".join(part for command in commands for part in command)
    assert " volume " not in f" {flattened} "
    assert "--all" not in flattened


def test_build_storage_maintenance_reports_concise_results(monkeypatch):
    calls = []
    monkeypatch.setattr(
        cli_mod,
        "_docker_output_soft",
        lambda command, *, timeout: "--max-used-space\n--min-free-space\n",
    )

    def fake_output(command, *, timeout):
        calls.append((tuple(command), timeout))
        return "deleted: sha256:old\nTotal reclaimed space: 4.2GB\n"

    monkeypatch.setattr(cli_mod, "_docker_output", fake_output)
    console = cli_mod.Console(
        file=cli_mod.io.StringIO(), force_terminal=False, width=500
    )

    cli_mod._maintain_docker_build_storage(console, phase="after")

    assert [call[0] for call in calls] == [
        ("docker", "image", "prune", "-f"),
        (
            "docker",
            "builder",
            "prune",
            "-f",
            "--max-used-space",
            "12GB",
            "--min-free-space",
            "8GB",
        ),
    ]
    output = console.file.getvalue()
    assert "deleted: sha256:old" not in output
    assert output.count("Total reclaimed space: 4.2GB") == 2
