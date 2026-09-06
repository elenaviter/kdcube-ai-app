"""Argument parser for canonical KDCube secret commands."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import Any

AddQuietArgument = Callable[[argparse.ArgumentParser], None]
_MANAGEMENT_COMMANDS = frozenset(
    {"metadata", "get", "set", "delete", "export", "import"}
)


def _add_target_options(command: argparse.ArgumentParser) -> None:
    target = command.add_mutually_exclusive_group()
    target.add_argument(
        "--endpoint",
        help="KDCube HTTPS origin, or a loopback HTTP origin for local use.",
    )
    target.add_argument(
        "--workdir",
        default=None,
        help="Fully-qualified local runtime workdir.",
    )
    command.add_argument("--tenant", default="", help="Target tenant.")
    command.add_argument("--project", default="", help="Target project.")


def _add_management_options(command: argparse.ArgumentParser) -> None:
    _add_target_options(command)
    command.add_argument(
        "--credential-stdin",
        action="store_true",
        help=(
            "Read the delegated bearer from the first standard-input line. "
            "The bearer is never accepted on the command line."
        ),
    )
    command.add_argument(
        "--invocation-id",
        help="Stable idempotency key used when retrying an approved request.",
    )
    command.add_argument(
        "--no-open",
        action="store_true",
        help="Return consent recovery without opening its browser page.",
    )
    command.add_argument(
        "--no-wait",
        action="store_true",
        help="Open consent without waiting for an interactive retry.",
    )
    command.add_argument("--json", action="store_true", help=argparse.SUPPRESS)


def _add_secret_target(command: argparse.ArgumentParser) -> None:
    command.add_argument(
        "key",
        help=(
            "Exact canonical key. Platform keys begin with platform.; "
            "bundle and user keys are relative to their explicit scope."
        ),
    )
    command.add_argument(
        "--scope",
        choices=("platform", "bundle", "user"),
        required=True,
        help="Platform, application-bundle, or user-owned scope.",
    )
    command.add_argument(
        "--bundle-id",
        default="",
        help="Exact application id; required for --scope bundle.",
    )
    command.add_argument(
        "--user-id",
        default="",
        help="Exact immutable user id; required for --scope user.",
    )
    _add_management_options(command)


def _add_local_backend_options(
    command: argparse.ArgumentParser,
    *,
    add_quiet: AddQuietArgument,
    default_path: Path,
) -> None:
    add_quiet(command)
    command.add_argument("--tenant", default="", help="Local runtime tenant.")
    command.add_argument("--project", default="", help="Local runtime project.")
    command.add_argument(
        "--workdir",
        default=None,
        help="Fully-qualified local runtime workdir.",
    )
    command.add_argument(
        "--path",
        default=str(default_path),
        help="Platform repository path.",
    )


def _add_host_vault_parser(
    commands: Any,
    *,
    add_quiet: AddQuietArgument,
    default_path: Path,
    compatibility_alias: bool = False,
) -> None:
    host_vault = commands.add_parser(
        "host-vault",
        help=(
            "Compatibility alias for `secrets backend host-vault`."
            if compatibility_alias
            else "Prepare, migrate, or recover the local host-vault backend."
        ),
    )
    actions = host_vault.add_subparsers(dest="secrets_action", required=True)

    prepare = actions.add_parser(
        "prepare",
        help="Project shadow configuration and recreate only the secrets broker.",
    )
    _add_local_backend_options(
        prepare,
        add_quiet=add_quiet,
        default_path=default_path,
    )
    prepare.add_argument("--dry-run", action="store_true")
    prepare.add_argument("--wait-seconds", type=float, default=120.0)
    prepare.add_argument("--json", action="store_true", dest="json_output")

    stage = actions.add_parser(
        "stage",
        help="Copy and verify file-backed values without changing providers.",
    )
    _add_local_backend_options(stage, add_quiet=add_quiet, default_path=default_path)
    stage.add_argument("--dry-run", action="store_true")
    stage.add_argument("--json", action="store_true", dest="json_output")

    activate = actions.add_parser(
        "activate",
        help="Switch verified local consumers from secrets-file to host-vault.",
    )
    _add_local_backend_options(
        activate,
        add_quiet=add_quiet,
        default_path=default_path,
    )
    activate.add_argument("--dry-run", action="store_true")
    activate.add_argument("--yes", action="store_true")
    activate.add_argument("--wait-seconds", type=float, default=120.0)
    activate.add_argument("--json", action="store_true", dest="json_output")

    recover = actions.add_parser(
        "recover",
        help="Recover an interrupted activation to verified file-backed operation.",
    )
    _add_local_backend_options(
        recover,
        add_quiet=add_quiet,
        default_path=default_path,
    )
    recover.add_argument("--yes", action="store_true")
    recover.add_argument("--wait-seconds", type=float, default=120.0)
    recover.add_argument("--json", action="store_true", dest="json_output")


def configure_secrets_parser(
    parser: argparse.ArgumentParser,
    *,
    add_quiet: AddQuietArgument,
    default_path: Path,
) -> None:
    """Configure logical secret commands and backend lifecycle commands."""

    commands = parser.add_subparsers(dest="secrets_command", required=True)
    metadata = commands.add_parser(
        "metadata",
        help="Check exact-key existence and provider capability without reading a value.",
    )
    _add_secret_target(metadata)

    get = commands.add_parser(
        "get",
        help="Disclose one exact value into a private local file.",
    )
    _add_secret_target(get)
    get.add_argument("--output", required=True, help="New private output file.")
    get.add_argument("--replace", action="store_true")

    set_command = commands.add_parser(
        "set",
        help="Set one exact value through the deployment-selected provider.",
    )
    _add_secret_target(set_command)
    set_command.add_argument(
        "--value-stdin",
        action="store_true",
        help=(
            "Read the exact value from standard input. With --credential-stdin, "
            "the bearer is the first line and the value is the remaining input."
        ),
    )

    delete = commands.add_parser(
        "delete",
        help="Delete one exact value through the deployment-selected provider.",
    )
    _add_secret_target(delete)

    export = commands.add_parser(
        "export",
        help="Export current values once after explicit browser confirmation.",
    )
    _add_target_options(export)
    export.add_argument(
        "--all",
        action="store_true",
        dest="all_secrets",
        help="Export the complete deployment inventory, including all users.",
    )
    export.add_argument("--platform-key", action="append", default=[])
    export.add_argument(
        "--bundle-key",
        action="append",
        default=[],
        metavar="BUNDLE_ID=KEY",
    )
    export.add_argument(
        "--user-key",
        action="append",
        default=[],
        metavar="USER_ID=KEY",
    )
    export.add_argument(
        "--user-bundle-key",
        action="append",
        default=[],
        metavar="USER_ID/BUNDLE_ID=KEY",
    )
    export.add_argument("--output-directory", required=True)
    export.add_argument(
        "--into-descriptor-directory",
        action="store_true",
        help=(
            "Write secrets.yaml and bundles.secrets.yaml into an existing "
            "ordinary descriptor directory."
        ),
    )
    export.add_argument(
        "--replace-descriptor-files",
        action="store_true",
        help=(
            "Replace only existing secrets.yaml and bundles.secrets.yaml; "
            "requires --into-descriptor-directory."
        ),
    )
    export.add_argument("--no-open", action="store_true")
    export.add_argument("--wait-seconds", type=float, default=300.0)
    export.add_argument("--json", action="store_true", help=argparse.SUPPRESS)

    import_command = commands.add_parser(
        "import",
        help=(
            "Apply values from the ordinary secrets.yaml and "
            "bundles.secrets.yaml descriptor pair."
        ),
    )
    _add_management_options(import_command)
    import_command.add_argument(
        "--input-directory",
        required=True,
        help="Directory containing secrets.yaml and bundles.secrets.yaml.",
    )
    import_command.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and inventory the descriptors without sending values.",
    )
    import_command.add_argument(
        "--yes",
        action="store_true",
        help="Confirm that every descriptor value should be applied.",
    )

    namespace = commands.add_parser(
        "namespace",
        help="Migrate persisted secret identities to the canonical namespace.",
    )
    namespace_actions = namespace.add_subparsers(
        dest="secrets_namespace_action",
        required=True,
    )
    migrate = namespace_actions.add_parser(
        "migrate",
        help="Move legacy platform roots under platform: without exposing values.",
    )
    add_quiet(migrate)
    migrate.add_argument("--tenant", default="")
    migrate.add_argument("--project", default="")
    migrate.add_argument("--workdir", default=None)
    migrate.add_argument("--dry-run", action="store_true")
    migrate.add_argument("--yes", action="store_true")
    migrate.add_argument("--json", action="store_true", dest="json_output")

    backend = commands.add_parser(
        "backend",
        help="Inspect or migrate the descriptor-selected secret backend.",
    )
    backend_commands = backend.add_subparsers(
        dest="secrets_backend_action",
        required=True,
    )
    status = backend_commands.add_parser(
        "status",
        help="Show where this local deployment is configured to store secrets.",
    )
    add_quiet(status)
    status.add_argument("--tenant", default="")
    status.add_argument("--project", default="")
    status.add_argument("--workdir", default=None)
    status.add_argument("--json", action="store_true", dest="json_output")
    _add_host_vault_parser(
        backend_commands,
        add_quiet=add_quiet,
        default_path=default_path,
    )
    _add_host_vault_parser(
        commands,
        add_quiet=add_quiet,
        default_path=default_path,
        compatibility_alias=True,
    )


def is_management_secret_command(args: argparse.Namespace) -> bool:
    return getattr(args, "secrets_command", "") in _MANAGEMENT_COMMANDS


def is_backend_status_command(args: argparse.Namespace) -> bool:
    return (
        getattr(args, "secrets_command", "") == "backend"
        and getattr(args, "secrets_backend_action", "") == "status"
    )


def is_secret_namespace_command(args: argparse.Namespace) -> bool:
    return (
        getattr(args, "secrets_command", "") == "namespace"
        and getattr(args, "secrets_namespace_action", "") == "migrate"
    )


def is_host_vault_lifecycle_command(args: argparse.Namespace) -> bool:
    return getattr(args, "secrets_command", "") == "host-vault" or (
        getattr(args, "secrets_command", "") == "backend"
        and getattr(args, "secrets_backend_action", "") == "host-vault"
    )


__all__ = [
    "configure_secrets_parser",
    "is_backend_status_command",
    "is_host_vault_lifecycle_command",
    "is_management_secret_command",
    "is_secret_namespace_command",
]
