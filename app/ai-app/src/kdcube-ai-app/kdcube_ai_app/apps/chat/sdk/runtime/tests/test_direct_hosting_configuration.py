from pathlib import Path

import pytest

from kdcube_ai_app.apps.chat.sdk.runtime.direct_hosting.configuration import (
    configured_run_directory,
    configured_tool_settings,
    configured_tools,
)


def test_tool_settings_are_owned_by_the_exact_tool_row(tmp_path: Path) -> None:
    config = {
        "agent": {
            "run_directory": "./runs",
            "tools": [
                {
                    "id": "search.primary",
                    "runtime": "local",
                    "settings": {"filter": {"allowlist": ["example.org"]}},
                },
                {"id": "files.create", "enabled": False},
            ],
        }
    }

    tools = configured_tools(config)

    assert tools[0].settings == {"filter": {"allowlist": ["example.org"]}}
    assert configured_tool_settings(config, tool_id="search.primary") == tools[0].settings
    assert configured_run_directory(
        config,
        config_path=tmp_path / "config.yaml",
    ) == (tmp_path / "runs").resolve()


def test_tool_settings_reject_unknown_id_and_non_mapping_settings() -> None:
    config = {"agent": {"tools": [{"id": "search.primary", "settings": []}]}}

    with pytest.raises(ValueError, match=r"agent\.tools\[0\]\.settings must be a mapping"):
        configured_tools(config)

    config = {"agent": {"tools": [{"id": "search.primary"}]}}
    with pytest.raises(ValueError, match="has no tool with id 'search.missing'"):
        configured_tool_settings(config, tool_id="search.missing")


def test_run_directory_must_be_a_path_string(tmp_path: Path) -> None:
    config = {"agent": {"run_directory": {"directory": "./runs"}, "tools": []}}

    with pytest.raises(ValueError, match="agent.run_directory must be a path string"):
        configured_run_directory(config, config_path=tmp_path / "config.yaml")
