from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[4]
CONFIG_IMPORTING_REQUIREMENTS = (
    "requirements-chat.txt",
    "requirements-chat-ingress.txt",
    "requirements-chat-processor.txt",
    "requirements-dbdeploy.txt",
    "requirements-metric-service.txt",
)
CONNECTION_HUB_REQUIREMENT = "connection-hub>=2026.09.01.0000"


def _connection_hub_requirement(filename: str) -> str:
    lines = (PROJECT_ROOT / filename).read_text(encoding="utf-8").splitlines()
    requirements = [
        line.strip()
        for line in lines
        if line.strip().startswith("connection-hub")
    ]
    assert len(requirements) == 1, f"{filename} must declare one Connection Hub requirement"
    requirement = requirements[0]
    assert requirement == CONNECTION_HUB_REQUIREMENT, (
        f"{filename} must use the shared Connection Hub compatibility floor "
        f"{CONNECTION_HUB_REQUIREMENT}"
    )
    return requirement


def test_shared_config_services_use_the_same_connection_hub_compatibility_floor() -> None:
    requirements = {
        _connection_hub_requirement(filename)
        for filename in CONFIG_IMPORTING_REQUIREMENTS
    }
    assert len(requirements) == 1, (
        "shared-config service images use different Connection Hub requirements: "
        f"{requirements}"
    )
