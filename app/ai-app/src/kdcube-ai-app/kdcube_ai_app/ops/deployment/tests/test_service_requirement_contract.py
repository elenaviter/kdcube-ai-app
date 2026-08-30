from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[4]
CONFIG_IMPORTING_REQUIREMENTS = (
    "requirements-chat.txt",
    "requirements-chat-ingress.txt",
    "requirements-chat-processor.txt",
    "requirements-dbdeploy.txt",
    "requirements-metric-service.txt",
)


def _prokura_requirement(filename: str) -> str:
    lines = (PROJECT_ROOT / filename).read_text(encoding="utf-8").splitlines()
    requirements = [line.strip() for line in lines if line.strip().startswith("prokura")]
    assert len(requirements) == 1, f"{filename} must declare one Prokura requirement"
    requirement = requirements[0]
    assert requirement.startswith("prokura=="), f"{filename} must use an exact Prokura pin"
    return requirement


def test_shared_config_services_use_the_same_prokura_release() -> None:
    pins = {_prokura_requirement(filename) for filename in CONFIG_IMPORTING_REQUIREMENTS}
    assert len(pins) == 1, f"shared-config service images use different Prokura releases: {pins}"
