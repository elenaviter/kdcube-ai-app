from pathlib import Path

import yaml


DOCKER_ROOT = Path(__file__).resolve().parents[1]
LEGACY_LOCAL_ONLY_DIRS = {"all_in_one"}
COMPOSE_FILES = tuple(
    path
    for path in sorted(DOCKER_ROOT.glob("*/docker-compose.yaml"))
    if path.parent.name not in LEGACY_LOCAL_ONLY_DIRS
)


def test_local_compose_services_bound_docker_managed_logs():
    assert COMPOSE_FILES
    assert {path.parent.name for path in COMPOSE_FILES} >= {
        "all_in_one_kdcube",
        "custom-ui-managed-infra",
        "local-infra-stack",
    }
    for compose_path in COMPOSE_FILES:
        compose = yaml.safe_load(compose_path.read_text(encoding="utf-8"))
        services = compose.get("services") or {}
        assert services, compose_path
        for service_name, service in services.items():
            logging = service.get("logging") or {}
            assert logging.get("driver") == "json-file", (compose_path, service_name)
            assert logging.get("options") == {
                "max-size": "20m",
                "max-file": "3",
            }, (compose_path, service_name)
