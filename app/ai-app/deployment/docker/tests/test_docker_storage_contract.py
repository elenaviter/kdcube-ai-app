from pathlib import Path

import yaml


DOCKER_ROOT = Path(__file__).resolve().parents[1]
COMPOSE_FILES = tuple(
    DOCKER_ROOT / directory / "docker-compose.yaml"
    for directory in (
        "all_in_one_kdcube",
        "custom-ui-managed-infra",
        "local-infra-stack",
    )
)


def test_local_compose_services_bound_docker_managed_logs():
    assert COMPOSE_FILES
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
