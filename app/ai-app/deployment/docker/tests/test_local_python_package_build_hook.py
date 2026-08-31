# SPDX-License-Identifier: MIT
from pathlib import Path


DOCKER_ROOT = Path(__file__).resolve().parents[1]
LOCAL_PACKAGE_DOCKERFILES = (
    "all_in_one_kdcube/Dockerfile_Chatproc",
    "all_in_one_kdcube/Dockerfile_Exec",
    "all_in_one_kdcube/Dockerfile_Ingress",
    "all_in_one_kdcube/Dockerfile_Metricservice",
    "all_in_one_kdcube/Dockerfile_PostgresSetup",
    "custom-ui-managed-infra/Dockerfile_Chatproc",
    "custom-ui-managed-infra/Dockerfile_Exec",
    "custom-ui-managed-infra/Dockerfile_Ingress",
    "custom-ui-managed-infra/Dockerfile_Metricservice",
    "custom-ui-managed-infra/Dockerfile_PostgresSetup",
)


def test_python_images_install_staged_maintainer_package_overrides():
    for relative_path in LOCAL_PACKAGE_DOCKERFILES:
        dockerfile = (DOCKER_ROOT / relative_path).read_text(encoding="utf-8")
        assert (
            "COPY deployment/docker/local-python-packages "
            "/tmp/kdcube-local-python-packages"
        ) in dockerfile, relative_path
        assert (
            "[ -f /tmp/kdcube-local-python-packages/requirements.txt ]"
        ) in dockerfile, relative_path
        assert "--no-deps --force-reinstall" in dockerfile, relative_path
        assert (
            "-r /tmp/kdcube-local-python-packages/requirements.txt"
        ) in dockerfile, relative_path
