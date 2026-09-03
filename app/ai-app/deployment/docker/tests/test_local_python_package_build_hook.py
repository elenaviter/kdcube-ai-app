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
        copy_instruction = (
            "COPY deployment/docker/local-python-packages "
            "/tmp/kdcube-local-python-packages"
        )
        seed_marker = (
            "# Seed maintainer-selected distributions before dependency resolution."
        )
        normal_install = "pip install --no-cache-dir -r requirements"
        reapply_marker = (
            "# Reapply the selected source after dependency resolution."
        )

        assert dockerfile.count(copy_instruction) == 1, relative_path
        assert dockerfile.count(seed_marker) == 1, relative_path
        assert dockerfile.count(reapply_marker) == 1, relative_path
        assert dockerfile.count(
            "[ -f /tmp/kdcube-local-python-packages/requirements.txt ]"
        ) == 2, relative_path
        assert dockerfile.count(
            "-r /tmp/kdcube-local-python-packages/requirements.txt"
        ) == 2, relative_path
        assert "--no-deps --force-reinstall" in dockerfile, relative_path

        copy_index = dockerfile.index(copy_instruction)
        seed_index = dockerfile.index(seed_marker)
        normal_install_index = dockerfile.index(normal_install)
        reapply_index = dockerfile.index(reapply_marker)
        assert (
            copy_index < seed_index < normal_install_index < reapply_index
        ), relative_path
        seed_install = dockerfile[seed_index:normal_install_index]
        reapply_install = dockerfile[reapply_index:]
        assert "pip install --no-cache-dir --no-deps" in seed_install, relative_path
        assert "--force-reinstall" not in seed_install, relative_path
        assert "--no-deps --force-reinstall" in reapply_install, relative_path
