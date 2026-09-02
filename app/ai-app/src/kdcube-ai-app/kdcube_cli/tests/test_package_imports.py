import os
from pathlib import Path
import subprocess
import sys


def test_frontend_config_import_does_not_load_cli_only_dependencies():
    package_root = Path(__file__).resolve().parents[1] / "src"
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(package_root), env.get("PYTHONPATH", "")) if part
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import kdcube_cli.frontend_config; "
            "assert 'rich' not in sys.modules",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_public_control_api_still_resolves_from_package_root():
    from kdcube_cli import DeploymentTargetRef

    target = DeploymentTargetRef.local(Path("/tmp/example"), tenant="t", project="p")

    assert target.target_id.endswith("/tmp/example")
    assert (target.tenant, target.project) == ("t", "p")
