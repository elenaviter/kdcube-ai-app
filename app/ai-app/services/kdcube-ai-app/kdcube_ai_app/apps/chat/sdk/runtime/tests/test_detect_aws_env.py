from kdcube_ai_app.apps.chat.sdk.runtime.external import detect_aws_env


class _Logger:
    def __init__(self):
        self.records = []

    def log(self, message, level="INFO"):
        self.records.append((level, message))


def test_check_and_apply_cloud_environment_passes_current_ecs_task_credentials(monkeypatch):
    monkeypatch.setenv("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI", "/v2/credentials/test")
    monkeypatch.setenv("AWS_REGION", "eu-west-1")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "eu-west-1")

    monkeypatch.setattr(
        detect_aws_env,
        "_read_aws_credentials_from_host",
        lambda: {
            "AWS_ACCESS_KEY_ID": "AKIA_TEST",
            "AWS_SECRET_ACCESS_KEY": "secret",
            "AWS_SESSION_TOKEN": "session",
            "AWS_REGION": "eu-west-1",
            "AWS_DEFAULT_REGION": "eu-west-1",
        },
    )

    env = {}
    logger = _Logger()

    detect_aws_env.check_and_apply_cloud_environment(env, logger)

    assert env["AWS_ACCESS_KEY_ID"] == "AKIA_TEST"
    assert env["AWS_SECRET_ACCESS_KEY"] == "secret"
    assert env["AWS_SESSION_TOKEN"] == "session"
    assert env["AWS_REGION"] == "eu-west-1"
    assert env["AWS_DEFAULT_REGION"] == "eu-west-1"
    assert env["AWS_EC2_METADATA_DISABLED"] == "true"
    assert any("passing current AWS credentials" in msg for _level, msg in logger.records)
