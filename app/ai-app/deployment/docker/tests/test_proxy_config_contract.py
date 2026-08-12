from __future__ import annotations

import unittest
from pathlib import Path


DOCKER_ROOT = Path(__file__).resolve().parents[1]
AI_APP_ROOT = DOCKER_ROOT.parents[1]
REPOSITORY_ROOT = AI_APP_ROOT.parents[1]


class ProxyConfigContractTest(unittest.TestCase):
    def test_proxy_image_validates_activated_config_before_start(self) -> None:
        dockerfile = (DOCKER_ROOT / "all_in_one_kdcube/Dockerfile_ProxyOpenResty").read_text(
            encoding="utf-8"
        )
        config_tool = (DOCKER_ROOT / "all_in_one_kdcube/nginx/kdcube-nginx-config").read_text(
            encoding="utf-8"
        )

        self.assertIn("ARG KDCUBE_AI_APP_SOURCE_PATH=.", dockerfile)
        self.assertIn(
            "COPY ${KDCUBE_AI_APP_SOURCE_PATH}/deployment/docker/all_in_one_kdcube/nginx/kdcube-nginx-config",
            dockerfile,
        )
        self.assertIn("kdcube-nginx-config render", dockerfile)
        self.assertIn('openresty -t -c "$runtime_candidate"', config_tool)
        self.assertIn('deployment_sha="${NGINX_CONFIG_SHA256:-}"', config_tool)
        self.assertIn("does not match this task definition", config_tool)
        self.assertIn("required by this task definition is missing", config_tool)
        self.assertIn('if [ "$candidate_sha" != "$NGINX_CONFIG_SHA256" ]', config_tool)
        self.assertIn("activate_template", config_tool)
        self.assertIn('flock -x 9', config_tool)
        self.assertIn('mv "$template_candidate" "$TEMPLATE_PATH"', config_tool)
        self.assertIn('exit 10', config_tool)
        self.assertIn('/run/kdcube/nginx-template.sha256', config_tool)

    def test_proxy_config_tool_resolves_from_release_and_local_build_contexts(self) -> None:
        relative_tool = Path("deployment/docker/all_in_one_kdcube/nginx/kdcube-nginx-config")
        release_source = AI_APP_ROOT / relative_tool
        local_source = REPOSITORY_ROOT / "app/ai-app" / relative_tool

        self.assertTrue(release_source.is_file())
        self.assertTrue(local_source.is_file())
        self.assertEqual(release_source.resolve(), local_source.resolve())

        for compose_path in (
            DOCKER_ROOT / "all_in_one_kdcube/docker-compose.yaml",
            DOCKER_ROOT / "custom-ui-managed-infra/docker-compose.yaml",
        ):
            with self.subTest(compose=compose_path):
                compose = compose_path.read_text(encoding="utf-8")
                self.assertIn("KDCUBE_AI_APP_SOURCE_PATH=app/ai-app", compose)

    def test_reference_configs_forward_one_normalized_scheme(self) -> None:
        local_configs = (
            "all_in_one_kdcube/nginx/conf/nginx_proxy.conf",
            "all_in_one_kdcube/nginx/conf/nginx_proxy_delegated.conf",
            "custom-ui-managed-infra/nginx/conf/nginx_proxy.conf",
            "custom-ui-managed-infra/nginx/conf/nginx_proxy_delegated.conf",
        )
        ecs_reference = (
            DOCKER_ROOT / "custom-ui-managed-infra/nginx/conf/nginx_proxy_ecs.conf"
        ).read_text(encoding="utf-8")

        for relative_path in local_configs:
            with self.subTest(config=relative_path):
                local_config = (DOCKER_ROOT / relative_path).read_text(encoding="utf-8")
                self.assertIn("KDCUBE_FORWARDED_PROTO_SOURCE: request", local_config)
                self.assertIn("map $scheme $forwarded_proto_last", local_config)
                self.assertIn(
                    "proxy_set_header X-Forwarded-Proto $forwarded_proto;",
                    local_config,
                )
                self.assertIn(
                    "rewrite ^/chatbot/(.*)$ /$1 break;",
                    local_config,
                )
                self.assertNotIn(
                    "proxy_set_header X-Forwarded-Proto $http_x_forwarded_proto;",
                    local_config,
                )
        self.assertIn("map $http_x_forwarded_proto $viewer_proto_last", ecs_reference)
        self.assertIn("rewrite ^/chatbot/(.*)$ /$1 break;", ecs_reference)
        self.assertIn("proxy_set_header X-Forwarded-Proto $forwarded_proto;", ecs_reference)
        self.assertIn("set_real_ip_from  <ALB_CIDR>;", ecs_reference)
        self.assertNotIn("set_real_ip_from  ${ALB_CIDR};", ecs_reference)
        self.assertNotIn(
            "proxy_set_header X-Forwarded-Proto $http_x_forwarded_proto;",
            ecs_reference,
        )


if __name__ == "__main__":
    unittest.main()
