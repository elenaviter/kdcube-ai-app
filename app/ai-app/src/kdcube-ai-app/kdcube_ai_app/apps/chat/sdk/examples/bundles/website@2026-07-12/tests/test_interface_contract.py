from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_descriptor_and_openapi_match_site_contract() -> None:
    descriptor = yaml.safe_load((ROOT / "config" / "bundles.template.yaml").read_text())
    item = descriptor["bundles"]["items"][0]
    site = item["config"]["ui"]["main_view"]["site"]
    assert item["id"] == "website@2026-07-12"
    assert site["enabled"] is True
    assert site["alias"] == "workspace"
    assert site["default"] is True
    assert site["hosts"] == []
    assert site["scene_application_id"] == "workspace@2026-03-31-13-36"

    contract = yaml.safe_load((ROOT / "interface" / "website.openapi.yaml").read_text())
    path = "/bundles/{tenant}/{project}/{bundle_id}/public/site_config"
    assert contract["paths"][path]["get"]["operationId"] == "site_config"


def test_entrypoint_keeps_site_config_public_and_async() -> None:
    source = (ROOT / "entrypoint.py").read_text()
    assert '@bundle_id(id="website@2026-07-12")' in source
    assert 'getattr(self.config, "ai_bundle_spec", None)' in source
    assert '"site_alias": str(site.get("alias") or "").strip()' in source
    assert '@api(method="GET", alias="site_config", route="public")' in source
    assert "async def site_config" in source
    assert '"src_folder": "ui/site"' in source


def test_site_script_accepts_site_and_control_plane_main_view_routes() -> None:
    source = (ROOT / "ui" / "site" / "site.js").read_text()
    assert "const routePatterns = [" in source
    assert "/\\/api\\/integrations\\/static\\/" in source
    assert "/\\/api\\/integrations\\/bundles\\/" in source
    assert "kdcube-site-context" in source


def test_site_shell_uses_relative_assets_and_runtime_platform_prefix() -> None:
    html = (ROOT / "ui" / "site" / "index.html").read_text()
    source = (ROOT / "ui" / "site" / "site.js").read_text()

    assert 'href="./styles.css"' in html
    assert 'src="./site.js"' in html
    assert 'href="/assets/' not in html
    assert 'src="/assets/' not in html
    assert '"/config.json"' not in source
    assert "platformConfig?.routesPrefix" in source
    assert "DEFAULT_PLATFORM_PREFIX = '/platform'" in source


def test_site_keeps_a_definite_height_chain_for_the_embedded_scene() -> None:
    html = (ROOT / "ui" / "site" / "index.html").read_text()
    source = (ROOT / "ui" / "site" / "styles.css").read_text()
    assert 'data-kdcube-resize-reporter="opt-out"' in html
    assert "height: 100%;" in source
    assert "height: 100dvh;" in source
    assert "grid-template-rows: auto minmax(0, 1fr);" in source
    assert "#workspace-scene" in source
    frame_block = source.split("#workspace-scene", 1)[1].split("}", 1)[0]
    assert "height: 100% !important;" in frame_block
