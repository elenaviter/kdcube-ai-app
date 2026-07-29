from fastapi import FastAPI, Response
from fastapi.testclient import TestClient

from kdcube_ai_app.apps.chat.proc.rest.integrations import integrations


def test_public_content_head_route_reaches_public_get_handler(monkeypatch):
    async def _fake_call_bundle_op_limited(**kwargs):
        assert kwargs["request"].method == "HEAD"
        assert kwargs["operation"] == "__content__"
        assert kwargs["path_tail"] == "blog/industry/sitemap.xml"
        return Response(content=b"ok", media_type="application/xml")

    monkeypatch.setattr(integrations, "_call_bundle_op_limited", _fake_call_bundle_op_limited)

    app = FastAPI()
    app.include_router(integrations.router, prefix="/api/integrations")

    response = TestClient(app).head(
        "/api/integrations/bundles/demo/demo/news@2026-05-20-12-05"
        "/public/__content__/blog/industry/sitemap.xml"
    )

    assert response.status_code == 200


def test_head_response_preserves_headers_and_status_without_body():
    source = Response(
        content=b"<xml>body</xml>",
        media_type="application/xml",
        headers={"ETag": '"abc"'},
        status_code=200,
    )

    response = integrations._head_response_from(source)

    assert response.status_code == 200
    assert response.body == b""
    assert response.headers["etag"] == '"abc"'
    assert response.headers["content-type"].startswith("application/xml")
    assert response.headers["content-length"] == str(len(b"<xml>body</xml>"))
