# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

# example_bootstrap.py
import os
from botocore.config import Config as BotoConfig
from kdcube_ai_app.auth.service_auth.base import IdpConfig
from kdcube_ai_app.auth.service_auth.factory import create_service_idp

def build_service_idp_from_env():
    provider = os.getenv("IDP_PROVIDER", "cognito")
    if provider == "cognito":
        cfg = IdpConfig(
            "cognito",
            region=os.getenv("COGNITO_REGION"),
            user_pool_id=os.getenv("COGNITO_USER_POOL_ID"),
            client_id=os.getenv("COGNITO_SERVICE_CLIENT_ID"),
            client_secret=os.getenv("COGNITO_SERVICE_CLIENT_SECRET", None),  # ok if None
            username=os.getenv("OIDC_SERVICE_ADMIN_USERNAME"),
            password=os.getenv("OIDC_SERVICE_ADMIN_PASSWORD"),
            new_password=os.getenv("OIDC_SERVICE_ADMIN_NEW_PASSWORD", None),  # <- only if first-login
            use_admin_api=True,
            boto_cfg=BotoConfig(retries={"max_attempts": 3, "mode": "standard"}),
        )
        return create_service_idp(cfg)
    # else: add other providers here
    return create_service_idp(IdpConfig(provider))

# kb_client_rest.py
import aiohttp
import asyncio
from typing import Optional, Dict, Any
from kdcube_ai_app.auth.service_auth.base import ServiceIdP, TokenBundle, build_auth_headers

ID_HEADER_NAME = os.getenv("KDCUBE_ID_TOKEN_HEADER_NAME", "X-ID-Token")

class KBServiceClient:
    def __init__(self, idp: ServiceIdP, base_url: str):
        self.idp = idp
        self.base_url = base_url.rstrip("/")
        self._tokens: Optional[TokenBundle] = None

    async def _ensure_tokens(self):
        if self._tokens is None:
            self._tokens = await asyncio.to_thread(self.idp.authenticate)
        elif self._tokens.is_access_expired():
            self._tokens = await asyncio.to_thread(self.idp.refresh, self._tokens)

    async def _post_json(self, url: str, payload: Dict[str, Any], headers: Dict[str, str], timeout_sec: int) -> Dict[str, Any]:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout_sec)) as sess:
            async with sess.post(url, json=payload, headers=headers) as resp:
                if resp.status in (401, 403):
                    raise PermissionError(await resp.text())
                resp.raise_for_status()
                return await resp.json()

    async def enhanced_search_on_behalf(
            self,
            *,
            project: Optional[str],
            query: str,
            on_behalf_session_id: str,
            top_k: int = 5,
            resource_id: Optional[str] = None,
            user_tokens_passthrough: Optional[Dict[str, str]] = None,
            id_header_name: str = ID_HEADER_NAME,
            timeout_sec: int = 20,
    ) -> Dict[str, Any]:

        await self._ensure_tokens()

        url = f"{self.base_url}/api/kb"
        if project:
            url += f"/{project}"
        url += "/search/enhanced"

        headers = build_auth_headers(
            self._tokens,
            id_header_name=id_header_name,
            on_behalf_session_id=on_behalf_session_id,
        )
        if user_tokens_passthrough:
            if "access_token" in user_tokens_passthrough:
                headers["X-User-Access-Token"] = user_tokens_passthrough["access_token"]
            if "id_token" in user_tokens_passthrough:
                headers["X-User-Id-Token"] = user_tokens_passthrough["id_token"]

        payload = {
            "query": query,
            "resource_id": resource_id,
            "top_k": top_k,
            "include_backtrack": True,
            "include_navigation": True,
        }

        try:
            return await self._post_json(url, payload, headers, timeout_sec)
        except PermissionError:
            # single refresh-and-retry
            self._tokens = await asyncio.to_thread(self.idp.refresh, self._tokens)
            headers = build_auth_headers(self._tokens, id_header_name=id_header_name, on_behalf_session_id=on_behalf_session_id)
            return await self._post_json(url, payload, headers, timeout_sec)

if __name__ == "__main__":

    import asyncio

    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())

    project = os.environ.get("DEFAULT_PROJECT_NAME")
    tenant = os.environ.get("TENANT_ID")

    kb_base_url = os.getenv("KDCUBE_KB_BASE_URL", "http://localhost:8000")
    on_behalf_session_id = "1575eaf7-ca97-4f7e-a6e3-fca107400a90"

    query = "usage of unauthorized ai app"

    async def run():
        idp = build_service_idp_from_env()
        client = KBServiceClient(idp, base_url=kb_base_url)

        result = await client.enhanced_search_on_behalf(
            project=project,
            query=query,
            on_behalf_session_id=on_behalf_session_id,
            top_k=5,
        )
        print(result)
        idp.close()

    asyncio.run(run())

