# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

from __future__ import annotations
from typing import Optional, Dict, Any
import base64, hashlib, hmac
import boto3
import time
from botocore.config import Config as BotoConfig

from kdcube_ai_app.auth.service_auth.base import ServiceIdP, TokenBundle

class CognitoServiceAuth(ServiceIdP):
    """
    Server-side sign-in + refresh for Amazon Cognito.

    - Admin password flow (ADMIN_USER_PASSWORD_AUTH) by default.
    - Handles NEW_PASSWORD_REQUIRED challenge (if `new_password` is provided).
    - Supports client secret (SECRET_HASH) when your app client has one.
    - Refresh via REFRESH_TOKEN_AUTH (keeps prior refresh token if Cognito omits it).
    """

    def __init__(
            self,
            *,
            region: str,
            user_pool_id: str,
            client_id: str,
            client_secret: Optional[str] = None,
            username: str,
            password: str,
            new_password: Optional[str] = None,        # <- used only if challenge arises
            use_admin_api: bool = True,                # keep your old switch
            boto_cfg: Optional[BotoConfig] = None,
    ):
        self.region = region
        self.user_pool_id = user_pool_id
        self.client_id = client_id
        self.client_secret = client_secret
        self.username = username
        self.password = password
        self.new_password = new_password
        self.use_admin_api = use_admin_api
        self._client = boto3.client("cognito-idp", region_name=region, config=boto_cfg)

    # ---------- helpers ----------

    def _secret_hash(self, username: str) -> Optional[str]:
        """Compute Cognito SECRET_HASH if app client has a secret; else None."""
        if not self.client_secret:
            return None
        mac = hmac.new(
            self.client_secret.encode("utf-8"),
            (username + self.client_id).encode("utf-8"),
            hashlib.sha256,
        ).digest()
        return base64.b64encode(mac).decode()

    def _to_bundle(self, result: Dict[str, Any], keep_refresh: Optional[str] = None) -> TokenBundle:
        auth = result["AuthenticationResult"]
        bundle = TokenBundle(
            access_token=auth["AccessToken"],
            id_token=auth["IdToken"],
            refresh_token=auth.get("RefreshToken") or keep_refresh,
            token_type=auth.get("TokenType", "Bearer"),
            access_expires_at=auth.get("ExpiresIn"),   # seconds
        )
        bundle.ensure_exp_fields()
        return bundle

    # ---------- ServiceIdP interface ----------

    def authenticate(self) -> TokenBundle:
        """Password sign-in; handles first-login challenge if needed."""
        params: Dict[str, Any] = {
            "ClientId": self.client_id,
            "AuthParameters": {
                "USERNAME": self.username,
                "PASSWORD": self.password,
            },
        }
        sh = self._secret_hash(self.username)
        if sh:
            params["AuthParameters"]["SECRET_HASH"] = sh

        if self.use_admin_api:
            params["UserPoolId"] = self.user_pool_id
            params["AuthFlow"] = "ADMIN_USER_PASSWORD_AUTH"
            res = self._client.admin_initiate_auth(**params)
        else:
            params["AuthFlow"] = "USER_PASSWORD_AUTH"
            res = self._client.initiate_auth(**params)

        # Success straight away
        if "AuthenticationResult" in res:
            return self._to_bundle(res)

        # Handle first-login password change
        if res.get("ChallengeName") == "NEW_PASSWORD_REQUIRED":
            if not self.new_password:
                raise RuntimeError(
                    "Cognito returned NEW_PASSWORD_REQUIRED but no new_password was provided."
                )

            challenge_responses = {
                "USERNAME": self.username,
                "NEW_PASSWORD": self.new_password,
            }
            if sh:
                challenge_responses["SECRET_HASH"] = sh

            if self.use_admin_api:
                res2 = self._client.admin_respond_to_auth_challenge(
                    UserPoolId=self.user_pool_id,
                    ClientId=self.client_id,
                    ChallengeName="NEW_PASSWORD_REQUIRED",
                    ChallengeResponses=challenge_responses,
                    Session=res["Session"],
                )
            else:
                res2 = self._client.respond_to_auth_challenge(
                    ClientId=self.client_id,
                    ChallengeName="NEW_PASSWORD_REQUIRED",
                    ChallengeResponses=challenge_responses,
                    Session=res["Session"],
                )

            if "AuthenticationResult" not in res2:
                raise RuntimeError("Challenge response did not return AuthenticationResult.")
            # From now on, the permanent password is the new one
            self.password = self.new_password
            return self._to_bundle(res2)

        # Other challenges not expected for this flow
        raise RuntimeError(f"Unexpected challenge: {res.get('ChallengeName')}")

    def refresh(self, tokens: TokenBundle) -> TokenBundle:
        """Refresh access/id tokens. Falls back to authenticate() if no refresh token."""
        if not tokens.refresh_token:
            return self.authenticate()

        params: Dict[str, Any] = {
            "ClientId": self.client_id,
            "AuthFlow": "REFRESH_TOKEN_AUTH",
            "AuthParameters": {
                "REFRESH_TOKEN": tokens.refresh_token,
            },
        }
        # When app client has a secret, Cognito expects SECRET_HASH on refresh too
        sh = self._secret_hash(self.username)
        if sh:
            params["AuthParameters"]["SECRET_HASH"] = sh

        # Note: AWS expects initiate_auth for refresh (even if you used admin flow for login)
        res = self._client.initiate_auth(**params)
        # Cognito may omit RefreshToken in the refresh response => keep the old one
        return self._to_bundle(res, keep_refresh=tokens.refresh_token)

    def close(self) -> None:
        pass  # boto3 client doesn't need explicit close

    # ---------- small convenience for callers ----------

    @staticmethod
    def build_headers(tokens: TokenBundle, id_token_header_name: str = "X-ID-Token") -> Dict[str, str]:
        """Headers for your FastAPI adapter (Authorization + ID token header)."""
        return {
            "Authorization": f"Bearer {tokens.access_token}",
            id_token_header_name: tokens.id_token,
        }

class ServiceAuthSession:
    def __init__(self, provider: CognitoServiceAuth, leeway_sec: int = 60):
        self.provider = provider
        self.leeway = leeway_sec
        self._bundle: Optional[TokenBundle] = None

    def get_bundle(self) -> TokenBundle:
        if not self._bundle:
            self._bundle = self.provider.authenticate()
            return self._bundle

        # refresh if close to expiry
        now = time.time()
        if self._bundle.access_expires_at and now >= (self._bundle.access_expires_at - self.leeway):
            self._bundle = self.provider.refresh(self._bundle)
        return self._bundle