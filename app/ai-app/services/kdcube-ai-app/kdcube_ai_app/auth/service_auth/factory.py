# service_auth/factory.py
from typing import Dict, Any
from kdcube_ai_app.auth.service_auth.base import IdpConfig, ServiceIdP
from kdcube_ai_app.auth.service_auth.providers.cognito import CognitoServiceAuth

def create_service_idp(cfg: IdpConfig) -> ServiceIdP:
    provider = cfg.provider
    kw = cfg.kwargs or {}

    if provider == "cognito":
        return CognitoServiceAuth(**kw)

    # future: plug more providers here
    # if provider == "auth0": return Auth0ServiceAuth(**kw)
    # if provider == "keycloak": return KeycloakServiceAuth(**kw)
    # if provider == "azuread": return AzureADServiceAuth(**kw)

    raise ValueError(f"Unsupported IdP provider: {provider}")
