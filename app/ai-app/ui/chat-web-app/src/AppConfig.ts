import {AuthType} from "./components/auth/AuthManager.tsx";

/*
    We may want to have an empty base for some values to make API calls relative to UI's base url
 */
function selectValue<T>(...args: T[]) {
    for (const arg of args) {
        if (arg === undefined || arg === null)
            continue
        return arg;
    }
    return null;
}

export const getKBAPIBaseAddress = (): string => {
    return selectValue(import.meta.env.CHAT_WEB_APP_KB_BASE, 'http://localhost:8000')
}

export const getKBSocketAddress = (): string => {
    return selectValue(import.meta.env.CHAT_WEB_APP_KB_SOCKET, 'http://localhost:8000')
}

export const getKBSocketSocketIOPath = (): string => {
    return selectValue(import.meta.env.CHAT_WEB_APP_KB_SOCKETIO_PATH, '/socket.io')
}

export const getChatBaseAddress = (): string => {
    return selectValue(import.meta.env.CHAT_WEB_APP_CHAT_BASE, 'http://localhost:8010')
}

export const getChatSocketAddress = (): string => {
    return selectValue(import.meta.env.CHAT_WEB_APP_CHAT_SOCKET, 'http://localhost:8010')
}

export const getChatSocketSocketIOPath = (): string => {
    return selectValue(import.meta.env.CHAT_WEB_APP_CHAT_SOCKETIO_PATH, '/socket.io')
}

export const getMonitoringBaseAddress = (): string => {
    return selectValue(import.meta.env.CHAT_WEB_APP_MONITORING_BASE, 'http://localhost:8010')
}

export const getAuthType = (): AuthType =>
    (import.meta.env.CHAT_WEB_APP_AUTH_TYPE || "oauth") as AuthType;

export const getExtraIdTokenHeaderName = (): string =>
    import.meta.env.CHAT_WEB_APP_EXTRA_ID_TOKEN_HEADER || "X-ID-Token";

export const getHardcodedAuthToken = (): string => {
    return import.meta.env.CHAT_WEB_APP_HARDCODED_AUTH_TOKEN
}

export const getCustomEmbedingEndpoint = (): string => {
    return import.meta.env.CHAT_WEB_APP_CUSTOM_EMBEDING_ENDPOINT || 'http://localhost:5005/serving/v1/embeddings'
}

export function getOAuthConfig() {
    const authority = import.meta.env.CHAT_WEB_APP_OIDC_AUTHORITY!; // EITHER hosted UI domain OR issuer+poolId
    const clientId  = import.meta.env.CHAT_WEB_APP_OIDC_CLIENT_ID!;
    const base      = getDefaultRoutePrefix(); // e.g. "/chatbot/chat"
    const origin    = window.location.origin;

    // These must exactly match what you configured in Cognito App Client (Callback URL(s) / Sign out URL(s))
    const redirect       = `${origin}${base}/callback`;
    const logoutRedirect = `${origin}${base}/chat`;

    // Optional sanity warning for common misconfig
    if (
        authority.includes("cognito-idp.") &&
        !authority.includes("amazoncognito.com") && // not the hosted UI form
        !/\/[A-Za-z0-9_-]+$/.test(authority)        // missing /<USER_POOL_ID>
    ) {
        console.warn(
            "[OIDC] For Cognito, 'authority' must be either the Hosted UI domain " +
            "(https://<domain>.auth.<region>.amazoncognito.com) or the issuer that includes the User Pool ID " +
            "(https://cognito-idp.<region>.amazonaws.com/<USER_POOL_ID>)."
        );
    }
    console.log("Redirect URI:", redirect);
    console.log("Logout URI:", logoutRedirect);
    return {
        authority,
        client_id: clientId,
        redirect_uri: redirect,
        post_logout_redirect_uri: logoutRedirect,
        response_type: "code",      // PKCE
        scope: (import.meta.env.CHAT_WEB_APP_OIDC_SCOPE ?? "openid email phone profile"), // no 'offline_access' for Cognito
        automaticSilentRenew: false, // Cognito + iframes = pain; use refresh tokens instead
        monitorSession: false,
        loadUserInfo: true,
    };
}

export const getWorkingScope = () => {
    return {
        project: import.meta.env.CHAT_WEB_APP_DEFAULT_PROJECT || 'default-project',
        tenant: import.meta.env.CHAT_WEB_APP_DEFAULT_TENANT || 'home',
    }
}

export const getDefaultRoutePrefix = (): string =>
    (import.meta.env.CHAT_WEB_APP_DEFAULT_ROUTE_PREFIX || "").replace(/\/+$/, ""); // no trailing slash


export function getCognitoHostedDomain(): string {
    // e.g. https://<your-domain>.auth.<region>.amazoncognito.com
    return "https://eu-west-1rulwzsq6h.auth.eu-west-1.amazoncognito.com";
}