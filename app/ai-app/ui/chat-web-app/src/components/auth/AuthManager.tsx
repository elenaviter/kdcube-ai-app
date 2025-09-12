/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2025 Elena Viter
 */

// components/auth/AuthManager.tsx
import {createContext, ReactNode, useContext} from "react";
import OAuthManager, {Callback, SignedOutPage, WithOAuthRequired} from "./OAuthManager.tsx";
import {AuthContextProps, useAuth} from "react-oidc-context";
import {Route} from "react-router-dom";
import {
    getDefaultRoutePrefix,
    getExtraIdTokenHeaderName,
    getHardcodedAuthToken, getOAuthConfig
} from "../../AppConfig.ts";

interface User {
    name?: string;
    email?: string;
    roles?: string[];
    permissions?: string[];
    groups?: string[];
    username?: string;
    raw?: Record<string, any>;
}

export interface AuthContextValue {
    getUserProfile: () => User | undefined
    getUserAuthToken: () => string | undefined
    getUserIdToken: () => string | undefined
    getRoutes: (rootPrefix?: string) => ReactNode | ReactNode[] | null
    getAuthType: () => AuthType
    appendAuthHeader: (headers: [string, string][] | Headers) => [string, string][] | Headers
    logout: () => Promise<void> | void
}

const defaultContextValue = {
    getUserProfile: () => undefined,
    getUserAuthToken: () => undefined,
    getRoutes: () => null,
    getAuthType: () => "none" as AuthType,
    appendAuthHeader: (headers: [string, string][] | Headers) => headers,
    logout: async () => {
    }                         // <— ADD
} as AuthContextValue;

const AuthContext = createContext<AuthContextValue>(defaultContextValue);

export type AuthType = "none" | "oauth" | "hardcoded"

const AuthManager = ({children, authType}: {
    children: ReactNode | ReactNode[],
    authType: AuthType
}) => {
    switch (authType) {
        case "none":
            return (<AuthContext value={defaultContextValue}>
                {children}
            </AuthContext>);
        case "oauth":
            return (
                <OAuthManager>
                    <WithOAuth>
                        {children}
                    </WithOAuth>
                </OAuthManager>
            )
        case "hardcoded":
            return (<WithHardcodedAuth>
                {children}
            </WithHardcodedAuth>)
    }
    throw "Unknown auth type"

}

export const appendAuthHeader = (
    headers: [string, string][] | Headers,
    authOrToken?: AuthContextProps | string
) => {
    let accessToken: string | undefined;
    let idToken: string | undefined;

    if (typeof authOrToken === "string") {
        // hardcoded mode: only a single opaque token; no ID token
        accessToken = authOrToken;
    } else if (authOrToken) {
        accessToken = authOrToken.user?.access_token || undefined;
        idToken = authOrToken.user?.id_token || undefined;
    }

    // Always: Bearer = access_token (do NOT ever use id_token as bearer)
    if (accessToken) {
        if (headers instanceof Headers) {
            headers.set("Authorization", `Bearer ${accessToken}`);
        } else {
            headers.push(["Authorization", `Bearer ${accessToken}`]);
        }
    }

    // Always also send the ID token in a separate header when we have one
    if (idToken) {
        const idHdr = getExtraIdTokenHeaderName();
        if (headers instanceof Headers) {
            headers.set(idHdr, idToken);
        } else {
            headers.push([idHdr, idToken]);
        }
    }
    return headers;
};


const WithOAuth = ({children}: { children: ReactNode | ReactNode[] }) => {
    const auth = useAuth();

    return (
        <AuthContext
            value={{
                getUserProfile: () => {
                    const user = mapClaimsToUser(auth?.user?.profile);
                    // console.log(user, auth?.user)
                    return user;
                },
                getUserAuthToken: () => auth?.user?.access_token,
                getUserIdToken: () => auth?.user?.id_token,
                getRoutes: (rootPrefix?: string) => {
                    rootPrefix = rootPrefix ? `${rootPrefix}/` : ""
                    return [
                        <Route key="cb" path={`${rootPrefix}callback`} element={<Callback/>}/>,
                        <Route key="so" path={`${rootPrefix}signedout`}
                               element={<SignedOutPage/>}/>,
                    ]
                },
                getAuthType: () => "oauth" as AuthType,
                appendAuthHeader: (h) => appendAuthHeader(h, auth),

                // Minimal, Cognito-approved logout:
                // 1) Clear SPA tokens (removeUser)
                // 2) Hit Hosted UI /logout to clear the Cognito cookie
                // 3) Return to /signedout (user must click “Sign in” to start a new login)
                logout: async () => {
                    const base = getDefaultRoutePrefix(); // e.g. "/chatbot/domain-expert"
                    const origin = window.location.origin;
                    const logoutRedirect = `${origin}${base}/chat`;
                    const cfg = getOAuthConfig();
                    const clientId = (auth as any)?.settings?.client_id || (cfg as any).client_id || "";
                    await auth.signoutRedirect({
                        extraQueryParams: {
                            client_id: clientId,
                            logout_uri: logoutRedirect
                        }
                    })
                },
            }}
        >
            {children}
        </AuthContext>
    );
};
const WithHardcodedAuth = ({children}: { children: ReactNode | ReactNode[] }) => {
    return (
        <AuthContext value={{
            getUserProfile: () => undefined,
            getUserAuthToken: () => getHardcodedAuthToken(),
            getUserIdToken: () => null,
            getRoutes: () => null,
            getAuthType: () => "hardcoded" as AuthType,
            appendAuthHeader: (headers: [string, string][] | Headers) => appendAuthHeader(headers, getHardcodedAuthToken()),

            logout: () => {
                // nothing to revoke; just navigate to the signed-out screen
                const base = getDefaultRoutePrefix();
                window.location.href = `${window.location.origin}${base}/signedout`;
            }
        }}>
            {children}
        </AuthContext>
    )
}

const listify = (v: unknown): string[] => {
    if (!v) return [];
    if (Array.isArray(v)) return v.map(String).filter(Boolean);
    if (typeof v === "string") return v.split(",").map(s => s.trim()).filter(Boolean);
    return [String(v)];
};

const listifyNonEmpty = (v: unknown): string[] | null => {
    const arr = listify(v) ?? [];
    // normalize + drop empties
    const clean = Array.isArray(arr) ? arr.map(String).map(s => s.trim()).filter(Boolean) : [];
    return clean.length ? clean : null; // null lets ?? skip empties
};


const mapClaimsToUser = (claims?: Record<string, any>) => {
    if (!claims) return undefined;
    const roles =
        listifyNonEmpty(claims?.roles) ??
        listifyNonEmpty(claims?.['custom:roles']) ??
        listifyNonEmpty(claims?.['cognito:groups']) ??
        listifyNonEmpty(claims?.realm_access?.roles) ??
        [];
    const permissions =
        listify(claims.permissions) || listify(claims["custom:permissions"]);
    const groups = listify(claims["cognito:groups"]) || listify(claims.groups);

    const username =
        claims["cognito:username"] || claims.preferred_username || claims.username || claims.email;
    const user = {
        sub: claims.sub,
        username,
        name: claims.name || claims.given_name,
        email: claims.email,
        roles,
        permissions,
        groups,
        raw: claims
    }
    // console.log("USER", user);
    return user;
};

export const useAuthManagerContext = () => {
    return useContext(AuthContext)
}

export const WithAuthRequired = ({children}: { children: ReactNode | ReactNode[] }) => {
    const authContext = useAuthManagerContext()
    switch (authContext.getAuthType()) {
        case "none":
        case "hardcoded":
            return (<>{children}</>);
        case "oauth":
            return (<WithOAuthRequired>{children}</WithOAuthRequired>)
    }
    throw "Unknown auth type"
}

export const withAuthRequired = (children: ReactNode | ReactNode[]) => {
    return (<WithAuthRequired>{children}</WithAuthRequired>)
}


export default AuthManager