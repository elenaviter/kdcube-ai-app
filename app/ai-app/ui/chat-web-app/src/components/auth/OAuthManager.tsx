// components/auth/OAuthManager.tsx
import {AuthProvider, useAuth} from "react-oidc-context";
import {ReactNode, useEffect, useRef} from "react";

import {Loader2} from "lucide-react";
import {getDefaultRoutePrefix, getOAuthConfig} from "../../AppConfig.ts";
import {useRefreshToken} from "./useRefreshToken.ts";

export const LoadingPage = () => (
    <div className="absolute inset-0 bg-white flex items-center justify-center z-50">
        <Loader2 className="h-8 w-8 animate-spin text-gray-600" />
    </div>
);

export function SignedOutPage() {
    const auth = useAuth();

    const onSignInClick = async () => {
        try {
            // make sure any cached user is gone
            await auth.removeUser();
        } catch {}
        await auth.signinRedirect({
            state: JSON.stringify({ navigate_to: `${getDefaultRoutePrefix()}/chat` }),
        });
    };

    return (
        <div className="min-h-screen bg-gray-50 flex items-center justify-center">
            <div className="max-w-md w-full bg-white rounded-lg shadow p-8 text-center space-y-4">
                <h1 className="text-xl font-semibold">You’ve signed out</h1>
                <button onClick={onSignInClick} className="px-4 py-2 rounded bg-blue-600 text-white">
                    Sign in
                </button>
            </div>
        </div>
    );
}


export const WithOAuthRequired = ({ children }) => {
    const auth = useAuth();
    const kickedOff = useRef(false);
    const base = getDefaultRoutePrefix();

    const path = typeof window !== "undefined" ? window.location.pathname : "";
    const isAuthRoute = path === `${base}/callback` || path === `${base}/signedout`;

    useEffect(() => {
        if (!isAuthRoute && !auth.isLoading && !auth.isAuthenticated && !auth.activeNavigator && !kickedOff.current) {
            kickedOff.current = true;
            auth
                .signinRedirect({
                    state: JSON.stringify({ navigate_to: path || "/" }),
                })
                .catch((e) => console.error("signinRedirect failed", e));
        }
    }, [auth.isLoading, auth.isAuthenticated, auth.activeNavigator, isAuthRoute, path]);

    if (auth.isLoading || auth.activeNavigator === "signinRedirect") return <LoadingPage />;
    if (auth.error) return <div>An error occurred. Please reload this page.</div>;
    if (!auth.isAuthenticated) return <LoadingPage />;
    return <>{children}</>;
};

export const Callback = () => <LoadingPage />;

export const DebugUserProfile = () => {
    const auth = useAuth()
    return (
        <div>
            <h1>Profile Page</h1>
            <pre>{JSON.stringify(auth.user?.profile, null, 2)}</pre>
        </div>
    )
}

const TokenRefresher = () => { useRefreshToken(); return null; };

const OAuthManager = ({ children }: { children: ReactNode }) => {
    return (
        <AuthProvider
            {...getOAuthConfig()}
            onSigninCallback={(user) => {
                const base = getDefaultRoutePrefix();
                let navigateTo = `${base}/chat`;
                try {
                    const raw = (user as any)?.state;
                    const parsed = typeof raw === "string" ? JSON.parse(raw) : raw;
                    const target = parsed?.navigate_to;
                    if (typeof target === "string") {
                        navigateTo = target.startsWith("/") ? target : `${base}/${target.replace(/^\//, "")}`;
                    }
                } catch {}
                // Force a real navigation so Router renders the target page immediately
                window.location.replace(navigateTo);
            }}
        >
            <TokenRefresher />
            {children}
        </AuthProvider>
    );
};

export default OAuthManager;