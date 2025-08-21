/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2025 Elena Viter
 */

// components/auth/useRefreshToken.ts
import { useEffect, useRef } from "react";
import { useAuth } from "react-oidc-context";
import { User } from "oidc-client-ts";

type TokenResponse = {
    access_token: string;
    id_token?: string;
    refresh_token?: string;
    token_type: string;
    expires_in: number;
    scope?: string;
};

export function useRefreshToken() {
    const auth = useAuth();
    const timerRef = useRef<number | null>(null);

    useEffect(() => {
        if (!auth.isAuthenticated || !auth.user) {
            clearTimer();
            return;
        }

        schedule();
        return clearTimer;

        function clearTimer() {
            if (timerRef.current !== null) {
                window.clearTimeout(timerRef.current);
                timerRef.current = null;
            }
        }

        function schedule() {
            clearTimer();
            const expiresAtMs = auth.user!.expires_at * 1000; // seconds→ms
            const now = Date.now();
            const refreshAt = Math.max(now + 5000, expiresAtMs - 60000); // 60s before expiry

            timerRef.current = window.setTimeout(async () => {
                try {
                    await refreshNow();
                } catch (e) {
                    console.warn("Token refresh failed, redirecting to login", e);
                    auth.signinRedirect({
                        state: JSON.stringify({ navigate_to: window.location.pathname }),
                    });
                    return;
                }
                schedule(); // schedule next cycle
            }, refreshAt - now);
        }

        async function refreshNow() {
            const user = auth.user!;
            const settings: any = (auth as any).settings || {};
            const authority: string = settings.authority!;
            const clientId: string = settings.client_id!;
            const scope: string | undefined = settings.scope;
            const metadataUrl =
                settings.metadataUrl || `${authority}/.well-known/openid-configuration`;

            if (!user.refresh_token) {
                // No refresh token → force a clean re-login before expiry
                throw new Error("No refresh_token present");
            }

            const metadata = await fetch(metadataUrl).then((r) => r.json());
            const tokenEndpoint: string = metadata.token_endpoint;

            const form = new URLSearchParams();
            form.set("client_id", clientId);
            form.set("grant_type", "refresh_token");
            form.set("refresh_token", user.refresh_token);
            if (scope) form.set("scope", scope);

            const res = await fetch(tokenEndpoint, {
                method: "POST",
                headers: { "Content-Type": "application/x-www-form-urlencoded" },
                body: form.toString(),
            });

            if (!res.ok) {
                const text = await res.text();
                throw new Error(`Refresh token exchange failed: ${text}`);
            }

            const tr: TokenResponse = await res.json();
            const nowSec = Math.floor(Date.now() / 1000);

            const nextUser = new User({
                ...user,
                access_token: tr.access_token,
                id_token: tr.id_token ?? user.id_token,
                refresh_token: tr.refresh_token ?? user.refresh_token, // rotation supported
                token_type: tr.token_type ?? "Bearer",
                scope: tr.scope ?? user.scope,
                expires_at: nowSec + (tr.expires_in ?? 3600),
            });

            // Prefer official updater if present
            if (typeof (auth as any).updateUser === "function") {
                (auth as any).updateUser(nextUser);
                return;
            }

            // Fallback: notify context + best-effort persist to default storage key
            if (auth.events && typeof auth.events.load === "function") {
                auth.events.load(nextUser);
                try {
                    const storageKey = `oidc.user:${authority}:${clientId}`;
                    window.localStorage.setItem(storageKey, JSON.stringify(nextUser));
                } catch {
                    /* ignore storage errors */
                }
            }
        }
    }, [auth.isAuthenticated, auth.user, auth.events, auth]);
}
