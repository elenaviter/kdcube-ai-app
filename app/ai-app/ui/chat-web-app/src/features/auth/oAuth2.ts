import {User, UserManager, UserManagerSettings} from "oidc-client-ts";
import {getOAuthConfig} from "../../AppConfig.ts";

interface UserManageCallbacks {
    onAccessTokenExpiring?: (ev: unknown) => void;
    onAccessTokenExpired?: (ev: unknown) => void;
    onUserLoaded?: (ev: User) => void;
    onUserUnloaded?: () => Promise<unknown>;
    onSilentRenewError?: (ev: unknown) => void;
    onUserSignedIn?: () => Promise<unknown>;
    onUserSignedOut?: () => Promise<unknown>;
    onUserSessionChanged?: () => Promise<unknown>;
}

export function createDefaultUserManager(callbacks?: UserManageCallbacks, settings?: UserManagerSettings) {
    settings = settings ?? getOAuthConfig();
    const mgr = new UserManager(settings);
    const events = mgr.events;

    events.addAccessTokenExpiring((ev) => {
        callbacks?.onAccessTokenExpiring?.(ev);
    });

    events.addAccessTokenExpired((ev) => {
        callbacks?.onAccessTokenExpired?.(ev);
    });

    events.addUserLoaded((ev) => {
        callbacks?.onUserLoaded?.(ev);
    });

    events.addUserUnloaded(async () => {
        await callbacks?.onUserUnloaded?.();
    });

    events.addSilentRenewError((ev) => {
        callbacks?.onSilentRenewError?.(ev);
    });

    events.addUserSignedIn(async () => {
        await callbacks?.onUserSignedIn?.();
    });

    events.addUserSignedOut(async () => {
        await callbacks?.onUserSignedOut?.();
    });

    events.addUserSessionChanged(async () => {
        await callbacks?.onUserSessionChanged?.();
    });

    return mgr;
}