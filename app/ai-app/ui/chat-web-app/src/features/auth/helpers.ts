import {store} from "../../app/store.ts";
import {User} from "oidc-client-ts";
import {AppUser} from "./authTypes.ts";
import {makeSerializable} from "../../utils/utils.ts";

export function getDefaultAuthToken(): string | null | undefined {
    return store.getState().auth.authToken;
}

export function getDefaultIdToken(): string | null | undefined {
    return store.getState().auth.idToken;
}

export function oAuthUserToAppUser(user: User): AppUser {
    return {
        name: user.profile.name,
        email: user.profile.email,
        username: user.profile.preferred_username ?? (user.profile.username as string),
        // roles?: string[];
        // permissions?: string[];
        // groups?: string[];
        raw: makeSerializable(user)
    }
}