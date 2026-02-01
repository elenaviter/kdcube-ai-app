export type AuthType = "none" | "cognito" | "hardcoded"

export interface AppUser {
    name?: string;
    email?: string;
    roles?: string[];
    permissions?: string[];
    groups?: string[];
    username?: string;
    raw?: unknown;
}