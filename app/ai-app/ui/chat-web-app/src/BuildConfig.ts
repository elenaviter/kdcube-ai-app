/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2025 Elena Viter
 */

import {controlPlanePublicPath} from "./controlPlaneMount.ts";

function selectValue<T>(...args: T[]) {
    for (const arg of args) {
        if (arg === undefined || arg === null)
            continue
        return arg;
    }
    return null;
}

function selectBool(...args: (string | boolean | undefined | null)[]) {
    for (const arg of args) {
        if (arg === undefined || arg === null)
            continue
        if (typeof arg === "string") {
            return (arg as string).toLowerCase() === "true";
        }
        return Boolean(arg);
    }
    return false;
}

export const chatAPIBasePath = selectValue<string>(import.meta.env.CHAT_WEB_APP_CHAT_API_BASE_PATH, '') as string
export const configEndpoint = selectValue<string>(import.meta.env.CHAT_WEB_APP_CONFIG_ENDPOINT, "/api/cp-frontend-config") as string
export const configPath = selectValue<string>(import.meta.env.CHAT_WEB_APP_CONFIG_FILE_PATH, controlPlanePublicPath("config.json")) as string
export const showDebugControls = selectBool(import.meta.env.CHAT_WEB_APP_SHOW_DEBUG_CONTROLS, false)
