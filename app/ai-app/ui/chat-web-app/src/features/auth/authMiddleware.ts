import {Middleware, UnknownAction} from "@reduxjs/toolkit";
import {AuthType} from "./authTypes.ts";
import CognitoAuth from "./cogitoAuth.ts";

export const LOG_IN = "auth/LogIn"

interface LogInPayload {
    navigateTo?: string | URL | null;
}

interface LogInAction extends UnknownAction {
    type: typeof LOG_IN;
    payload?: LogInPayload | null;
}

export const logIn = (navigateTo?: string | URL | null): LogInAction => {
    return {
        type: LOG_IN,
        payload: navigateTo ? {navigateTo} : null,
    }
}

export const LOG_OUT = "auth/LogOut"

interface LogOutPayload {
    navigateTo?: string | URL | null;
}

interface LogOutAction extends UnknownAction {
    type: typeof LOG_OUT;
    payload?: LogOutPayload | null;
}

export const logOut = (navigateTo?: string | URL | null): LogOutAction => {
    return {
        type: LOG_OUT,
        payload: navigateTo ? {navigateTo} : null,
    }
}

export const LOG_IN_CALLBACK = "auth/LogInCallback"

interface LogInCallbackPayload {
    navigateTo?: string | URL | null;

}

interface LogInCallbackAction extends UnknownAction {
    type: typeof LOG_IN_CALLBACK;
    payload?: LogInCallbackPayload | null;
}

export const logInCallback = (navigateTo?: string | URL | null): LogInCallbackAction => {
    return {
        type: LOG_IN_CALLBACK,
        payload: navigateTo ? {navigateTo} : null
    }
}

export type AuthActions = LogInAction | LogOutAction | LogInCallbackAction

export interface AuthMiddlewareProvider {
    getMiddleware: () => Middleware
}

export const authMiddleware = (authType: AuthType): Middleware => {
    let middlewareProvider: AuthMiddlewareProvider

    switch (authType) {
        case "none":
        case "hardcoded":
            return () => (next) => (action) => next(action)
        case "cognito":
            middlewareProvider = new CognitoAuth()
            return middlewareProvider.getMiddleware()
        default:
            throw new Error("Unknown auth type");
    }
}