import {Middleware, UnknownAction} from "@reduxjs/toolkit";
import {User, UserManager} from "oidc-client-ts";
import {AuthActions, AuthMiddlewareProvider, LOG_IN, LOG_IN_CALLBACK, LOG_OUT} from "./authMiddleware.ts";
import {AppStore} from "../../app/store.ts";
import {finishLoading, setCredentials, setLoggedOut, startLoading} from "./authSlice.ts";
import {oAuthUserToAppUser} from "./helpers.ts";
import {createDefaultUserManager} from "./oAuth2.ts";

class CognitoAuth implements AuthMiddlewareProvider {
    private userManager: UserManager | null = null;

    getMiddleware(): Middleware {
        return ((store: AppStore) => (next: (action: AuthActions) => unknown) => (action: AuthActions) => {
            const setUser = (user: User) => {
                store.dispatch(setCredentials({
                    user: oAuthUserToAppUser(user),
                    authToken: user!.access_token,
                    idToken: user!.id_token,
                    loggedIn: true,
                }))
            }

            const getUserManager = (store: AppStore,): UserManager => {
                if (!this.userManager) {
                    this.userManager = createDefaultUserManager({
                            onUserSignedIn: async () => {
                                console.debug("onUserSignedIn")
                            },
                            onUserLoaded: (user) => {
                                console.debug("onUserLoaded")
                                if (user) {
                                    setUser(user)
                                }
                            },
                            onUserSessionChanged: async () => {
                                console.debug("onUserSessionChanged")
                            },
                            onAccessTokenExpiring: async () => {
                                try {
                                    await this.userManager!.signinSilent()
                                } catch (e) {
                                    console.error(e)
                                    await this.userManager!.removeUser()
                                    store.dispatch(setLoggedOut())
                                }
                            }
                        }
                    )
                }
                return this.userManager
            }

            const actionHandlers = async (store: AppStore, action: UnknownAction) => {
                const userManager = getUserManager(store)

                const checkLogin = async () => {
                    const user = await userManager!.getUser()
                    if (user) {
                        if (user.expired) {
                            try {
                                await userManager!.signinSilent()
                            } catch (e) {
                                console.error(e)
                                await userManager!.removeUser()
                                await userManager!.clearStaleState()
                                await userManager!.signinRedirect({
                                    state: action.payload ? JSON.stringify(action.payload) : undefined,
                                })
                            }
                        } else {
                            setUser(user)
                        }
                    } else {
                        await userManager!.clearStaleState()
                        await userManager!.signinRedirect({
                            state: action.payload ? JSON.stringify(action.payload) : undefined,
                        })
                    }
                    return user
                }

                switch (action.type) {
                    case LOG_IN: {
                        await checkLogin()
                        break
                    }
                    case LOG_OUT: {
                        await userManager!.signoutRedirect()
                        break
                    }
                    case LOG_IN_CALLBACK: {
                        if (store.getState().auth.loading) {
                            break;
                        }
                        store.dispatch(startLoading())
                        const user = await userManager!.getUser() ?? await userManager.signinCallback()
                        if (user) {
                            const userState = user.state ? JSON.parse(user.state as string) : null
                            const navigateTo = userState && userState.navigateTo ? userState.navigateTo : "/"
                            store.dispatch(finishLoading(navigateTo))
                        } else {
                            console.error("wtf") //todo: handle this
                        }
                        break;
                    }
                }
            }

            next(action);
            actionHandlers(store as AppStore, action as AuthActions).catch(console.error);
        }) as Middleware
    }
}

export default CognitoAuth