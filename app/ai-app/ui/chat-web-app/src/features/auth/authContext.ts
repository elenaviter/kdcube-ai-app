import {createContext, ReactNode} from "react";

export interface AuthManager2ContextValue {
    logIn: (navigateTo?:string) => void;
    logOut: () => void;
    getRoutes: (rootPrefix?: string) => ReactNode | ReactNode[]
}

const AuthManager2Context = createContext<AuthManager2ContextValue>({
    logIn: (_unused_navigateTo?:string) => {
        throw "Not implemented"
    },
    logOut: () => {
        throw "Not implemented"
    },
    getRoutes: ()=> {
        throw "Not implemented"
    }
})

export default AuthManager2Context;