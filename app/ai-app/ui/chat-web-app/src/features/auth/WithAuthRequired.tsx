import {useAppDispatch, useAppSelector} from "../../app/store.ts";
import {selectIsLoggedIn} from "./authSlice.ts";
import {useEffect} from "react";
import {logIn} from "./authMiddleware.ts";
import {WithReactChildren} from "../../types/common.ts";

type WithAuthRequiredProps = WithReactChildren

const WithAuthRequired = ({children}: WithAuthRequiredProps) => {
    const loggedIn = useAppSelector(selectIsLoggedIn)

    const dispatch = useAppDispatch();

    useEffect(() => {
        if (!loggedIn) {
            const path = typeof window !== "undefined" ? window.location.pathname : undefined;
            dispatch(logIn(path))
        }
    }, [loggedIn, dispatch]);

    if (!loggedIn) return null;

    return (
        <>{children}</>
    )
}

export default WithAuthRequired;