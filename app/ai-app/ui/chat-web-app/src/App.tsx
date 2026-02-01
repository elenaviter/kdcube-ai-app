/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2025 Elena Viter
 */

import './App.css'
import {BrowserRouter as Router, Route, Routes} from "react-router-dom"
import NotFoundPage from "./components/notfound/NotFoundPage.tsx";
import {ReactNode, useCallback, useMemo} from "react";
import KnowledgeBasePage from "./components/kb/KnowledgeBasePage.tsx";
import {getChatPagePath, getDefaultRoutePrefix} from "./AppConfig.ts";
import {store} from "./app/store.ts";
import {Provider} from "react-redux";
import Dummy from "./components/chat/Dummy.tsx";
import AuthCallback from "./features/auth/AuthCallback.tsx";
import WithAuthRequired from "./features/auth/WithAuthRequired.tsx";
import ChatPage from "./components/chat/ChatPage.tsx";

const prefix = getDefaultRoutePrefix();


function AppRouter() {
    const withAuthRequired = useCallback((children: ReactNode | ReactNode[]) => {
        return <WithAuthRequired>{children}</WithAuthRequired>
    }, [])

    const chatPagePath = useMemo(()=>{
        return getChatPagePath();
    }, [])

    const chatPage = useMemo(() => {
        return withAuthRequired(<ChatPage/>)
    }, [withAuthRequired])

    return useMemo(() => {
        return <Provider store={store}>
            <Router>
                <Routes>
                    <Route path={`${prefix}/callback`} element={<AuthCallback/>}/>
                    <Route path={chatPagePath} element={chatPage}/>
                    <Route path={`${chatPagePath}/:conversationID`} element={chatPage}/>
                    <Route path={`${prefix}/kb`} element={withAuthRequired(<KnowledgeBasePage/>)}/>
                    <Route path={`${prefix}/dummy`} element={withAuthRequired(<Dummy/>)}/>
                    <Route path='*' element={<NotFoundPage/>}/>
                </Routes>
            </Router>
        </Provider>
    }, [chatPage, withAuthRequired, chatPagePath])
}

export default AppRouter
