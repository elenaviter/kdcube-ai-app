/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2025 Elena Viter
 */

import './App.css'
import {BrowserRouter as Router, Routes, Route} from "react-router-dom"
import NotFoundPage from "./components/notfound/NotFoundPage.tsx";
import ChatPage from "./components/chat/ChatPage.tsx";
import * as React from "react";
import KnowledgeBasePage from "./components/kb/KnowledgeBasePage.tsx";
import SystemMonitor from "./components/monitoring/monitoring.tsx";
import AuthManager, {useAuthManagerContext, withAuthRequired} from "./components/auth/AuthManager.tsx";
import {getAuthType, getDefaultRoutePrefix} from "./AppConfig.ts";

const App: React.FC = () => {

    return (
        <AuthManager authType={getAuthType()}>
            <AppRouterRoutes/>
        </AuthManager>
    )
}

const prefix = getDefaultRoutePrefix();

const AppRouterRoutes = () => {
    const authContext = useAuthManagerContext();
    return (
        <Routes>
            {/*get additional routes (if any) for current auth type*/}
            {authContext.getRoutes()}

            <Route path={`${prefix}/chat`} element={withAuthRequired(<ChatPage/>)}/>
            <Route path={`${prefix}/kb`}  element={withAuthRequired(<KnowledgeBasePage/>)}/>
            <Route path={`${prefix}/monitoring`}  element={withAuthRequired(<SystemMonitor/>)}/>
            <Route path='*' element={<NotFoundPage/>}/>

        </Routes>
    )
}

function AppRouter() {
    return (
        <Router>
            <App/>
        </Router>
    )
}

export default AppRouter
