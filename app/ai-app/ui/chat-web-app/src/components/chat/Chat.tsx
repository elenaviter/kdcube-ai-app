/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2025 Elena Viter
 */

// Chat.tsx
import React, {useCallback, useEffect, useMemo, useRef, useState} from "react";
import {Bot, Loader, LogOut, Wifi, WifiOff} from "lucide-react";

import ChatInterface from "./ChatInterface/ChatInterface.tsx";
import {useAppDispatch, useAppSelector} from "../../app/store.ts";
import {useGetSuggestedQuestionsQuery} from "../../features/suggestedQuestions/suggestedQuestions.ts";
import {
    selectChatConnected,
    selectChatStayConnected,
    selectProject,
    selectTenant
} from "../../features/chat/chatStateSlice.ts";
import {logOut} from "../../features/auth/authMiddleware.ts";
import ChatSidePanel from "../../features/chatSidePanel/ChatSidePanel.tsx";

// -----------------------------------------------------------------------------
// Helper: KB search results wrapper
// -----------------------------------------------------------------------------
// const UpdatedSearchResultsHistory = ({searchHistory, onClose, kbEndpoint}: {
//     searchHistory: any[];
//     onClose: () => void;
//     kbEndpoint: string;
// }) => {
//     return (
//         <EnhancedKBSearchResults
//             searchResults={searchHistory}
//             onClose={onClose}
//             kbEndpoint={kbEndpoint}
//         />
//     );
// };

const SingleChatApp: React.FC = () => {
    const dispatch = useAppDispatch()
    const tenant = useAppSelector(selectTenant);
    const project = useAppSelector(selectProject);
    const stayConnected = useAppSelector(selectChatStayConnected)
    const connected = useAppSelector(selectChatConnected)
    // const userRoles = useAppSelector(selectRoles)

    //RTK
    const {data: suggestedQuestions, isFetching: updatingQuestions} = useGetSuggestedQuestionsQuery({tenant, project});


    // const [showKB, setShowKB] = useState<boolean>(false);
    // const [showKbResults, setShowKbResults] = useState<boolean>(false);

    // const [kbSearchHistory, setKbSearchHistory] = useState<any[]>([]);
    // const [newKbSearchCount, setNewKbSearchCount] = useState<number>(0);

    // KB helpers
    // const handleKbSearchResults = useCallback((searchResponse: any, isAutomatic: boolean = true) => {
    //     const enrichedResponse = {
    //         ...searchResponse,
    //         searchType: isAutomatic ? 'automatic' : 'manual',
    //         timestamp: new Date()
    //     };
    //     setKbSearchHistory(prev => [enrichedResponse, ...prev.slice(0, 9)]);
    //     setNewKbSearchCount(prev => prev + 1);
    //     setTimeout(() => setNewKbSearchCount(0), 5000);
    // }, []);
    // const handleShowKbResults = useCallback(() => {
    //     setShowKbResults(true);
    //     setNewKbSearchCount(0);
    // }, []);
    // const handleCloseKbResults = useCallback(() => setShowKbResults(false), []);


    const connectionStatus = useMemo(() => {
        if (stayConnected && !connected) return {
            icon: <Loader size={14} className="animate-spin"/>,
            text: 'Connecting...',
            color: 'text-yellow-600 bg-yellow-50'
        };
        if (connected) return {icon: <Wifi size={14}/>, text: 'Connected', color: 'text-green-600 bg-green-50'};
        return {icon: <WifiOff size={14}/>, text: 'Disconnected', color: 'text-red-600 bg-red-50'};
    }, [stayConnected, connected]);

    // Logout
    const handleLogout = useCallback(() => {
        dispatch(logOut())
    }, [dispatch]);

    // const hideKB = () => setShowKB(false);


    const renderFullHeader = () => {
        return (
            <div className="bg-white border-b border-gray-200 px-6 py-4">
                <div className="flex items-center justify-between">
                    <div className="flex items-center">
                        <div
                            className="w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg mr-3 flex items-center justify-center">

                            <Bot size={20} className="text-white"/>
                        </div>
                        <div>
                            <h1 className="text-xl font-semibold text-gray-900">
                                AI Assistant
                            </h1>
                        </div>
                    </div>

                    <div className="flex items-center gap-2">
                        {/* Connection status pill */}
                        <div className={`flex items-center px-3 py-1 rounded-lg text-sm ${connectionStatus.color}`}>
                            {connectionStatus.icon}
                            <span className="ml-2 font-medium">{connectionStatus.text}</span>
                        </div>

                        {/*<button*/}
                        {/*    onClick={() => setShowKB(!showKB)}*/}
                        {/*    className="relative flex items-center px-3 py-2 rounded-lg bg-gray-100 text-gray-600 hover:bg-gray-200"*/}
                        {/*    title="View KB"*/}
                        {/*>*/}
                        {/*    <Database size={16} className="mr-1"/><span className="text-sm">KB</span>*/}
                        {/*</button>*/}

                        {/*<button*/}
                        {/*    onClick={handleShowKbResults}*/}
                        {/*    className={`relative flex items-center px-3 py-2 rounded-lg transition-colors ${*/}
                        {/*        kbSearchHistory.length > 0 ? 'bg-blue-100 text-blue-700 hover:bg-blue-200' : 'bg-gray-100 text-gray-600 hover:bg-gray-200'*/}
                        {/*    }`}*/}
                        {/*    title="View KB Search Results"*/}
                        {/*>*/}
                        {/*    <Search size={16} className="mr-1"/>*/}
                        {/*    <span className="text-sm">KB Search</span>*/}
                        {/*    {kbSearchHistory.length > 0 && (*/}
                        {/*        <span*/}
                        {/*            className="ml-1 text-xs bg-blue-200 text-blue-800 px-1 rounded">{kbSearchHistory.length}</span>*/}
                        {/*    )}*/}
                        {/*    {newKbSearchCount > 0 && (*/}
                        {/*        <span*/}
                        {/*            className="absolute -top-1 -right-1 w-2 h-2 bg-red-500 rounded-full animate-pulse"/>*/}
                        {/*    )}*/}
                        {/*</button>*/}

                        <button
                            onClick={handleLogout}
                            className="flex items-center px-3 py-2 bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-lg"
                            title="Sign out"
                        >
                            <LogOut size={16} className="mr-1"/><span className="text-sm">Logout</span>
                        </button>
                    </div>
                </div>
            </div>
        )
    }

    const chatContainerRef = useRef<HTMLDivElement>(null)
    const [fullChatWidth, setFullChatWidth] = useState<number>(0)


    useEffect(() => {
        function handleResize() {
            if (!chatContainerRef.current)
                return;
            const width = chatContainerRef.current.clientWidth;
            setFullChatWidth(width)
        }

        window.addEventListener('resize', handleResize);
        handleResize();

        return () => window.removeEventListener('resize', handleResize);
    }, []);


    return (
        <div id={SingleChatApp.name} className="flex h-screen bg-slate-100">
            {/* Main Column */}
            <div className="flex-1 flex flex-col">
                {/* Header */}
                {/*{renderSimpleHeader()}*/}
                {renderFullHeader()}

                {/* Body: Chat + optionally Steps / KB Results / System Monitor */}
                <div className={`flex-1 flex overflow-hidden transition-all duration-300`}>
                    <ChatSidePanel/>
                    {/* Chat Column */}
                    <div className={`flex-1 flex flex-col`} ref={chatContainerRef}>
                        {/* Quick Questions */}
                        <div className="px-6 py-4 bg-gray-50 border-b border-gray-200">
                            {updatingQuestions ?
                                (<div className="w-full flex">
                                    <Loader size={28} className='animate-spin text-gray-300 mx-auto'/>
                                </div>) :
                                (<>
                                    <h4 className="text-sm font-medium text-gray-700 mb-2">Try these questions:</h4>
                                    <div className="flex flex-wrap gap-2">
                                        {/*{quickQuestions && quickQuestions.map((q, idx) => (*/}
                                        {/*    <button key={idx} onClick={() => sendMessage(q)}*/}
                                        {/*            disabled={isProcessing || !isSocketConnected}*/}
                                        {/*            className="px-3 py-1 text-xs bg-white text-gray-700 border border-gray-200 rounded-full hover:bg-gray-50 hover:border-gray-300 disabled:opacity-50">*/}
                                        {/*        {q}*/}
                                        {/*    </button>*/}
                                        {/*))}*/}

                                    </div>
                                </>)
                            }
                        </div>

                        <ChatInterface maxWidth={fullChatWidth * (3 / 5)}/>
                    </div>

                    {/* KB Search Results Panel */}
                    {/*{showKbResults && (*/}
                    {/*    <div className="border-l border-gray-200 bg-white relative" style={{width: `700px`}}>*/}
                    {/*        /!* simple draggable bar *!/*/}
                    {/*        <div*/}
                    {/*            className="absolute left-0 top-0 bottom-0 w-1 cursor-col-resize hover:bg-blue-300 group">*/}
                    {/*            <div*/}
                    {/*                className="absolute left-0 top-1/2 transform -translate-y-1/2 -translate-x-1 opacity-0 group-hover:opacity-100">*/}
                    {/*                <GripVertical size={16} className="text-gray-400"/>*/}
                    {/*            </div>*/}
                    {/*        </div>*/}
                    {/*        {kbSearchHistory.length > 0 ? (*/}
                    {/*            <UpdatedSearchResultsHistory*/}
                    {/*                searchHistory={kbSearchHistory}*/}
                    {/*                onClose={handleCloseKbResults}*/}
                    {/*                kbEndpoint={config.kb_search_endpoint || `${getKBAPIBaseAddress()}/api/kb`}*/}
                    {/*            />*/}
                    {/*        ) : (*/}
                    {/*            <div className="h-full flex flex-col">*/}
                    {/*                <div*/}
                    {/*                    className="px-4 py-3 border-b border-gray-200 bg-gray-50 flex items-center justify-between">*/}
                    {/*                    <h3 className="font-semibold text-gray-900 text-sm">KB Search Results</h3>*/}
                    {/*                    <button onClick={handleCloseKbResults}*/}
                    {/*                            className="p-1 hover:bg-gray-200 rounded text-gray-500 hover:text-gray-700">*/}
                    {/*                        <X size={14}/>*/}
                    {/*                    </button>*/}
                    {/*                </div>*/}
                    {/*                <div className="flex-1 flex items-center justify-center text-gray-500">*/}
                    {/*                    <div className="text-center">*/}
                    {/*                        <Database size={24} className="mx-auto mb-2 opacity-50"/>*/}
                    {/*                        <p>No KB search results yet</p>*/}
                    {/*                        <p className="text-xs mt-1">Results will appear here when RAG retrieval*/}
                    {/*                            occurs</p>*/}
                    {/*                    </div>*/}
                    {/*                </div>*/}
                    {/*            </div>*/}
                    {/*        )}*/}
                    {/*    </div>*/}
                    {/*)}*/}
                </div>
            </div>

            {/* KB Side Panel */}
            {/*{showKB && (*/}
            {/*    <div className="fixed inset-0 z-50 flex">*/}
            {/*        <div className="absolute inset-0 bg-transparent backdrop-blur-xs" onClick={hideKB}/>*/}
            {/*        <div className="ml-auto transition-transform h-full w-1/2">*/}
            {/*            <KBPanel onClose={hideKB}/>*/}
            {/*        </div>*/}
            {/*    </div>*/}
            {/*)}*/}
        </div>
    );
};

export default SingleChatApp;
