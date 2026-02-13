/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2025 Elena Viter
 */

// Chat.tsx
import React, {useCallback, useEffect, useMemo, useRef, useState} from "react";
import {Loader} from "lucide-react";

import ChatInterface from "./ChatInterface/ChatInterface.tsx";
import {useAppSelector} from "../../app/store.ts";
import {useGetSuggestedQuestionsQuery} from "../../features/suggestedQuestions/suggestedQuestions.ts";
import {
    selectConversationId,
    selectCurrentTurn,
    selectProject,
    selectTenant
} from "../../features/chat/chatStateSlice.ts";
import ChatSidePanel from "../../features/chatSidePanel/ChatSidePanel.tsx";
import ChatHeader from "./ChatHeader.tsx";
import AnimatedExpander from "../AnimatedExpander.tsx";
import ChatCanvas from "../../features/canvas/ChatCanvas.tsx";
import {CanvasItemLink, ChatCanvasContext, ChatCanvasContextValue} from "../../features/canvas/canvasContext.tsx";
import {getChatBaseAddress, getExtraIdTokenHeaderName} from "../../AppConfig.ts";
import {selectAuthToken, selectIdToken} from "../../features/auth/authSlice.ts";
import {
    addCanvasItemExtension,
    getCanvasArtifactTypes,
    getCanvasItemLinkGenerator
} from "../../features/extensions/canvasExtensions.tsx";
import {CodeExecArtifactType} from "../../features/logExtensions/codeExec/types.ts";
import CodeExecCanvasItem from "../../features/logExtensions/codeExec/CodeExecCanvasItem.tsx";
import {getCodeExecArtifactLink, matchesCodeExecArtifact} from "../../features/logExtensions/codeExec/utils.ts";
import {CanvasArtifactType} from "../../features/logExtensions/canvas/types.ts";
import CanvasItem from "../../features/logExtensions/canvas/CanvasItem.tsx";
import {getCanvasArtifactLink, matchesCanvasArtifact} from "../../features/logExtensions/canvas/utils.ts";
import {WebSearchArtifactType} from "../../features/logExtensions/webSearch/types.ts";
import WebSearchCanvasItem from "../../features/logExtensions/webSearch/WebSearchCanvasItem.tsx";
import {getWebSearchArtifactLink, matchesWebSearchArtifact} from "../../features/logExtensions/webSearch/utils.ts";
import {addChatLogExtension} from "../../features/extensions/logExtesnions.ts";
import {CanvasLogItem} from "../../features/logExtensions/canvas/CanvasLogItem.tsx";
import CodeExecLogItem from "../../features/logExtensions/codeExec/CodeExecLogItem.tsx";
import WebSearchLogItem from "../../features/logExtensions/webSearch/WebSearchLogItem.tsx";
import {TimelineTextArtifactType} from "../../features/logExtensions/timelineText/types.ts";
import TimelineTextLogItem from "../../features/logExtensions/timelineText/TimelineTextLogItem.tsx";

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

//chat log extensions
addChatLogExtension(CanvasArtifactType, CanvasLogItem)
addChatLogExtension(CodeExecArtifactType, CodeExecLogItem)
addChatLogExtension(WebSearchArtifactType, WebSearchLogItem)
addChatLogExtension(TimelineTextArtifactType, TimelineTextLogItem)

// canvas extension
addCanvasItemExtension(CanvasArtifactType, CanvasItem, getCanvasArtifactLink, matchesCanvasArtifact)
addCanvasItemExtension(WebSearchArtifactType, WebSearchCanvasItem, getWebSearchArtifactLink, matchesWebSearchArtifact)

const SingleChatApp: React.FC = () => {

    const tenant = useAppSelector(selectTenant);
    const project = useAppSelector(selectProject);
    const currentTurn = useAppSelector(selectCurrentTurn);
    const authToken = useAppSelector(selectAuthToken)
    const idToken = useAppSelector(selectIdToken)
    const conversationId = useAppSelector(selectConversationId);

    //RTK
    const {data: suggestedQuestions, isFetching: updatingQuestions} = useGetSuggestedQuestionsQuery({tenant, project});

    const chatCanvasRef = useRef<HTMLDivElement>(null);


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
    // const hideKB = () => setShowKB(false);

    const [canvasItemLink, setCanvasItemLink] = useState<CanvasItemLink | null>(null);
    const [overrideCanvasItemLink, setOverrideCanvasItemLink] = useState<boolean>(false);

    const lastCanvasItem = useMemo(() => {
        if (currentTurn == null) return null;
        const canvasArtifactTypes = getCanvasArtifactTypes()
        const canvasArtifacts = currentTurn.artifacts.filter(artifact => {
            return canvasArtifactTypes.includes(artifact.artifactType);
        })
        return canvasArtifacts.length > 0 ? canvasArtifacts[0] : null;
    }, [currentTurn])

    useEffect(() => {
        setCanvasItemLink(null);
    }, [conversationId]);

    useEffect(() => {
        if (currentTurn) {
            if (!overrideCanvasItemLink && lastCanvasItem) {
                setCanvasItemLink(getCanvasItemLinkGenerator(lastCanvasItem.artifactType)(lastCanvasItem))
            }
        } else {
            setOverrideCanvasItemLink(false)
        }

    }, [canvasItemLink, currentTurn, lastCanvasItem, overrideCanvasItemLink]);

    const showItem = useCallback((link: CanvasItemLink | null) => {
        if (currentTurn) {
            setOverrideCanvasItemLink(true);
        }
        setCanvasItemLink(link);
    }, [currentTurn])

    const chatCanvasContextValue = useMemo<ChatCanvasContextValue>(() => {
        return {
            showItem,
            itemLink: canvasItemLink
        }
    }, [canvasItemLink, showItem])

    useEffect(() => {
        const onIFrameRequest = (event: MessageEvent) => {
            if (event.data?.type === 'CONFIG_REQUEST') {
                console.debug(`[onIFrameRequest] CONFIG_REQUEST received`, event);

                const requestedFields = event.data.data?.requestedFields;
                const identity = event.data.data?.identity;

                if (!requestedFields || !Array.isArray(requestedFields)) {
                    return;
                }
                const baseUrl = getChatBaseAddress() || window.location.origin;
                const configMap: Record<string, () => unknown> = {
                    'baseUrl': () => baseUrl,
                    'accessToken': () => authToken,
                    'idToken': () => idToken,
                    'idTokenHeader': () => getExtraIdTokenHeaderName(),
                    'defaultTenant': () => tenant,
                    'defaultProject': () => project,
                    'defaultAppBundleId': () => null
                };

                const config = requestedFields.reduce((result, field) => {
                    if (field in configMap) {
                        result[field] = configMap[field]();
                    }
                    return result;
                }, {} as Record<string, unknown>);

                console.debug(`[onIFrameRequest] Sending config`, config);

                event.source?.postMessage({
                    type: 'CONN_RESPONSE',
                    identity: identity,
                    config: config
                }); //, event.origin);
            }
        };

        window.addEventListener("message", onIFrameRequest);
        return () => {
            window.removeEventListener("message", onIFrameRequest);
        };
    }, [tenant, project, authToken, idToken]);


    return useMemo(() => {
        return <div id={SingleChatApp.name}
                    className="flex flex-col h-full w-full min-h-0 min-w-0 bg-slate-100 overflow-hidden">
            <ChatHeader/>

            <div className={`flex flex-row overflow-hidden flex-1 w-full min-h-0 min-w-0`}>
                <ChatSidePanel/>
                {/* Chat Column */}
                <div className={`flex-1 flex flex-col h-full`}>
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
                    <div className={`flex flex-row flex-1 min-h-0 min-w-0`}>
                        <ChatCanvasContext value={chatCanvasContextValue}>
                            <ChatInterface/>
                            <AnimatedExpander contentRef={chatCanvasRef} expanded={!!canvasItemLink}>
                                <ChatCanvas ref={chatCanvasRef}/>
                            </AnimatedExpander>
                        </ChatCanvasContext>
                    </div>
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
                {/*                kbEndpoint={config.kb_search_endpoint || `${getKBAPIBaseAddress()}/kb`}*/}
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
    }, [canvasItemLink, chatCanvasContextValue, updatingQuestions])
};

export default SingleChatApp;
