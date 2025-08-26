/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2025 Elena Viter
 */

import React, {useCallback, useEffect, useMemo, useRef, useState} from 'react';
import {
    BookOpen,
    Bot,
    Database,
    GripVertical,
    Loader,
    LogOut,
    Search,
    Server,
    Settings,
    Sparkles,
    Wifi,
    WifiOff,
    X
} from 'lucide-react';

import {ConfigProvider, useConfigProvider} from './ChatConfigProvider';
import {
    ChatCompleteData,
    ChatDeltaData,
    ChatErrorData,
    ChatEventHandlers,
    ChatRequest,
    ChatStartData,
    ChatStepData,
    getChatServiceSingleton,
    SocketChatOptions,
    UIMessage,
    UseSocketChatReturn,
    WireChatMessage
} from './ChatService';

import {ChatConfigPanel} from './config/ChatConfigPanel';
import {SystemMonitorPanel} from '../monitoring/monitoring';
import KBPanel from '../kb/KBPanel';
import {EnhancedKBSearchResults} from './SearchResults';

import {AuthContextValue, useAuthManagerContext} from '../auth/AuthManager';
import {
    getChatBaseAddress,
    getChatSocketAddress,
    getChatSocketSocketIOPath,
    getKBAPIBaseAddress,
    getWorkingScope
} from '../../AppConfig';

import {
    BundleInfo,
    ChatLogItem,
    createAssistantChatStep,
    createChatMessage,
    EmbedderInfo,
    ModelInfo,
    StepUpdate
} from './types/chat';
import ChatInterface from "./ChatInterface.tsx";
import {apiService} from "../kb/ApiService.tsx";

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------
const server_url = `${getChatBaseAddress()}/landing`;

// -----------------------------------------------------------------------------
// Local Socket.IO hook (decoupled from UI widgets)
// -----------------------------------------------------------------------------
export function useSocketChat(options: SocketChatOptions): UseSocketChatReturn {
    const [isConnected, setIsConnected] = useState(false);
    const [isConnecting, setIsConnecting] = useState(false);
    const [socketId, setSocketId] = useState<string | undefined>(undefined);
    const [connectionError, setConnectionError] = useState<string | null>(null);

    const authContext = useAuthManagerContext();

    const stableOpts = useMemo<SocketChatOptions>(() => ({
        baseUrl: options.baseUrl,
        path: options.path ?? '/socket.io',
        transports: options.transports ?? ['websocket'],
        reconnectionAttempts: options.reconnectionAttempts ?? 5,
        timeout: options.timeout ?? 10000,
        autoConnect: false,
        forceNew: false,
        project: options.project,
        tenant: options.tenant,
        namespace: options.namespace ?? '/',
        authContext
    }), [
        options.baseUrl, options.path, options.transports, options.reconnectionAttempts, options.timeout,
        options.project, options.tenant, options.namespace
    ]);

    const service = useMemo(() => getChatServiceSingleton(stableOpts), [stableOpts]);

    useEffect(() => {
        return () => {
        };
    }, [service]);

    const connect = useCallback(async (handlers: ChatEventHandlers, ac?: AuthContextValue) => {
        setIsConnecting(true);
        setConnectionError(null);

        const enhancedHandlers: ChatEventHandlers = {
            ...handlers,
            onConnect: () => {
                setIsConnected(true);
                setIsConnecting(false);
                setSocketId(service.socketId);
                setConnectionError(null);
                handlers.onConnect?.();
            },
            onDisconnect: (reason: string) => {
                setIsConnected(false);
                setSocketId(undefined);
                handlers.onDisconnect?.(reason);
            },
            onConnectError: (error: Error) => {
                setIsConnecting(false);
                setConnectionError(error.message);
                handlers.onConnectError?.(error);
            },
        };

        await service.connect(enhancedHandlers, ac);
    }, [service]);

    const disconnect = useCallback(() => {
        service.disconnect();
        setIsConnected(false);
        setIsConnecting(false);
        setSocketId(undefined);
        setConnectionError(null);
    }, [service]);

    const sendMessage = useCallback((request: ChatRequest) => {
        if (!service.connected) throw new Error('Not connected to chat service');
        service.sendChatMessage(request);
    }, [service]);

    const ping = useCallback(() => {
        if (!service.connected) throw new Error('Not connected to chat service');
        service.ping();
    }, [service]);

    return {isConnected, isConnecting, socketId, connect, disconnect, sendMessage, ping, connectionError};
}

// -----------------------------------------------------------------------------
// Helper: KB search results wrapper
// -----------------------------------------------------------------------------
const UpdatedSearchResultsHistory = ({searchHistory, onClose, kbEndpoint}: {
    searchHistory: any[];
    onClose: () => void;
    kbEndpoint: string;
}) => {
    return (
        <EnhancedKBSearchResults
            searchResults={searchHistory}
            onClose={onClose}
            kbEndpoint={kbEndpoint}
        />
    );
};

// -----------------------------------------------------------------------------
// Types specific to this file (UI message)
// -----------------------------------------------------------------------------
interface ChatMessage {
    id: number;
    sender: 'user' | 'assistant';
    text: string;
    timestamp: Date;
    isError?: boolean;
    metadata?: {
        is_our_domain?: boolean;
        classification_reasoning?: string;
        selected_model?: string;
        execution_summary?: {
            total_time: string;
            total_steps: number;
        };
        retrieved_docs?: number;
        reranked_docs?: number;
        config_info?: {
            provider?: string;
            custom_embedding_endpoint?: boolean;
        };
    };
}

// -----------------------------------------------------------------------------
// Main Component
// -----------------------------------------------------------------------------
const SingleChatApp: React.FC = () => {
    // Persisted config provider (decoupled from UI widgets)
    const configProvider = useMemo(() => new ConfigProvider({
        storageKey: 'ai_assistant_config_v1',
        encryptionKey: 'ai_config_secure_key'
    }), []);

    const {
        config,
        isValid: isConfigValid,
        validationErrors,
        updateConfig,
        setConfigValue
    } = useConfigProvider(configProvider);

    const authContext = useAuthManagerContext();
    const workingScope = getWorkingScope();
    const project = workingScope.project;
    const tenant = workingScope.tenant;

    // Socket
    const {
        isConnected: isSocketConnected,
        isConnecting: isSocketConnecting,
        socketId,
        connect: connectSocket,
        disconnect: disconnectSocket,
        sendMessage: sendSocketMessage,
        connectionError
    } = useSocketChat({
        baseUrl: getChatSocketAddress(),
        path: getChatSocketSocketIOPath(),
        authContext,
        project,
        tenant
    });

    // Messages state
    const [messages, setMessages] = useState<ChatMessage[]>([{
        id: 1,
        sender: 'assistant',
        text: "Hello! I'm your AI assistant. Choose options in Config and ask me anything — I'll show processing steps live.",
        timestamp: new Date()
    }]);

    // Streaming control
    const streamingTaskIdRef = useRef<string | null>(null);
    const streamingMsgIdRef = useRef<number | null>(null);
    const deltaBufferRef = useRef<string>('');
    const flushTimerRef = useRef<number | null>(null);
    const sawFirstDeltaRef = useRef(false);

    const [isProcessing, setIsProcessing] = useState<boolean>(false);

    // Panels and header meta (decoupled)
    const [showConfig, setShowConfig] = useState<boolean>(() => config.show_config);
    const [showKB, setShowKB] = useState<boolean>(false);
    const [showKbResults, setShowKbResults] = useState<boolean>(false);
    const [showSystemMonitor, setShowSystemMonitor] = useState<boolean>(false);

    const [currentSteps, setCurrentSteps] = useState<StepUpdate[]>([]);
    const [kbSearchHistory, setKbSearchHistory] = useState<any[]>([]);
    const [newKbSearchCount, setNewKbSearchCount] = useState<number>(0);

    const [headerModel, setHeaderModel] = useState<ModelInfo | undefined>();
    const [headerEmbedder, setHeaderEmbedder] = useState<EmbedderInfo | undefined>();
    const [headerBundle, setHeaderBundle] = useState<BundleInfo | undefined>();

    // Sync toggles to persisted config
    useEffect(() => {
        setShowConfig(config.show_config);
    }, [config.show_config]);

    const handleShowConfigChange = useCallback((show: boolean) => {
        setShowConfig(show);
        setConfigValue('show_config', show);
    }, [setConfigValue]);

    // KB helpers
    const handleKbSearchResults = useCallback((searchResponse: any, isAutomatic: boolean = true) => {
        const enrichedResponse = {
            ...searchResponse,
            searchType: isAutomatic ? 'automatic' : 'manual',
            timestamp: new Date()
        };
        setKbSearchHistory(prev => [enrichedResponse, ...prev.slice(0, 9)]);
        setNewKbSearchCount(prev => prev + 1);
        setTimeout(() => setNewKbSearchCount(0), 5000);
    }, []);
    const handleShowKbResults = useCallback(() => {
        setShowKbResults(true);
        setNewKbSearchCount(0);
    }, []);
    const handleCloseKbResults = useCallback(() => setShowKbResults(false), []);

    // Cleanup flush timer
    useEffect(() => {
        return () => {
            if (flushTimerRef.current != null) {
                window.clearTimeout(flushTimerRef.current);
                flushTimerRef.current = null;
            }
        };
    }, []);

    // Connect Socket.IO
    const didConnectRef = useRef(false);
    useEffect(() => {
        if (didConnectRef.current) return;
        didConnectRef.current = true;
        (async () => {
            try {
                await connectSocket(chatEventHandlers, authContext);
            } catch (e) {
                console.error('Failed to initialize socket:', e);
            }
        })();
        return () => {
            disconnectSocket();
            didConnectRef.current = false;
        };
    }, []);

    // // Quick questions
    // const quickQuestions: string[] = [
    //     "What light, watering, and soil do my common houseplants need?",
    //     "Why are my leaves yellow/brown/curling, and how do I fix it?",
    //     "How can I prevent and treat pests like spider mites and fungus gnats?",
    //     "When should I repot, and what potting mix should I use?"
    // ];
    const [updatingQustions, setUpdatingQustions] = useState<boolean>(false);
    const [quickQuestions, setQuickQuestions] = useState([]);


    useEffect(() => {
        setUpdatingQustions(true);
        apiService.getSuggestedQuestions(tenant, project, authContext).then((data) => {
            setQuickQuestions(data);
        }).catch((e) => {
            console.error(e);
        }).finally(() => setUpdatingQustions(false));
    }, [project, tenant, config, authContext]);

    // Connection status
    const connectionStatus = useMemo(() => {
        if (isSocketConnecting) return {
            icon: <Loader size={14} className="animate-spin"/>,
            text: 'Connecting...',
            color: 'text-yellow-600 bg-yellow-50'
        };
        if (isSocketConnected) return {icon: <Wifi size={14}/>, text: 'Connected', color: 'text-green-600 bg-green-50'};
        return {icon: <WifiOff size={14}/>, text: 'Disconnected', color: 'text-red-600 bg-red-50'};
    }, [isSocketConnected, isSocketConnecting]);

    // Logout
    const handleLogout = useCallback(async () => {
        try {
            disconnectSocket();
            await authContext.logout();
        } catch (e) {
            console.error('Logout error:', e);
        }
    }, [disconnectSocket, authContext]);

    // Streaming flush
    const flushBuffered = useCallback(() => {
        if (!deltaBufferRef.current) return;
        const chunk = deltaBufferRef.current;
        deltaBufferRef.current = '';
        setMessages(prev => {
            const msgId = streamingMsgIdRef.current;
            if (msgId == null) return prev;
            const idx = prev.findIndex(m => m.id === msgId);
            if (idx === -1) return prev;
            const updated = [...prev];
            const current = updated[idx];
            const safeText = typeof current.text === 'string' ? current.text : '';
            updated[idx] = {...current, text: safeText + chunk};
            return updated;
        });
        flushTimerRef.current = null;
    }, []);

    // Socket handlers
    const chatEventHandlers: ChatEventHandlers = useMemo(() => ({
        onConnect: () => { /* no-op */
        },
        onSessionInfo: (info) => {
            console.log('Server session:', info.session_id, info.user_type);
        },
        onDisconnect: (reason: string) => {
            console.log('Disconnected:', reason);
            setIsProcessing(false);
            if (flushTimerRef.current) {
                window.clearTimeout(flushTimerRef.current);
                flushTimerRef.current = null;
            }
            deltaBufferRef.current = '';
            streamingTaskIdRef.current = null;
            streamingMsgIdRef.current = null;
            sawFirstDeltaRef.current = false;
        },
        onConnectError: (error: Error) => {
            console.error('Connect error:', error);
            setIsProcessing(false);
        },

        onChatStart: (data: ChatStartData & { task_id?: string }) => {
            if (data.task_id) streamingTaskIdRef.current = data.task_id;
            sawFirstDeltaRef.current = false;
            deltaBufferRef.current = '';
        },

        onChatDelta: (data: ChatDeltaData) => {
            if (streamingTaskIdRef.current && data.task_id !== streamingTaskIdRef.current) return;

            if (streamingMsgIdRef.current == null) {
                setMessages(prev => {
                    const last = prev[prev.length - 1];
                    if (last && last.sender === 'assistant' && (last.text ?? '') === '' && !last.isError) {
                        streamingMsgIdRef.current = last.id;
                        return prev;
                    }
                    const id = Date.now();
                    streamingMsgIdRef.current = id;
                    return [...prev, {
                        id,
                        sender: 'assistant',
                        text: '',
                        timestamp: new Date(Date.parse(data.timestamp))
                    }];
                });
            }
            if (!sawFirstDeltaRef.current) {
                sawFirstDeltaRef.current = true;
                setIsProcessing(false);
            }

            deltaBufferRef.current += (data.delta || '');
            if (flushTimerRef.current == null) {
                flushTimerRef.current = window.setTimeout(() => {
                    flushBuffered();
                }, 24) as unknown as number;
            }
        },

        onChatStep: (data: ChatStepData) => {
            if (data.step === 'rag_retrieval' && (data as any)?.data?.kb_search_results) {
                handleKbSearchResults((data as any).data.kb_search_results, true);
            }
            const stepUpdate: StepUpdate = {
                step: data.step, status: data.status, timestamp: new Date(Date.parse(data.timestamp)),
                elapsed_time: data.elapsed_time, error: data.error, data: data.data, title: data.title, turn_id: data.turn_id
            };

            setCurrentSteps(prev => {
                const existing = prev.find(s => s.step === data.step && s.turn_id === data.turn_id);
                return existing ? prev.map(s => (s.step === data.step ? stepUpdate : s)) : [...prev, stepUpdate];
            });
        },

        onChatComplete: (data: ChatCompleteData & { task_id?: string }) => {
            if (flushTimerRef.current != null) {
                window.clearTimeout(flushTimerRef.current);
                flushTimerRef.current = null;
            }
            if (deltaBufferRef.current) flushBuffered();
            const msgId = streamingMsgIdRef.current;

            setMessages(prev => {
                if (msgId != null) {
                    const idx = prev.findIndex(m => m.id === msgId);
                    if (idx !== -1) {
                        const updated = [...prev];
                        const current = updated[idx];
                        const finalText = data.final_answer && data.final_answer.length > (current.text || '').length
                            ? data.final_answer : current.text;
                        updated[idx] = {
                            ...current, text: finalText, timestamp: new Date(Date.parse(data.timestamp)),
                            metadata: {
                                ...current.metadata,
                                is_our_domain: (data as any).is_our_domain,
                                classification_reasoning: (data as any).classification_reasoning,
                                selected_model: (data as any).selected_model,
                                retrieved_docs: (data as any).retrieved_docs?.length || 0,
                                reranked_docs: (data as any).reranked_docs?.length || 0,
                                config_info: (data as any).config_info,
                            }
                        };
                        return updated;
                    }
                }
                return [...prev, {
                    id: Date.now() + 1, sender: 'assistant', text: data.final_answer, timestamp: new Date(Date.parse(data.timestamp)),
                    metadata: {
                        is_our_domain: (data as any).is_our_domain,
                        classification_reasoning: (data as any).classification_reasoning,
                        selected_model: (data as any).selected_model,
                        retrieved_docs: (data as any).retrieved_docs?.length || 0,
                        reranked_docs: (data as any).reranked_docs?.length || 0,
                        config_info: (data as any).config_info
                    }
                }];
            });

            streamingTaskIdRef.current = null;
            streamingMsgIdRef.current = null;
            deltaBufferRef.current = '';
            sawFirstDeltaRef.current = false;
            setIsProcessing(false);
        },

        onChatError: (data: ChatErrorData) => {
            if (flushTimerRef.current != null) {
                window.clearTimeout(flushTimerRef.current);
                flushTimerRef.current = null;
            }
            deltaBufferRef.current = '';
            streamingTaskIdRef.current = null;
            streamingMsgIdRef.current = null;
            sawFirstDeltaRef.current = false;
            setMessages(prev => [...prev, {
                id: Date.now() + 1, sender: 'assistant',
                text: `I encountered an error: ${data.error}. Please check your configuration and try again.`,
                timestamp: new Date(Date.parse(data.timestamp)), isError: true
            }]);
            setIsProcessing(false);
        }
    }), [flushBuffered, handleKbSearchResults]);

    // Send message
    const sendMessage = useCallback(async (message: string): Promise<void> => {
        if (!message.trim() || isProcessing) return;
        if (!isSocketConnected) {
            alert('Not connected to chat service.');
            return;
        }

        const userMessage: ChatMessage = {
            id: Date.now(),
            sender: 'user',
            text: message.trim(),
            timestamp: new Date()
        };
        setMessages(prev => [...prev, userMessage]);
        setIsProcessing(true);

        const toWire = (msgs: UIMessage[]): WireChatMessage[] =>
            msgs.filter(m => m.sender === 'user' || m.sender === 'assistant')
                .map(m => ({role: m.sender, content: m.text, timestamp: m.timestamp.toISOString(), id: (m as any).id}));

        try {
            sendSocketMessage({
                message: userMessage.text,
                chat_history: toWire(messages),
                config,   // includes agentic_bundle_id set in ChatConfigPanel
                project, tenant
            });
        } catch (error) {
            console.error('Error sending message via socket:', error);
            setMessages(prev => [...prev, {
                id: Date.now() + 1,
                sender: 'assistant',
                text: `I couldn't send your message: ${(error as Error).message}`,
                timestamp: new Date(),
                isError: true
            }]);
            setIsProcessing(false);
        }
    }, [isProcessing, isSocketConnected, sendSocketMessage, messages, config, project, tenant]);

    // UI helpers
    const hideKB = () => setShowKB(false);
    const toggleSystemMonitor = () => setShowSystemMonitor(prev => !prev);

    const chatLogItems: ChatLogItem[] = messages.map(createChatMessage)
    chatLogItems.push(...currentSteps.map(createAssistantChatStep))
    chatLogItems.sort((a, b) => a.timestamp.getTime() - b.timestamp.getTime());

    const renderFullHeader = () => {
      return (
          <div className="bg-white border-b border-gray-200 px-6 py-4">
              <div className="flex items-center justify-between">
                  <div className="flex items-center">
                      <div
                          className="w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg mr-3 flex items-center justify-center">
                          {headerModel?.provider === 'anthropic' ? <Sparkles size={20} className="text-white"/> :
                              <Bot size={20} className="text-white"/>}
                      </div>
                      <div>
                          <h1 className="text-xl font-semibold text-gray-900">
                              {headerModel?.description || 'AI Assistant'}
                          </h1>
                          <p className="text-sm text-gray-500 flex items-center">
                              <Server size={14} className="mr-1"/>
                              {headerModel?.provider || 'Unknown'} • {headerModel?.has_classifier ? ' Domain Classification' : ' Direct Processing'}
                              <span className="flex items-center ml-1">
                    <Database size={12} className="mr-1"/>
                                  {headerEmbedder ? `${headerEmbedder.provider}${headerEmbedder.model ? ` (${headerEmbedder.model})` : ''}` : 'Embeddings'}
                  </span>
                              {headerBundle && (
                                  <span className="flex items-center ml-1">
                      • <Server size={12} className="mx-1"/> Bundle: {headerBundle.name || headerBundle.id}
                    </span>
                              )}
                              {config.kb_search_endpoint && (
                                  <span className="flex items-center ml-1"> • <BookOpen size={12}
                                                                                        className="mr-1"/> KB Search</span>
                              )}
                              <span className="flex items-center ml-2"> • {connectionStatus.icon}<span
                                  className="ml-1 text-xs">Streaming</span></span>
                          </p>
                      </div>
                  </div>

                  <div className="flex items-center gap-2">
                      {/* Connection status pill */}
                      <div className={`flex items-center px-3 py-1 rounded-lg text-sm ${connectionStatus.color}`}>
                          {connectionStatus.icon}
                          <span className="ml-2 font-medium">{connectionStatus.text}</span>
                          {socketId &&
                              <span className="ml-2 text-xs opacity-75">({socketId.slice(0, 8)}...)</span>}
                      </div>

                      <button
                          onClick={() => setShowKB(!showKB)}
                          className="relative flex items-center px-3 py-2 rounded-lg bg-gray-100 text-gray-600 hover:bg-gray-200"
                          title="View KB"
                      >
                          <Database size={16} className="mr-1"/><span className="text-sm">KB</span>
                      </button>

                      <button
                          onClick={handleShowKbResults}
                          className={`relative flex items-center px-3 py-2 rounded-lg transition-colors ${
                              kbSearchHistory.length > 0 ? 'bg-blue-100 text-blue-700 hover:bg-blue-200' : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                          }`}
                          title="View KB Search Results"
                      >
                          <Search size={16} className="mr-1"/>
                          <span className="text-sm">KB Search</span>
                          {kbSearchHistory.length > 0 && (
                              <span
                                  className="ml-1 text-xs bg-blue-200 text-blue-800 px-1 rounded">{kbSearchHistory.length}</span>
                          )}
                          {newKbSearchCount > 0 && (
                              <span
                                  className="absolute -top-1 -right-1 w-2 h-2 bg-red-500 rounded-full animate-pulse"/>
                          )}
                      </button>

                      <button
                          onClick={() => handleShowConfigChange(!showConfig)}
                          className="flex items-center px-3 py-2 bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-lg"
                      >
                          <Settings size={16} className="mr-1"/><span className="text-sm">Config</span>
                      </button>

                      <button
                          onClick={toggleSystemMonitor}
                          className={`relative flex items-center px-3 py-2 rounded-lg transition-colors ${
                              showSystemMonitor ? 'bg-green-100 text-green-700 hover:bg-green-200' : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                          }`}
                          title={showSystemMonitor ? "Hide Monitor" : "Show Monitor"}
                      >
                          <Server size={16} className="mr-1"/>
                          <span className="text-sm">Monitor</span>
                          <div className="ml-2 w-2 h-2 bg-green-400 rounded-full animate-pulse"/>
                          {showSystemMonitor && <div className="ml-1 w-1 h-1 bg-green-600 rounded-full"/>}
                      </button>

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

    const renderSimpleHeader = () => {
        return (
            <div className="bg-white border-b border-gray-200 px-6 py-4">
                <div className="flex items-center justify-between">
                    <div>
                        AI Chat
                    </div>
                    <div className="flex-1"/>
                    <div className="flex items-center gap-2">
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

    return (
        <div id={SingleChatApp.name} className="flex h-screen bg-gray-50">
            {/* Config Panel (widget) */}
            {showConfig && !!authContext.getUserProfile()?.roles?.includes('kdcube:role:super-admin') && (
                <ChatConfigPanel
                    visible={showConfig}
                    onClose={() => handleShowConfigChange(false)}
                    authContext={authContext}
                    config={config}
                    setConfigValue={setConfigValue}
                    className="w-[520px]"
                    updateConfig={updateConfig}
                    validationErrors={validationErrors}
                    onMetaChange={({model, embedder, bundle}) => {
                        setHeaderModel(model);
                        setHeaderEmbedder(embedder);
                        setHeaderBundle(bundle);
                    }}
                />
            )}

            {/* Main Column */}
            <div className="flex-1 flex flex-col">
                {/* Header */}
                {renderSimpleHeader()}

                {/* Body: Chat + optionally Steps / KB Results / System Monitor */}
                <div className={`flex-1 flex overflow-hidden transition-all duration-300`}>
                    {/* Chat Column */}
                    <div className={`flex-1 flex flex-col ${showSystemMonitor ? 'mr-4' : ''}`}>
                        {/* Quick Questions */}
                        <div className="px-6 py-4 bg-gray-50 border-b border-gray-200">
                            {updatingQustions ?
                                (<div className="w-full flex">
                                    <Loader size={28} className='animate-spin text-gray-300 mx-auto'/>
                                </div>) :
                                (<>
                                    <h4 className="text-sm font-medium text-gray-700 mb-2">Try these questions:</h4>
                                    <div className="flex flex-wrap gap-2">
                                        {quickQuestions.map((q, idx) => (
                                            <button key={idx} onClick={() => sendMessage(q)}
                                                    disabled={isProcessing || !isSocketConnected}
                                                    className="px-3 py-1 text-xs bg-white text-gray-700 border border-gray-200 rounded-full hover:bg-gray-50 hover:border-gray-300 disabled:opacity-50">
                                                {q}
                                            </button>
                                        ))}
                                    </div>
                                </>)
                            }
                        </div>

                        <ChatInterface chatLogItems={chatLogItems} onSendMessage={sendMessage}
                                       userInputEnabled={isSocketConnected && isConfigValid}
                                       isProcessing={isProcessing}/>
                    </div>

                    {/* KB Search Results Panel */}
                    {showKbResults && (
                        <div className="border-l border-gray-200 bg-white relative" style={{width: `700px`}}>
                            {/* simple draggable bar */}
                            <div
                                className="absolute left-0 top-0 bottom-0 w-1 cursor-col-resize hover:bg-blue-300 group">
                                <div
                                    className="absolute left-0 top-1/2 transform -translate-y-1/2 -translate-x-1 opacity-0 group-hover:opacity-100">
                                    <GripVertical size={16} className="text-gray-400"/>
                                </div>
                            </div>
                            {kbSearchHistory.length > 0 ? (
                                <UpdatedSearchResultsHistory
                                    searchHistory={kbSearchHistory}
                                    onClose={handleCloseKbResults}
                                    kbEndpoint={config.kb_search_endpoint || `${getKBAPIBaseAddress()}/api/kb`}
                                />
                            ) : (
                                <div className="h-full flex flex-col">
                                    <div
                                        className="px-4 py-3 border-b border-gray-200 bg-gray-50 flex items-center justify-between">
                                        <h3 className="font-semibold text-gray-900 text-sm">KB Search Results</h3>
                                        <button onClick={handleCloseKbResults}
                                                className="p-1 hover:bg-gray-200 rounded text-gray-500 hover:text-gray-700">
                                            <X size={14}/>
                                        </button>
                                    </div>
                                    <div className="flex-1 flex items-center justify-center text-gray-500">
                                        <div className="text-center">
                                            <Database size={24} className="mx-auto mb-2 opacity-50"/>
                                            <p>No KB search results yet</p>
                                            <p className="text-xs mt-1">Results will appear here when RAG retrieval
                                                occurs</p>
                                        </div>
                                    </div>
                                </div>
                            )}
                        </div>
                    )}
                </div>
            </div>

            {/* KB Side Panel */}
            {showKB && (
                <div className="fixed inset-0 z-50 flex">
                    <div className="absolute inset-0 bg-transparent backdrop-blur-xs" onClick={hideKB}/>
                    <div className="ml-auto transition-transform h-full w-1/2">
                        <KBPanel onClose={hideKB}/>
                    </div>
                </div>
            )}

            {/* System Monitor Panel (widget) */}
            {showSystemMonitor && (
                <div className="border-l border-gray-200 bg-white relative flex-shrink-0" style={{width: `360px`}}>
                    <SystemMonitorPanel onClose={toggleSystemMonitor}/>
                </div>
            )}
        </div>
    );
};

export default SingleChatApp;
