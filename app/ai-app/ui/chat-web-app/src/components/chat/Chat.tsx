/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2025 Elena Viter
 */

import React, {useState, useRef, useEffect, useCallback, useMemo} from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeHighlight from 'rehype-highlight';

import {ConfigProvider, useConfigProvider} from './ChatConfigProvider';
import {
    // useSocketChat,
    ChatEventHandlers,
    ChatStepData,
    ChatCompleteData,
    ChatErrorData,
    ChatStartData,
    ChatRequest,
    getChatServiceSingleton,
    UseSocketChatReturn,
    SocketChatOptions,
    ChatDeltaData,
    UIMessage, WireChatMessage
} from './ChatService';
import {
    Bot,
    User,
    Send,
    MessageSquare,
    CheckCircle2,
    Clock,
    AlertCircle,
    Settings,
    Play,
    Loader,
    ChevronRight,
    Database,
    Search,
    Zap,
    FileText,
    Sparkles,
    Eye,
    EyeOff,
    X,
    Server,
    Download,
    Upload,
    RotateCcw,
    Wifi,
    WifiOff,
    BookOpen,
    GripVertical, LogOut,
} from 'lucide-react';
import {EnhancedKBSearchResults} from "./SearchResults";
import KBPanel from "../kb/KBPanel.tsx";
import {SystemMonitorPanel} from "../monitoring/monitoring.tsx";
import {AuthContextValue, useAuthManagerContext} from "../auth/AuthManager.tsx";
import {
    getChatBaseAddress,
    getChatSocketAddress,
    getChatSocketSocketIOPath,
    getKBAPIBaseAddress,
    getWorkingScope
} from "../../AppConfig.ts";

const server_url = `${getChatBaseAddress()}/landing`;
const serving_server_url = 'http://localhost:5005/serving/v1';

export function useSocketChat(options: SocketChatOptions): UseSocketChatReturn {
    const [isConnected, setIsConnected] = useState(false);
    const [isConnecting, setIsConnecting] = useState(false);
    const [socketId, setSocketId] = useState<string | undefined>(undefined);
    const [connectionError, setConnectionError] = useState<string | null>(null);

    const authContext = useAuthManagerContext();

    // keep only stable, identity-defining fields in the memo
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
        options.baseUrl,
        options.path,
        options.transports,           // ← add
        options.reconnectionAttempts, // ← add
        options.timeout,              // ← add
        options.project,
        options.tenant,
        options.namespace
    ]);

    const service = useMemo(() => getChatServiceSingleton(stableOpts), [stableOpts]);

    // We don’t auto-disconnect the singleton on unmount (caller decides who owns it)
    useEffect(() => {
        return () => {};
    }, [service]);


    const connect = useCallback(async (handlers: ChatEventHandlers, authContext?: AuthContextValue) => {
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

        await service.connect(enhancedHandlers, authContext);
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

    return {
        isConnected,
        isConnecting,
        socketId,
        connect,
        disconnect,
        sendMessage,
        ping,
        connectionError,
    };
}

// Resizable Panel Hook
const useResizablePanel = (initialWidth: number, minWidth: number = 200, maxWidth: number = 600) => {
    const [width, setWidth] = useState(initialWidth);
    const [isResizing, setIsResizing] = useState(false);
    const startXRef = useRef(0);
    const startWidthRef = useRef(initialWidth);

    const handleMouseDown = useCallback((e: React.MouseEvent) => {
        e.preventDefault();
        setIsResizing(true);
        startXRef.current = e.clientX;
        startWidthRef.current = width;
    }, [width]);

    const handleMouseMove = useCallback((e: MouseEvent) => {
        if (!isResizing) return;

        const deltaX = startXRef.current - e.clientX;
        const newWidth = Math.min(maxWidth, Math.max(minWidth, startWidthRef.current + deltaX));
        setWidth(newWidth);
    }, [isResizing, minWidth, maxWidth]);

    const handleMouseUp = useCallback(() => {
        setIsResizing(false);
    }, []);

    useEffect(() => {
        if (isResizing) {
            document.addEventListener('mousemove', handleMouseMove);
            document.addEventListener('mouseup', handleMouseUp);
            document.body.style.cursor = 'col-resize';
            document.body.style.userSelect = 'none';

            return () => {
                document.removeEventListener('mousemove', handleMouseMove);
                document.removeEventListener('mouseup', handleMouseUp);
                document.body.style.cursor = '';
                document.body.style.userSelect = '';
            };
        }
    }, [isResizing, handleMouseMove, handleMouseUp]);

    return {
        width,
        handleMouseDown,
        isResizing
    };
};

// Updated Search Results History Component
const UpdatedSearchResultsHistory = ({searchHistory, onClose, kbEndpoint}) => {
    return (
        <EnhancedKBSearchResults
            searchResults={searchHistory}
            onClose={onClose}
            kbEndpoint={kbEndpoint}
        />
    );
};

// Type definitions (keeping existing ones)
interface ModelInfo {
    id: string;
    name: string;
    provider: string;
    description: string;
    has_classifier: boolean;
}

interface EmbedderInfo {
    id: string;
    provider: string;
    model: string;
    dimension: number;
    description: string;
}

interface EmbeddingProvider {
    name: string;
    description: string;
    requires_api_key: boolean;
    requires_endpoint: boolean;
}

interface ChatMessage {
    id: number;
    sender: 'user' | 'assistant';
    text: string;
    timestamp: string;
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

interface StepData {
    message?: string;
    model?: string;
    embedding_type?: string;
    embedding_endpoint?: string;
    is_our_domain?: boolean;
    confidence?: number;
    query_count?: number;
    retrieved_count?: number;
    answer_length?: number;
    queries?: string[];
    sources?: string[];
    avg_relevance?: number;
}

interface StepUpdate {
    step: string;
    status: 'started' | 'completed' | 'error';
    timestamp: string;
    elapsed_time?: string;
    error?: string;
    data?: StepData;
}

interface EmbeddingTestResponse {
    status: string;
    embedder_id: string;
    provider: string;
    model: string;
    embedding_size: number;
    test_text: string;
    embedding_preview: number[];
}

interface KBSearchResult {
    query: string;
    relevance_score: number;
    heading: string;
    subheading: string;
    backtrack: Record<string, any>;
}

interface KBSearchResponse {
    query: string;
    results: KBSearchResult[];
    total_results: number;
    search_metadata: Record<string, any>;
}

// Fallback models
const fallbackModels: Record<string, ModelInfo> = {
    'gpt-4o': {
        id: 'gpt-4o',
        name: 'gpt-4o',
        provider: 'openai',
        description: 'GPT-4 Optimized - Latest OpenAI model',
        has_classifier: true
    },
    'gpt-4o-mini': {
        id: 'gpt-4o-mini',
        name: 'gpt-4o-mini',
        provider: 'openai',
        description: 'GPT-4 Optimized Mini - High performance, cost-effective',
        has_classifier: true
    }
};

const SingleChatApp: React.FC = () => {
    // Initialize ConfigProvider
    const configProvider = useMemo(() => new ConfigProvider({
        storageKey: 'ai_assistant_config_v1',
        encryptionKey: 'ai_config_secure_key'
    }), []);

    // Use the config provider hook
    const {
        config,
        isValid: isConfigValid,
        validationErrors,
        updateConfig,
        setConfigValue,
        resetConfig
    } = useConfigProvider(configProvider);

    const authContext = useAuthManagerContext();
    const workingScope = getWorkingScope();
    // Socket.IO chat service
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
        authContext: authContext,
        project: workingScope.project,
        tenant: workingScope.tenant
    });

    // State management
    const [messages, setMessages] = useState<ChatMessage[]>([
        {
            id: 1,
            sender: 'assistant',
            text: "Hello! I'm your AI assistant. Choose a model and ask me anything - I'll show you how I process your request step by step.",
            timestamp: new Date().toISOString()
        }
    ]);
    // Track the single streaming message
    const streamingTaskIdRef = useRef<string | null>(null);
    const streamingMsgIdRef = useRef<number | null>(null);

    // Delta buffering and first-delta detection
    const deltaBufferRef = useRef<string>('');
    const flushTimerRef = useRef<number | null>(null);
    const sawFirstDeltaRef = useRef(false);


    const handleLogout = useCallback(async () => {
        try {
            // close your socket cleanly first
            disconnectSocket();
            await authContext.logout();  // now available on the context
        } catch (e) {
            console.error('Logout error:', e);
        }
    }, [disconnectSocket, authContext]);

    const flushBuffered = useCallback(() => {
        if (!deltaBufferRef.current) return;
        const chunk = deltaBufferRef.current;
        deltaBufferRef.current = '';

        setMessages(prev => {
            const msgId = streamingMsgIdRef.current;
            if (msgId == null) return prev; // no streaming bubble yet

            const idx = prev.findIndex(m => m.id === msgId);
            if (idx === -1) return prev;    // bubble was removed somehow

            const updated = [...prev];
            const current = updated[idx];
            const safeText = typeof current.text === 'string' ? current.text : '';
            updated[idx] = { ...current, text: safeText + chunk };
            return updated;
        });

        flushTimerRef.current = null;
    }, [setMessages]);

    useEffect(() => {
        return () => {
            if (flushTimerRef.current != null) {
                window.clearTimeout(flushTimerRef.current);
                flushTimerRef.current = null;
            }
        };
    }, []);

    const [input, setInput] = useState<string>('');
    const [isProcessing, setIsProcessing] = useState<boolean>(false);
    const [currentSteps, setCurrentSteps] = useState<StepUpdate[]>([]);
    const [showSteps, setShowSteps] = useState<boolean>(() => config.show_steps);
    const [showConfig, setShowConfig] = useState<boolean>(() => config.show_config);
    const [availableModels, setAvailableModels] = useState<Record<string, ModelInfo>>({});
    const [availableEmbedders, setAvailableEmbedders] = useState<Record<string, EmbedderInfo>>({});
    const [embeddingProviders, setEmbeddingProviders] = useState<Record<string, EmbeddingProvider>>({});
    const [kbTestQuery, setKbTestQuery] = useState<string>('How does data versioning work?');
    const [kbSearchHistory, setKbSearchHistory] = useState<any[]>([]);
    const [showKbResults, setShowKbResults] = useState<boolean>(false);
    const [newKbSearchCount, setNewKbSearchCount] = useState<number>(0);
    const [showTestSuccess, setShowTestSuccess] = useState<boolean>(false);
    const [showKB, setShowKB] = useState<boolean>(false);
    const [showSystemMonitor, setShowSystemMonitor] = useState<boolean>(false);


    // Resizable panels - WIDER KB panel
    const stepsPanel = useResizablePanel(320, 250, 500);
    const kbPanel = useResizablePanel(600, 400, 900); // Much wider for better search results
    const monitorPanel = useResizablePanel(350, 280, 500);

    // Refs
    const chatRef = useRef<HTMLDivElement>(null);
    const didConnectRef = useRef(false);

    const hideKB = () => {
        setShowKB(false)
    };

    const toggleSystemMonitor = () => {
        setShowSystemMonitor(!showSystemMonitor);
    };

    // Sync UI state with config
    useEffect(() => {
        setShowSteps(config.show_steps);
        setShowConfig(config.show_config);
    }, [config.show_steps, config.show_config]);

    // Update config when UI state changes
    const handleShowStepsChange = useCallback((show: boolean) => {
        setShowSteps(show);
        setConfigValue('show_steps', show);
    }, [setConfigValue]);

    const handleShowConfigChange = useCallback((show: boolean) => {
        setShowConfig(show);
        setConfigValue('show_config', show);
    }, [setConfigValue]);

    // Enhanced KB search results handler with search type tracking
    const handleKbSearchResults = useCallback((searchResponse: any, isAutomatic: boolean = true) => {
        const enrichedResponse = {
            ...searchResponse,
            searchType: isAutomatic ? 'automatic' : 'manual',
            timestamp: new Date().toISOString()
        };

        setKbSearchHistory(prev => [enrichedResponse, ...prev.slice(0, 9)]);
        setNewKbSearchCount(prev => prev + 1);
        setTimeout(() => setNewKbSearchCount(0), 5000);
    }, []);

    const handleShowKbResults = useCallback(() => {
        setShowKbResults(true);
        setNewKbSearchCount(0);
    }, []);

    const handleCloseKbResults = useCallback(() => {
        setShowKbResults(false);
    }, []);

    // Socket.IO Event Handlers
    const chatEventHandlers: ChatEventHandlers = useMemo(() => ({
        onConnect: () => {
            console.log('Connected to chat service');
        },
        onSessionInfo: (info) => {
            console.log('Server session:', info.session_id, info.user_type);
        },
        onDisconnect: (reason: string) => {
            console.log('Disconnected from chat service:', reason);
            setIsProcessing(false);
            // cleanup streaming state
            if (flushTimerRef.current) { window.clearTimeout(flushTimerRef.current); flushTimerRef.current = null; }
            deltaBufferRef.current = '';
            streamingTaskIdRef.current = null;
            streamingMsgIdRef.current = null;
            sawFirstDeltaRef.current = false;
        },
        onConnectError: (error: Error) => {
            console.error('Failed to connect to chat service:', error);
            setIsProcessing(false);
        },

        // --- STREAMING ---
        onChatStart: (data: ChatStartData & { task_id?: string }) => {
            console.log('Chat started:', data);

            // Note: do NOT create a message here. Just mark the task.
            if (data.task_id) streamingTaskIdRef.current = data.task_id;
            sawFirstDeltaRef.current = false;
            deltaBufferRef.current = '';
            // keep your existing behavior
            setCurrentSteps([]);
        },
        onChatDelta: (data: ChatDeltaData) => {
            // If server sends task_id, ignore mismatched streams
            if (streamingTaskIdRef.current && data.task_id !== streamingTaskIdRef.current) return;

            // Lazily create (or reuse) the streaming bubble on the first delta
            if (streamingMsgIdRef.current == null) {
                // Try to reuse a last empty assistant bubble if it somehow exists
                setMessages(prev => {
                    const last = prev[prev.length - 1];
                    if (last && last.sender === 'assistant' && (last.text ?? '') === '' && !last.isError) {
                        streamingMsgIdRef.current = last.id;
                        return prev;
                    }
                    const id = Date.now();
                    streamingMsgIdRef.current = id;
                    return [
                        ...prev,
                        {
                            id,
                            sender: 'assistant',
                            text: '',
                            timestamp: data.timestamp, // fine if undefined
                        }
                    ];
                });
            }

            // First visible token → hide the banner
            if (!sawFirstDeltaRef.current) {
                sawFirstDeltaRef.current = true;
                setIsProcessing(false);
            }

            // Buffer + throttle UI updates
            deltaBufferRef.current += (data.delta || '');
            if (flushTimerRef.current == null) {
                flushTimerRef.current = window.setTimeout(() => {
                    flushBuffered();
                }, 24) as unknown as number;
            }
        },

        onChatStep: (data: ChatStepData) => {
            console.log('Chat step:', data);

            if (data.step === 'rag_retrieval' && data.data?.kb_search_results) {
                handleKbSearchResults(data.data.kb_search_results, true);
            }

            const stepUpdate: StepUpdate = {
                step: data.step,
                status: data.status,
                timestamp: data.timestamp,
                elapsed_time: data.elapsed_time,
                error: data.error,
                data: data.data
            };

            setCurrentSteps(prev => {
                const existing = prev.find(step => step.step === data.step);
                return existing ? prev.map(s => (s.step === data.step ? stepUpdate : s)) : [...prev, stepUpdate];
            });
        },

        onChatComplete: (data: ChatCompleteData & { task_id?: string }) => {
            console.log('Chat completed:', data);

            if (flushTimerRef.current != null) { window.clearTimeout(flushTimerRef.current); flushTimerRef.current = null; }
            if (deltaBufferRef.current) flushBuffered();

            const msgId = streamingMsgIdRef.current;

            setMessages(prev => {
                if (msgId != null) {
                    const idx = prev.findIndex(m => m.id === msgId);
                    if (idx !== -1) {
                        const updated = [...prev];
                        const current = updated[idx];
                        const finalText = data.final_answer && data.final_answer.length > (current.text || '').length
                            ? data.final_answer
                            : current.text;

                        updated[idx] = {
                            ...current,
                            text: finalText,
                            timestamp: data.timestamp,
                            metadata: {
                                ...current.metadata,
                                is_our_domain: data.is_our_domain,
                                classification_reasoning: data.classification_reasoning,
                                selected_model: data.selected_model,
                                retrieved_docs: data.retrieved_docs?.length || 0,
                                reranked_docs: data.reranked_docs?.length || 0,
                                config_info: data.config_info,
                            }
                        };
                        return updated;
                    }
                }

                // No streaming bubble ever created (no deltas) → append one normal assistant message
                return [...prev, {
                    id: Date.now() + 1,
                    sender: 'assistant',
                    text: data.final_answer,
                    timestamp: data.timestamp,
                    metadata: {
                        is_our_domain: data.is_our_domain,
                        classification_reasoning: data.classification_reasoning,
                        selected_model: data.selected_model,
                        retrieved_docs: data.retrieved_docs?.length || 0,
                        reranked_docs: data.reranked_docs?.length || 0,
                        config_info: data.config_info
                    }
                }];
            });

            // cleanup stream state
            streamingTaskIdRef.current = null;
            streamingMsgIdRef.current = null;
            deltaBufferRef.current = '';
            sawFirstDeltaRef.current = false;

            setIsProcessing(false);
        },

        onChatError: (data: ChatErrorData) => {
            console.error('Chat error:', data);

            if (flushTimerRef.current != null) { window.clearTimeout(flushTimerRef.current); flushTimerRef.current = null; }
            deltaBufferRef.current = '';
            streamingTaskIdRef.current = null;
            streamingMsgIdRef.current = null;
            sawFirstDeltaRef.current = false;

            setMessages(prev => [...prev, {
                id: Date.now() + 1,
                sender: 'assistant',
                text: `I apologize, but I encountered an error: ${data.error}. Please check your configuration and try again.`,
                timestamp: data.timestamp,
                isError: true
            }]);

            setIsProcessing(false);
        }
    }), [handleKbSearchResults, flushBuffered]);

    // Connect to Socket.IO on component mount
    useEffect(() => {
        if (didConnectRef.current) return;
        didConnectRef.current = true;
        const initializeSocket = async () => {
            try {
                await connectSocket(chatEventHandlers, authContext);
            } catch (error) {
                console.error('Failed to initialize socket connection:', error);
            }
        };

        initializeSocket();

        return () => {
            disconnectSocket();
            didConnectRef.current = false;
        };
    }, []);

    // Load models and embedders
    useEffect(() => {
        const loadModelsAndEmbedders = async (): Promise<void> => {
            try {
                // Load models
                const headers: HeadersInit = [
                    ['Content-Type', 'application/json']
                ];
                // config
                authContext.appendAuthHeader(headers);
                const modelsResponse = await fetch(`${server_url}/models`, {
                    method: 'GET',
                    headers
                });
                const modelsData = await modelsResponse.json();
                setAvailableModels(modelsData.available_models || {});

                // Load embedders

                const embeddersResponse = await fetch(`${server_url}/embedders`,
                    {
                        method: 'GET',
                        headers
                    });
                const embeddersData = await embeddersResponse.json();
                setAvailableEmbedders(embeddersData.available_embedders || {});
                setEmbeddingProviders(embeddersData.providers || {});

                // Set default embedder if not set
                if (!config.selected_embedder && embeddersData.default_embedder) {
                    setConfigValue('selected_embedder', embeddersData.default_embedder);
                }

            } catch (error) {
                console.error('Error loading models and embedders:', error);
                setAvailableModels(fallbackModels);
            }
        };

        loadModelsAndEmbedders();
    }, [config.selected_embedder, setConfigValue]);

    // Helper functions
    const getSelectedEmbedderInfo = useCallback((): EmbedderInfo => {
        return availableEmbedders[config.selected_embedder] || {} as EmbedderInfo;
    }, [availableEmbedders, config.selected_embedder]);

    const selectedEmbedderInfo = getSelectedEmbedderInfo();
    const requiresCustomEndpoint = selectedEmbedderInfo.provider === 'custom';
    const requiresOpenAIKey = selectedEmbedderInfo.provider === 'openai';
    const selectedModelInfo = availableModels[config.selected_model] || {} as ModelInfo;

    const project = workingScope.project;
    const tenant = workingScope.tenant;

    // Export/Import functions
    const handleExportConfig = useCallback((): void => {
        const exported = configProvider.export(false);
        const blob = new Blob([JSON.stringify(exported, null, 2)], {type: 'application/json'});
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `ai-assistant-config-${new Date().toISOString().split('T')[0]}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }, [configProvider]);

    const handleImportConfig = useCallback((event: React.ChangeEvent<HTMLInputElement>): void => {
        const file = event.target.files?.[0];
        if (!file) return;

        const reader = new FileReader();
        reader.onload = (e) => {
            try {
                const result = e.target?.result;
                if (typeof result === 'string') {
                    const imported = JSON.parse(result);
                    if (configProvider.import(imported, true)) {
                        alert('Configuration imported successfully!');
                    } else {
                        alert('Failed to import configuration. Please check the file format.');
                    }
                }
            } catch (error) {
                alert('Invalid configuration file format.');
            }
        };
        reader.readAsText(file);
        event.target.value = '';
    }, [configProvider]);

    // Test embeddings function
    const testEmbeddings = useCallback(async (): Promise<void> => {
        const embedderInfo = getSelectedEmbedderInfo();

        if (embedderInfo.provider === 'custom' && !config.custom_embedding_endpoint) {
            alert('Please enter a custom embedding endpoint to test');
            return;
        }

        if (embedderInfo.provider === 'openai' && !config.openai_api_key) {
            alert('Please enter your OpenAI API key to test embeddings');
            return;
        }

        try {
            const headers: HeadersInit = [
                ['Content-Type', 'application/json']
            ];
            authContext.appendAuthHeader(headers)

            const response = await fetch(`${server_url}/test-embeddings`, {
                method: 'POST',
                headers,
                body: JSON.stringify(config)
            });

            const data: EmbeddingTestResponse = await response.json();

            if (response.ok) {
                alert(`✅ Embedding test successful!\n\nEmbedder: ${data.embedder_id}\nProvider: ${data.provider}\nModel: ${data.model}\nEmbedding Size: ${data.embedding_size}`);
            } else {
                alert(`❌ Embedding test failed:\n\n${(data as any).detail?.error || 'Unknown error'}`);
            }
        } catch (error) {
            alert(`❌ Embedding test failed:\n\n${(error as Error).message}`);
        }
    }, [config, getSelectedEmbedderInfo]);

    // Test KB search function with visual results - mark as manual
    const testKBSearch = useCallback(async (): Promise<void> => {
        if (!config.kb_search_endpoint) {
            alert('Please enter a KB search endpoint to test');
            return;
        }

        if (!kbTestQuery.trim()) {
            alert('Please enter a test query');
            return;
        }

        try {
            const searchUrl = `${config.kb_search_endpoint}/${project}/search/enhanced`;
            const headers: HeadersInit = [
                ['Content-Type', 'application/json']
            ];
            authContext.appendAuthHeader(headers)
            console.log("Search URL", searchUrl)

            const response = await fetch(searchUrl, {
                method: 'POST',
                headers,
                body: JSON.stringify({
                    query: kbTestQuery.trim(),
                    top_k: 5
                })
            });

            const data: KBSearchResponse = await response.json();
            console.log(`Search.query=${kbTestQuery.trim()}`, `response=`, data);

            if (response.ok) {
                // Add test results to search history as MANUAL search
                handleKbSearchResults(data, false);

                // Auto-open the KB results panel
                setTimeout(() => {
                    handleShowKbResults();
                }, 100);

                // Show success notification
                setShowTestSuccess(true);
                setTimeout(() => setShowTestSuccess(false), 3000);

            } else {
                alert(`❌ KB Search test failed:\n\n${(data as any).detail || 'Unknown error'}`);
            }
        } catch (error) {
            alert(`❌ KB Search test failed:\n\n${(error as Error).message}`);
        }
    }, [config.kb_search_endpoint, kbTestQuery, handleKbSearchResults, handleShowKbResults]);

    // Auto-scroll chat
    useEffect(() => {
        if (chatRef.current) {
            chatRef.current.scrollTop = chatRef.current.scrollHeight;
        }
    }, [messages, currentSteps]);

    // Send message function (now uses Socket.IO)
    const sendMessage = useCallback(async (): Promise<void> => {
        if (!input.trim() || isProcessing) return;

        if (!isSocketConnected) {
            alert('Not connected to chat service. Please check your connection.');
            return;
        }

        const userMessage: ChatMessage = {
            id: Date.now(),
            sender: 'user',
            text: input.trim(),
            timestamp: new Date().toISOString()
        };

        setMessages(prev => [...prev, userMessage]);
        setInput('');
        setIsProcessing(true);
        setCurrentSteps([]);

        const toWire = (msgs: UIMessage[]): WireChatMessage[] =>
            msgs
                .filter(m => m.sender === 'user' || m.sender === 'assistant')
                .map(m => ({
                    role: m.sender,
                    content: m.text,
                    timestamp: m.timestamp,
                    id: m.id,
                }));
        try {
            // Send via Socket.IO instead of fetch
            sendSocketMessage({
                message: userMessage.text,
                chat_history: toWire(messages),
                config,
                project,
                tenant
            });

            // The response will be handled by the socket event handlers
        } catch (error) {
            console.error('Error sending message via socket:', error);

            const errorMessage: ChatMessage = {
                id: Date.now() + 1,
                sender: 'assistant',
                text: `I apologize, but I encountered an error sending your message: ${(error as Error).message}. Please check your connection and try again.`,
                timestamp: new Date().toISOString(),
                isError: true
            };

            setMessages(prev => [...prev, errorMessage]);
            setIsProcessing(false);
        }
    }, [input, isProcessing, isSocketConnected, sendSocketMessage, messages, config]);

    const handleKeyPress = useCallback((e: React.KeyboardEvent): void => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    }, [sendMessage]);

    // Step rendering functions
    const getStepIcon = useCallback((stepName: string): React.ReactNode => {
        switch (stepName) {
            case 'classifier':
                return <Zap size={16}/>;
            case 'query_writer':
                return <FileText size={16}/>;
            case 'rag_retrieval':
                return <Database size={16}/>;
            case 'reranking':
                return <Search size={16}/>;
            case 'answer_generator':
                return <MessageSquare size={16}/>;
            case 'workflow_start':
                return <Play size={16}/>;
            case 'workflow_complete':
                return <CheckCircle2 size={16}/>;
            default:
                return <Clock size={16}/>;
        }
    }, []);

    const getStepColor = useCallback((status: string): string => {
        switch (status) {
            case 'completed':
                return 'text-green-600 bg-green-50 border-green-200';
            case 'started':
                return 'text-blue-600 bg-blue-50 border-blue-200';
            case 'error':
                return 'text-red-600 bg-red-50 border-red-200';
            default:
                return 'text-gray-600 bg-gray-50 border-gray-200';
        }
    }, []);

    const getStepName = useCallback((stepName: string): string => {
        const names: Record<string, string> = {
            'classifier': 'Domain Classification',
            'query_writer': 'Query Generation',
            'rag_retrieval': 'Document Retrieval',
            'reranking': 'Document Reranking',
            'answer_generator': 'Answer Generation',
            'workflow_start': 'Starting Workflow',
            'workflow_complete': 'Workflow Complete',
            'workflow_error': 'Workflow Error'
        };
        return names[stepName] || stepName.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase());
    }, []);

    // Quick questions
    const quickQuestions: string[] = [
        "What light, watering, and soil do my common houseplants need?",
        "Why are my leaves yellow/brown/curling, and how do I fix it?",
        "How can I prevent and treat pests like spider mites and fungus gnats?",
        "When should I repot, and what potting mix should I use?"
    ];

    // Connection status indicator
    const getConnectionStatus = () => {
        if (isSocketConnecting) {
            return {
                icon: <Loader size={14} className="animate-spin"/>,
                text: 'Connecting...',
                color: 'text-yellow-600 bg-yellow-50'
            };
        }
        if (isSocketConnected) {
            return {icon: <Wifi size={14}/>, text: 'Connected', color: 'text-green-600 bg-green-50'};
        }
        return {icon: <WifiOff size={14}/>, text: 'Disconnected', color: 'text-red-600 bg-red-50'};
    };

    const connectionStatus = getConnectionStatus();

    return (
        <div className="flex h-screen bg-gray-50">
            {/* Configuration Sidebar */}
            {showConfig && (
                <div className="w-80 bg-white border-r border-gray-200 p-6 overflow-y-auto">
                    <div className="flex items-center justify-between mb-6">
                        <h2 className="text-lg font-semibold text-gray-900">Configuration</h2>
                        <button
                            onClick={() => handleShowConfigChange(false)}
                            className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
                        >
                            <X size={16}/>
                        </button>
                    </div>

                    <div className="space-y-4">
                        {/* Connection Status */}
                        <div className="border-b pb-4">
                            <h3 className="text-sm font-medium text-gray-700 mb-3">Connection Status</h3>
                            <div className={`flex items-center px-3 py-2 rounded-lg ${connectionStatus.color}`}>
                                {connectionStatus.icon}
                                <span className="ml-2 text-sm font-medium">{connectionStatus.text}</span>
                                {socketId && (
                                    <span className="ml-2 text-xs opacity-75">({socketId.slice(0, 8)}...)</span>
                                )}
                            </div>
                            {connectionError && (
                                <div className="mt-2 text-xs text-red-600 bg-red-50 p-2 rounded">
                                    Error: {connectionError}
                                </div>
                            )}
                        </div>

                        {/* Configuration Management */}
                        <div className="border-b pb-4">
                            <h3 className="text-sm font-medium text-gray-700 mb-3">Config Management</h3>
                            <div className="flex gap-2">
                                <button
                                    onClick={handleExportConfig}
                                    className="flex-1 flex items-center justify-center px-3 py-2 bg-blue-100 text-blue-700 rounded-lg hover:bg-blue-200 transition-colors text-sm"
                                >
                                    <Download size={14} className="mr-1"/>
                                    Export
                                </button>
                                <label
                                    className="flex-1 flex items-center justify-center px-3 py-2 bg-green-100 text-green-700 rounded-lg hover:bg-green-200 transition-colors text-sm cursor-pointer">
                                    <Upload size={14} className="mr-1"/>
                                    Import
                                    <input
                                        type="file"
                                        accept=".json"
                                        onChange={handleImportConfig}
                                        className="hidden"
                                    />
                                </label>
                                <button
                                    onClick={resetConfig}
                                    className="flex items-center justify-center px-3 py-2 bg-red-100 text-red-700 rounded-lg hover:bg-red-200 transition-colors text-sm"
                                    title="Reset to defaults"
                                >
                                    <RotateCcw size={14}/>
                                </button>
                            </div>
                        </div>

                        {/* Validation Errors */}
                        {validationErrors.length > 0 && (
                            <div className="border border-red-200 bg-red-50 rounded-lg p-3">
                                <h4 className="text-sm font-medium text-red-800 mb-2">Configuration Issues:</h4>
                                <ul className="text-xs text-red-700 space-y-1">
                                    {validationErrors.map((error, index) => (
                                        <li key={index} className="flex items-start">
                                            <AlertCircle size={12} className="mr-1 mt-0.5 flex-shrink-0"/>
                                            {error}
                                        </li>
                                    ))}
                                </ul>
                            </div>
                        )}

                        {/* Model Selection */}
                        <div>
                            <label className="block text-sm font-medium text-gray-700 mb-2">
                                AI Assistant Model
                            </label>
                            <select
                                value={config.selected_model}
                                onChange={(e) => setConfigValue('selected_model', e.target.value)}
                                className="w-full p-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                            >
                                {Object.entries(availableModels).map(([modelId, modelInfo]) => (
                                    <option key={modelId} value={modelId}>
                                        {modelInfo.description}
                                    </option>
                                ))}
                            </select>
                            <div className="mt-2 p-2 bg-gray-50 rounded text-xs">
                                <div><strong>Provider:</strong> {selectedModelInfo.provider || 'Unknown'}</div>
                                <div><strong>Classification:</strong> {selectedModelInfo.has_classifier ? 'Yes' : 'No'}
                                </div>
                            </div>
                        </div>

                        {/* API Keys */}
                        <div>
                            <label className="block text-sm font-medium text-gray-700 mb-2">
                                OpenAI API Key {selectedModelInfo.provider === 'openai' &&
                                <span className="text-red-500">*</span>}
                            </label>
                            <input
                                type="password"
                                value={config.openai_api_key}
                                onChange={(e) => setConfigValue('openai_api_key', e.target.value)}
                                placeholder="sk-..."
                                className="w-full p-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                            />
                        </div>

                        <div>
                            <label className="block text-sm font-medium text-gray-700 mb-2">
                                Claude API Key {selectedModelInfo.provider === 'anthropic' &&
                                <span className="text-red-500">*</span>}
                            </label>
                            <input
                                type="password"
                                value={config.claude_api_key}
                                onChange={(e) => setConfigValue('claude_api_key', e.target.value)}
                                placeholder="sk-ant-..."
                                className="w-full p-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                            />
                        </div>

                        {/* KB Search Configuration */}
                        <div className="border-t pt-4">
                            <h3 className="text-sm font-medium text-gray-700 mb-3 flex items-center">
                                <BookOpen size={16} className="mr-2"/>
                                Knowledge Base Configuration
                            </h3>

                            <div>
                                <label className="block text-sm font-medium text-gray-700 mb-2">
                                    KB Search Endpoint
                                </label>
                                <input
                                    type="url"
                                    value={config.kb_search_endpoint || ''}
                                    onChange={(e) => setConfigValue('kb_search_endpoint', e.target.value)}
                                    placeholder="http://localhost:8000/api/kb"
                                    className="w-full p-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 mb-3"
                                />

                                <label className="block text-sm font-medium text-gray-700 mb-2">
                                    Test Query
                                </label>
                                <div className="flex gap-2">
                                    <input
                                        type="text"
                                        value={kbTestQuery}
                                        onChange={(e) => setKbTestQuery(e.target.value)}
                                        placeholder="Enter a test query..."
                                        className="flex-1 p-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                                    />
                                    <button
                                        onClick={testKBSearch}
                                        disabled={!config.kb_search_endpoint || !kbTestQuery.trim()}
                                        className={`px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                                            config.kb_search_endpoint && kbTestQuery.trim()
                                                ? 'bg-purple-500 text-white hover:bg-purple-600'
                                                : 'bg-gray-300 text-gray-500 cursor-not-allowed'
                                        }`}
                                        title="Test KB search endpoint"
                                    >
                                        <Play size={14}/>
                                    </button>
                                </div>
                                <div className="mt-2 text-xs text-gray-500">
                                    Base URL for knowledge base search API. The system will append '/search' to this
                                    endpoint.
                                </div>
                            </div>
                        </div>

                        {/* Embedding Configuration */}
                        <div className="border-t pt-4">
                            <h3 className="text-sm font-medium text-gray-700 mb-3 flex items-center">
                                <Database size={16} className="mr-2"/>
                                Embedding Configuration
                            </h3>

                            <div>
                                <label className="block text-sm font-medium text-gray-700 mb-2">
                                    Embedding Model
                                </label>
                                <select
                                    value={config.selected_embedder}
                                    onChange={(e) => {
                                        const newEmbedderId = e.target.value;
                                        const newEmbedderInfo = availableEmbedders[newEmbedderId] || {} as EmbedderInfo;

                                        updateConfig({
                                            selected_embedder: newEmbedderId,
                                            custom_embedding_endpoint: newEmbedderInfo.provider === 'openai'
                                                ? ''
                                                : config.custom_embedding_endpoint || `${serving_server_url}/embeddings`
                                        });
                                    }}
                                    className="w-full p-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                                >
                                    {Object.entries(availableEmbedders).map(([embedderId, embedderInfo]) => (
                                        <option key={embedderId} value={embedderId}>
                                            {embedderInfo.description}
                                        </option>
                                    ))}
                                </select>

                                <div className="mt-2 p-2 bg-gray-50 rounded text-xs space-y-1">
                                    <div><strong>Provider:</strong> {selectedEmbedderInfo.provider || 'Unknown'}</div>
                                    <div><strong>Model:</strong> {selectedEmbedderInfo.model || 'Unknown'}</div>
                                    <div><strong>Dimensions:</strong> {selectedEmbedderInfo.dimension || 'Unknown'}
                                    </div>
                                </div>
                            </div>

                            {/* Custom Endpoint for custom embedders */}
                            {requiresCustomEndpoint && (
                                <div className="mt-4">
                                    <label className="block text-sm font-medium text-gray-700 mb-2">
                                        Custom Embedding Endpoint
                                        <span className="text-red-500 ml-1">*</span>
                                    </label>
                                    <div className="flex gap-2">
                                        <input
                                            type="url"
                                            value={config.custom_embedding_endpoint}
                                            onChange={(e) => setConfigValue('custom_embedding_endpoint', e.target.value)}
                                            placeholder="http://localhost:5005/serving/v1/embeddings"
                                            className="flex-1 p-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                                        />
                                        <button
                                            onClick={testEmbeddings}
                                            disabled={!config.custom_embedding_endpoint}
                                            className={`px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                                                config.custom_embedding_endpoint
                                                    ? 'bg-blue-500 text-white hover:bg-blue-600'
                                                    : 'bg-gray-300 text-gray-500 cursor-not-allowed'
                                            }`}
                                            title="Test embedding endpoint"
                                        >
                                            <Play size={14}/>
                                        </button>
                                    </div>
                                </div>
                            )}

                            {/* Test Button for OpenAI embeddings */}
                            {selectedEmbedderInfo.provider === 'openai' && (
                                <div className="mt-4">
                                    <button
                                        onClick={testEmbeddings}
                                        disabled={!config.openai_api_key}
                                        className={`w-full px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                                            config.openai_api_key
                                                ? 'bg-green-500 text-white hover:bg-green-600'
                                                : 'bg-gray-300 text-gray-500 cursor-not-allowed'
                                        }`}
                                    >
                                        <Play size={14} className="inline mr-2"/>
                                        Test OpenAI Embeddings
                                    </button>
                                </div>
                            )}
                        </div>

                        {/* Debug Information */}
                        {config.debug_mode && (
                            <div className="border-t pt-4">
                                <h3 className="text-sm font-medium text-gray-700 mb-3">Debug Info</h3>
                                <div className="text-xs bg-gray-100 p-3 rounded font-mono">
                                    <div>Valid: {isConfigValid ? 'Yes' : 'No'}</div>
                                    <div>Storage Size: {configProvider.getStorageSize()} bytes</div>
                                    <div>Has Changes: {configProvider.hasChanges() ? 'Yes' : 'No'}</div>
                                    <div>Socket ID: {socketId || 'Not connected'}</div>
                                </div>
                            </div>
                        )}
                    </div>
                </div>
            )}

            {/* Main Chat Area */}
            {/*<div className="flex-1 flex flex-col">*/}
            <div className="flex-1 flex flex-col">
                {/* Header */}
                <div className="bg-white border-b border-gray-200 px-6 py-4">
                    <div className="flex items-center justify-between">
                        <div className="flex items-center">
                            <div
                                className="w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg mr-3 flex items-center justify-center">
                                {selectedModelInfo.provider === 'anthropic' ? (
                                    <Sparkles size={20} className="text-white"/>
                                ) : (
                                    <Bot size={20} className="text-white"/>
                                )}
                            </div>
                            <div>
                                <h1 className="text-xl font-semibold text-gray-900">
                                    {selectedModelInfo.description || 'AI Assistant'}
                                </h1>
                                <p className="text-sm text-gray-500 flex items-center">
                                    <Server size={14} className="mr-1"/>
                                    {selectedModelInfo.provider || 'Unknown'} •
                                    {selectedModelInfo.has_classifier ? ' Domain Classification' : ' Direct Processing'} •
                                    <span className="flex items-center ml-1">
                                        <Database size={12} className="mr-1"/>
                                        {selectedEmbedderInfo.provider === 'openai'
                                            ? `OpenAI (${selectedEmbedderInfo.model})`
                                            : selectedEmbedderInfo.provider === 'custom'
                                                ? `Custom (${selectedEmbedderInfo.model})`
                                                : 'Unknown Embeddings'
                                        }
                                    </span>
                                    {config.kb_search_endpoint && (
                                        <span className="flex items-center ml-1">
                                            • <BookOpen size={12} className="mr-1"/>
                                            KB Search
                                        </span>
                                    )}
                                    <span className="flex items-center ml-2">
                                        • {connectionStatus.icon}
                                        <span className="ml-1 text-xs">Streaming</span>
                                    </span>
                                </p>
                            </div>
                        </div>

                        <div className="flex items-center gap-2">
                            {!isConfigValid && (
                                <div
                                    className="flex items-center text-amber-600 bg-amber-50 px-3 py-1 rounded-lg text-sm">
                                    <AlertCircle size={16} className="mr-1"/>
                                    Configuration Required
                                </div>
                            )}

                            {showTestSuccess && (
                                <div
                                    className="flex items-center text-green-600 bg-green-50 px-3 py-1 rounded-lg text-sm animate-pulse">
                                    <CheckCircle2 size={16} className="mr-1"/>
                                    KB Search test successful! Results shown in panel →
                                </div>
                            )}

                            <button
                                onClick={() => setShowKB(!showKB)}
                                className={`relative flex items-center px-3 py-2 rounded-lg bg-gray-100 text-gray-600 hover:bg-gray-200`}
                                title="View KB">
                                <Database size={16} className="mr-1"/>
                                <span className="text-sm">KB</span>
                            </button>

                            {/* KB Search Results Button */}
                            <button
                                onClick={handleShowKbResults}
                                className={`relative flex items-center px-3 py-2 rounded-lg transition-colors ${
                                    kbSearchHistory.length > 0
                                        ? 'bg-blue-100 text-blue-700 hover:bg-blue-200'
                                        : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                                }`}
                                title="View KB Search Results"
                            >
                                <Search size={16} className="mr-1"/>
                                <span className="text-sm">KB Search</span>
                                {kbSearchHistory.length > 0 && (
                                    <span className="ml-1 text-xs bg-blue-200 text-blue-800 px-1 rounded">
                                        {kbSearchHistory.length}
                                    </span>
                                )}
                                {newKbSearchCount > 0 && (
                                    <span
                                        className="absolute -top-1 -right-1 w-2 h-2 bg-red-500 rounded-full animate-pulse"/>
                                )}
                            </button>

                            <button
                                onClick={() => handleShowStepsChange(!showSteps)}
                                className="flex items-center px-3 py-2 text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded-lg transition-colors"
                            >
                                {showSteps ? <EyeOff size={16} className="mr-1"/> : <Eye size={16} className="mr-1"/>}
                                <span className="text-sm">{showSteps ? 'Hide' : 'Show'} Steps</span>
                            </button>

                            <button
                                onClick={() => handleShowConfigChange(!showConfig)}
                                className="flex items-center px-3 py-2 bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-lg transition-colors"
                            >
                                <Settings size={16} className="mr-1"/>
                                <span className="text-sm">Config</span>
                            </button>
                            {/* System Monitor Button */}
                            {/* System Monitor Toggle Button */}
                            <button
                                onClick={toggleSystemMonitor}
                                className={`relative flex items-center px-3 py-2 rounded-lg transition-colors ${
                                    showSystemMonitor
                                        ? 'bg-green-100 text-green-700 hover:bg-green-200'
                                        : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                                }`}
                                title={showSystemMonitor ? "Hide Monitor" : "Show Monitor"}>
                                <Server size={16} className="mr-1"/>
                                <span className="text-sm">Monitor</span>
                                {/* Live status indicator dot */}
                                <div className="ml-2 w-2 h-2 bg-green-400 rounded-full animate-pulse" />
                                {/* Toggle indicator */}
                                {showSystemMonitor && (
                                    <div className="ml-1 w-1 h-1 bg-green-600 rounded-full" />
                                )}
                            </button>
                            <button
                                onClick={handleLogout}
                                className="flex items-center px-3 py-2 bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-lg transition-colors"
                                title="Sign out"
                            >
                                <LogOut size={16} className="mr-1" />
                                <span className="text-sm">Logout</span>
                            </button>
                        </div>
                    </div>
                </div>

                <div className={`flex-1 flex overflow-hidden transition-all duration-300`}>
                    {/* Chat Messages */}
                    {/*<div className="flex-1 flex flex-col">*/}
                    <div className={`flex-1 flex flex-col ${showSystemMonitor ? 'mr-4' : ''}`}>
                    {/* Quick Questions */}
                        <div className="px-6 py-4 bg-gray-50 border-b border-gray-200">
                            <h4 className="text-sm font-medium text-gray-700 mb-2">Try these questions:</h4>
                            <div className="flex flex-wrap gap-2">
                                {quickQuestions.map((question, index) => (
                                    <button
                                        key={index}
                                        onClick={() => setInput(question)}
                                        disabled={isProcessing || !isSocketConnected}
                                        className="px-3 py-1 text-xs bg-white text-gray-700 border border-gray-200 rounded-full hover:bg-gray-50 hover:border-gray-300 transition-colors disabled:opacity-50"
                                    >
                                        {question}
                                    </button>
                                ))}
                            </div>
                        </div>

                        {/* Messages */}
                        <div
                            ref={chatRef}
                            className="flex-1 overflow-y-auto px-6 py-4 space-y-4"
                        >
                            {messages.map(message => (
                                <div
                                    key={message.id}
                                    className={`flex ${message.sender === 'user' ? 'justify-end' : 'justify-start'}`}
                                >
                                    {message.sender === 'assistant' && (
                                        <div
                                            className="w-8 h-8 rounded-full bg-gradient-to-br from-blue-500 to-purple-600 mr-3 flex items-center justify-center flex-shrink-0">
                                            {selectedModelInfo.provider === 'anthropic' ? (
                                                <Sparkles size={16} className="text-white"/>
                                            ) : (
                                                <Bot size={16} className="text-white"/>
                                            )}
                                        </div>
                                    )}

                                    <div className="flex flex-col max-w-3xl">
                                        <div
                                            className={`p-4 rounded-lg ${
                                                message.sender === 'user'
                                                    ? 'bg-blue-500 text-white'
                                                    : message.isError
                                                        ? 'bg-red-50 text-red-800 border border-red-200'
                                                        : 'bg-white text-gray-800 border border-gray-200 shadow-sm'
                                            }`}
                                        >
                                            {message.sender === 'assistant' ? (
                                                <ReactMarkdown
                                                    remarkPlugins={[remarkGfm]}
                                                    rehypePlugins={[rehypeHighlight]}
                                                    components={{
                                                        code({inline, className, children, ...props}) {
                                                            return inline ? (
                                                                <code className="px-1 py-0.5 rounded bg-gray-100" {...props}>{children}</code>
                                                            ) : (
                                                                <pre className="p-3 rounded bg-gray-900 text-gray-100 overflow-auto">
            <code className={className} {...props}>{children}</code>
          </pre>
                                                            );
                                                        },
                                                        a({children, ...props}) {
                                                            return <a className="text-blue-600 underline" target="_blank" rel="noreferrer" {...props}>{children}</a>;
                                                        }
                                                    }}
                                                >
                                                    {message.text}
                                                </ReactMarkdown>
                                            ) : (
                                                <p className="text-sm leading-relaxed whitespace-pre-wrap">{message.text}</p>
                                            )}
                                        </div>

                                        {/* Message metadata */}
                                        {message.metadata && (
                                            <div className="mt-2 text-xs text-gray-500 space-y-1">
                                                {message.metadata.is_our_domain !== undefined && (
                                                    <div className="flex items-center">
                                                        <span className="mr-2">Classification:</span>
                                                        <span className={`px-2 py-1 rounded text-xs ${
                                                            message.metadata.is_our_domain
                                                                ? 'bg-green-100 text-green-700'
                                                                : 'bg-gray-100 text-gray-700'
                                                        }`}>
                                                            {message.metadata.is_our_domain ? 'Our Domain' : 'Not Our Domain'}
                                                        </span>
                                                    </div>
                                                )}
                                                {message.metadata.retrieved_docs && message.metadata.retrieved_docs > 0 && (
                                                    <div className="flex items-center">
                                                        <Database size={12} className="mr-1"/>
                                                        <span>KB Search: {message.metadata.retrieved_docs} documents retrieved</span>
                                                    </div>
                                                )}
                                            </div>
                                        )}
                                    </div>

                                    {message.sender === 'user' && (
                                        <div
                                            className="w-8 h-8 rounded-full bg-gray-300 ml-3 flex items-center justify-center flex-shrink-0">
                                            <User size={16} className="text-gray-600"/>
                                        </div>
                                    )}
                                </div>
                            ))}

                            {/* Processing indicator */}
                            {isProcessing && (
                                <div className="flex justify-start">
                                    <div
                                        className="w-8 h-8 rounded-full bg-gradient-to-br from-blue-500 to-purple-600 mr-3 flex items-center justify-center flex-shrink-0">
                                        <Loader size={16} className="text-white animate-spin"/>
                                    </div>
                                    <div
                                        className="bg-white text-gray-800 border border-gray-200 shadow-sm p-4 rounded-lg">
                                        <div className="flex items-center text-gray-600">
                                            <Loader size={16} className="animate-spin mr-2"/>
                                            <span
                                                className="text-sm">Processing with {selectedModelInfo.description || 'AI model'}...</span>
                                        </div>
                                    </div>
                                </div>
                            )}
                        </div>

                        {/* Input Area */}
                        <div className="px-6 py-4 bg-white border-t border-gray-200">
                            <div className="flex space-x-3">
                                <textarea
                                    value={input}
                                    onChange={(e) => setInput(e.target.value)}
                                    onKeyPress={handleKeyPress}
                                    placeholder={
                                        !isSocketConnected
                                            ? "Connecting to chat service..."
                                            : !isConfigValid
                                                ? "Please configure your API keys first..."
                                                : "Ask me anything..."
                                    }
                                    disabled={isProcessing || !isConfigValid || !isSocketConnected}
                                    className="flex-1 p-3 border border-gray-300 rounded-lg resize-none focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 disabled:bg-gray-100 disabled:text-gray-500"
                                    rows={2}
                                />
                                <button
                                    onClick={sendMessage}
                                    disabled={!input.trim() || isProcessing || !isConfigValid || !isSocketConnected}
                                    className={`px-6 py-3 rounded-lg font-medium transition-colors ${
                                        input.trim() && !isProcessing && isConfigValid && isSocketConnected
                                            ? 'bg-blue-500 text-white hover:bg-blue-600'
                                            : 'bg-gray-300 text-gray-500 cursor-not-allowed'
                                    }`}
                                >
                                    <Send size={18}/>
                                </button>
                            </div>
                        </div>
                    </div>

                    {/* Resizable Step Execution Panel */}
                    {showSteps && (
                        <div
                            className="bg-white border-l border-gray-200 flex flex-col relative"
                            style={{width: `${stepsPanel.width}px`}}
                        >
                            {/* Resize Handle */}
                            <div
                                className="absolute left-0 top-0 bottom-0 w-1 cursor-col-resize hover:bg-blue-300 transition-colors group"
                                onMouseDown={stepsPanel.handleMouseDown}
                            >
                                <div
                                    className="absolute left-0 top-1/2 transform -translate-y-1/2 -translate-x-1 opacity-0 group-hover:opacity-100 transition-opacity">
                                    <GripVertical size={16} className="text-gray-400"/>
                                </div>
                            </div>

                            <div
                                className="px-4 py-3 border-b border-gray-200 bg-gray-50 flex items-center justify-between">
                                <div>
                                    <h3 className="font-semibold text-gray-900 text-sm">Execution Steps</h3>
                                    <p className="text-xs text-gray-500 mt-1">
                                        {currentSteps.length > 0 ? 'Real-time processing steps' : 'Steps will appear here during processing'}
                                    </p>
                                </div>
                                <button
                                    onClick={() => handleShowStepsChange(false)}
                                    className="p-1 hover:bg-gray-200 rounded text-gray-500 hover:text-gray-700"
                                >
                                    <X size={14}/>
                                </button>
                            </div>

                            <div className="flex-1 overflow-y-auto p-4">
                                {currentSteps.length === 0 && !isProcessing && (
                                    <div className="text-center text-gray-500 py-8">
                                        <Clock size={24} className="mx-auto mb-2 opacity-50"/>
                                        <p className="text-sm">No active processing</p>
                                    </div>
                                )}

                                <div className="space-y-3">
                                    {currentSteps.map((step, index) => (
                                        <div
                                            key={`${step.step}-${index}`}
                                            className={`border rounded-lg p-3 ${getStepColor(step.status)}`}
                                        >
                                            <div className="flex items-center justify-between mb-2">
                                                <div className="flex items-center">
                                                    {step.status === 'started' ? (
                                                        <Loader size={16} className="animate-spin mr-2"/>
                                                    ) : step.status === 'error' ? (
                                                        <AlertCircle size={16} className="mr-2"/>
                                                    ) : (
                                                        <div className="mr-2">{getStepIcon(step.step)}</div>
                                                    )}
                                                    <span className="font-medium text-sm">
                                                        {getStepName(step.step)}
                                                    </span>
                                                </div>
                                                {step.elapsed_time && (
                                                    <span className="text-xs opacity-75">
                                                        {step.elapsed_time}
                                                    </span>
                                                )}
                                            </div>

                                            {step.error && (
                                                <div
                                                    className="text-xs mb-2 p-2 bg-red-100 rounded border-l-2 border-red-400">
                                                    <strong>Error:</strong> {step.error}
                                                </div>
                                            )}

                                            {step.data && Object.keys(step.data).length > 0 && (
                                                <div className="text-xs space-y-1">
                                                    {step.data.message && (
                                                        <div><strong>Message:</strong> {step.data.message}</div>
                                                    )}
                                                    {step.data.model && (
                                                        <div><strong>Model:</strong> {step.data.model}</div>
                                                    )}
                                                    {step.data.embedding_type && (
                                                        <div><strong>Embeddings:</strong> {step.data.embedding_type}
                                                        </div>
                                                    )}
                                                    {step.data.query_count && (
                                                        <div><strong>Queries:</strong> {step.data.query_count}</div>
                                                    )}
                                                    {step.data.retrieved_count && (
                                                        <div><strong>Documents:</strong> {step.data.retrieved_count}
                                                        </div>
                                                    )}
                                                    {step.data.answer_length && (
                                                        <div><strong>Answer:</strong> {step.data.answer_length} chars
                                                        </div>
                                                    )}
                                                    {step.data.queries && step.data.queries.length > 0 && (
                                                        <div>
                                                            <strong>Queries:</strong>
                                                            <ul className="ml-2 mt-1">
                                                                {step.data.queries.map((query, i) => (
                                                                    <li key={i} className="flex items-start">
                                                                        <ChevronRight size={12}
                                                                                      className="mt-0.5 mr-1 flex-shrink-0"/>
                                                                        <span>{query}</span>
                                                                    </li>
                                                                ))}
                                                            </ul>
                                                        </div>
                                                    )}
                                                    {step.data.sources && step.data.sources.length > 0 && (
                                                        <div>
                                                            <strong>Sources:</strong> {step.data.sources.join(', ')}
                                                        </div>
                                                    )}
                                                    {step.data.avg_relevance !== undefined && (
                                                        <div><strong>Avg
                                                            Relevance:</strong> {(step.data.avg_relevance * 100).toFixed(1)}%
                                                        </div>
                                                    )}
                                                </div>
                                            )}
                                        </div>
                                    ))}
                                </div>
                            </div>
                        </div>
                    )}

                    {/* WIDER Resizable KB Search Results Panel */}
                    {showKbResults && (
                        <div
                            className="border-l border-gray-200 bg-white relative"
                            style={{width: `${kbPanel.width}px`}}
                        >
                            {/* Resize Handle */}
                            <div
                                className="absolute left-0 top-0 bottom-0 w-1 cursor-col-resize hover:bg-blue-300 transition-colors group"
                                onMouseDown={kbPanel.handleMouseDown}
                            >
                                <div
                                    className="absolute left-0 top-1/2 transform -translate-y-1/2 -translate-x-1 opacity-0 group-hover:opacity-100 transition-opacity">
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
                                        <button
                                            onClick={handleCloseKbResults}
                                            className="p-1 hover:bg-gray-200 rounded text-gray-500 hover:text-gray-700"
                                        >
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

            {/*KB Panel*/}
            {showKB && (
                <div className="fixed inset-0 z-50 flex">
                    <div className="absolute inset-0 bg-transparent backdrop-blur-xs" onClick={hideKB}/>
                    <div className="ml-auto transition-transform h-full w-1/2">
                        <KBPanel onClose={hideKB}/>
                    </div>
                </div>
            )}

            {/*System Monitor Panel*/}
            {showSystemMonitor && (
                <div
                    className="border-l border-gray-200 bg-white relative flex-shrink-0"
                    style={{width: `${monitorPanel.width}px`}}
                >
                    {/* Resize Handle */}
                    <div
                        className="absolute left-0 top-0 bottom-0 w-1 cursor-col-resize hover:bg-blue-300 transition-colors group"
                        onMouseDown={monitorPanel.handleMouseDown}
                    >
                        <div
                            className="absolute left-0 top-1/2 transform -translate-y-1/2 -translate-x-1 opacity-0 group-hover:opacity-100 transition-opacity">
                            <GripVertical size={16} className="text-gray-400"/>
                        </div>
                    </div>

                    <SystemMonitorPanel onClose={toggleSystemMonitor} />
                </div>
            )}
        </div>
    );
};

export default SingleChatApp;