// Chat.tsx
import React, {useCallback, useEffect, useMemo, useRef, useState} from "react";
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
    WifiOff, X
} from "lucide-react";
import {
    ChatCompleteEnvelope,
    ChatDeltaEnvelope,
    ChatErrorEnvelope,
    ChatEventHandlers,
    ChatRequest,
    ChatStartEnvelope,
    ChatStepEnvelope,
    downloadBlob,
    getChatServiceSingleton,
    getResourceByRN,
    getSuggestedQuestions,
    SocketChatOptions,
    UIMessage,
    UseSocketChatReturn,
    WireChatMessage,
} from "./ChatService";

import {useAuthManagerContext} from "../auth/AuthManager";
import {
    getChatBaseAddress,
    getChatSocketAddress,
    getChatSocketSocketIOPath,
    getKBAPIBaseAddress,
    getWorkingScope
} from "../../AppConfig";

import {
    AssistantThinkingItem, BundleInfo,
    ChatLogItem,
    createAssistantChatStep,
    createAssistantThinkingItem,
    createChatMessage,
    createDownloadItem,
    createSourceLinks,
    DownloadItem, EmbedderInfo, ModelInfo,
    RichLink,
    StepUpdate,
} from "./types/chat";
import ChatInterface, {ChatInterfaceContext, ChatInterfaceContextValue} from "./ChatInterface/ChatInterface.tsx";

import {handleContentDownload, openUrlSafely} from "../shared.ts";
import {ChatConfigPanel} from "./config/ChatConfigPanel.tsx";
import {ConfigProvider, useConfigProvider} from "./ChatConfigProvider.tsx";
import KBPanel from "../kb/KBPanel.tsx";
import {SystemMonitorPanel} from "../monitoring/monitoring.tsx";
import EnhancedKBSearchResults from "./SearchResults.tsx";

/* ============================
   Local Socket.IO hook (v1)
   ============================ */

export function useSocketChat(options: SocketChatOptions): UseSocketChatReturn {
    const [isConnected, setIsConnected] = useState(false);
    const [isConnecting, setIsConnecting] = useState(false);
    const [socketId, setSocketId] = useState<string | undefined>(undefined);
    const [connectionError, setConnectionError] = useState<string | null>(null);

    const authContext = useAuthManagerContext();

    const stableOpts = useMemo<SocketChatOptions>(
        () => ({
            baseUrl: options.baseUrl,
            path: options.path ?? "/socket.io",
            reconnectionAttempts: options.reconnectionAttempts ?? 10,
            timeout: options.timeout ?? 10000,
            project: options.project,
            tenant: options.tenant,
            namespace: options.namespace ?? "/",
            authContext,
        }),
        [options.baseUrl, options.path, options.reconnectionAttempts, options.timeout, options.project, options.tenant, options.namespace]
    );

    const service = useMemo(() => getChatServiceSingleton(stableOpts), [stableOpts]);

    const connect = useCallback(
        async (handlers: ChatEventHandlers, ac = authContext) => {
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
        },
        [service, authContext]
    );

    const disconnect = useCallback(() => {
        service.disconnect();
        setIsConnected(false);
        setIsConnecting(false);
        setSocketId(undefined);
        setConnectionError(null);
    }, [service]);

    const sendMessage = useCallback(
        (request: ChatRequest, attachments?: File[]) => {
            if (!service.connected) throw new Error("Not connected to chat service");
            service.sendChatMessage(request, attachments);
        },
        [service]
    );

    const ping = useCallback(() => {
        if (!service.connected) throw new Error("Not connected to chat service");
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

interface ChatMessage {
    id: number;
    sender: "user" | "assistant";
    text: string;
    timestamp: Date;
    isError?: boolean;
    attachments?: File[] //only relevant for user message
    metadata?: {
        turn_id?: string;
    };
}

/* ==========
   Component
   ========== */

const SingleChatApp: React.FC = () => {
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
        sendMessage: sendSocketMessage
    } =
        useSocketChat({
            baseUrl: getChatSocketAddress(),
            path: getChatSocketSocketIOPath(),
            authContext,
            project,
            tenant,
            reconnectionAttempts: Infinity
        });

    // Client turn id (temporary until server echoes it back)
    const activeTurnIdRef = useRef<string | null>(null);

    // Messages state — greeting pinned to epoch so it’s always the first item.
    const [messages, setMessages] = useState<ChatMessage[]>([
        {
            id: 1,
            sender: "assistant",
            text: "Hello! I'm your AI assistant application and currently under active development.",
            timestamp: new Date(0),
            metadata: {},
        }
    ])

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
    const [followUpQuestion, setFollowUpQuestion] = useState<string[]>([]);

    // Thinking (per-turn container, per-agent rows)
    const thinkingItemIdRef = useRef<number | null>(null);
    const thinkingBufferRef = useRef<Record<string, string>>({});
    const thinkingFlushTimerRef = useRef<Record<string, number | null>>({});
    const [thinkingItems, setThinkingItems] = useState<AssistantThinkingItem[]>([]);

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
            Object.values(thinkingFlushTimerRef.current || {}).forEach((t) => {
                if (t != null) window.clearTimeout(t as any);
            });
            thinkingFlushTimerRef.current = {};
        }
    }, []);

    // Connect Socket.IO
    const didConnectRef = useRef(false);
    useEffect(() => {
        if (didConnectRef.current) return;
        didConnectRef.current = true;

        let cancelled = false;

        const waitForToken = async (timeoutMs = 2000, intervalMs = 100) => {
            const start = Date.now();
            while (!cancelled) {
                if (authContext?.getUserAuthToken?.()) return true;
                if (Date.now() - start >= timeoutMs) return false;
                await new Promise(r => setTimeout(r, intervalMs));
            }
            return false;
        };

        (async () => {
            try {
                await waitForToken();
                await connectSocket(chatEventHandlers, authContext);
            } catch (e) {
                console.error("Failed to initialize socket:", e);
                setTimeout(() => connectSocket(chatEventHandlers, authContext).catch(console.error), 750);
            }
        })();

        return () => {
            cancelled = true;
            disconnectSocket();
            didConnectRef.current = false;
        };
    }, []);

    // Suggested questions
    const [updatingQustions, setUpdatingQustions] = useState<boolean>(false);
    const [quickQuestions, setQuickQuestions] = useState<string[]>([]);
    useEffect(() => {
        setUpdatingQustions(true);
        getSuggestedQuestions(tenant, project, authContext, headerBundle?.id)
            .then(setQuickQuestions)
            .catch(console.error)
            .finally(() => setUpdatingQustions(false));
    }, [project, tenant, headerBundle]);


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
            console.error("Logout error:", e);
        }
    }, [disconnectSocket, authContext]);

    /* ===== helpers ===== */

    // When answer streaming begins or complete/error, end the global thinking (header).
    // Do NOT force per-agent end times unless we received completed: true for that agent.
    const deactivateThinking = (turnId?: string, ts?: string) => {
        const ended = new Date(Date.parse(ts || new Date().toISOString()));
        setThinkingItems((prev) =>
            prev.map((it) => {
                if (it.id !== thinkingItemIdRef.current) return it;

                // Mark the overall thinking item as inactive and set its overall endedAt.
                // Keep per-agent endedAt untouched unless it was already set by completed: true.
                const agentTimes = {...(it.agentTimes || {})};
                Object.keys(agentTimes).forEach((k) => {
                    const rec = agentTimes[k];
                    if (rec.active) {
                        agentTimes[k] = {...rec, active: false}; // ← leave rec.endedAt as-is (possibly undefined)
                    }
                });

                return new AssistantThinkingItem(
                    it.id,
                    it.timestamp,
                    it.turn_id ?? turnId,
                    false,
                    it.endedAt ?? ended,
                    {...(it.agents || {})},
                    agentTimes
                );
            })
        );
    };

    const flushBuffered = useCallback(() => {
        if (!deltaBufferRef.current) return;
        const chunk = deltaBufferRef.current;
        deltaBufferRef.current = "";
        setMessages((prev) => {
            const msgId = streamingMsgIdRef.current;
            if (msgId == null) return prev;
            const idx = prev.findIndex((m) => m.id === msgId);
            if (idx === -1) return prev;
            const updated = [...prev];
            const current = updated[idx];
            const safeText = typeof current.text === "string" ? current.text : "";
            updated[idx] = {...current, text: safeText + chunk};
            return updated;
        });
        flushTimerRef.current = null;
    }, []);

    const flushThinkingBuffered = (agent: string) => {
        const chunk = thinkingBufferRef.current[agent];
        if (!chunk || thinkingItemIdRef.current == null) return;
        thinkingBufferRef.current[agent] = "";
        setThinkingItems((prev) =>
            prev.map((it) => {
                if (it.id !== thinkingItemIdRef.current) return it;
                const nextAgents = {...(it.agents || {})};
                nextAgents[agent] = (nextAgents[agent] || "") + chunk;
                return new AssistantThinkingItem(
                    it.id,
                    it.timestamp,
                    it.turn_id,
                    it.active,
                    it.endedAt,
                    nextAgents,
                    {...(it.agentTimes || {})}
                );
            })
        );
        if (thinkingFlushTimerRef.current[agent] != null) {
            window.clearTimeout(thinkingFlushTimerRef.current[agent] as any);
        }
        thinkingFlushTimerRef.current[agent] = null;
    };

    const upsertThinkingItem = (agent?: string, turnId?: string, ts?: string) => {
        const agentKey = agent || "agent";
        if (thinkingItemIdRef.current == null) {
            const id = Date.now();
            thinkingItemIdRef.current = id;
            const startedAt = new Date(Date.parse(ts || new Date().toISOString()));
            setThinkingItems((prev) => {
                if (prev.some((p) => p.id === id)) return prev;
                return [
                    ...prev,
                    createAssistantThinkingItem({
                        id,
                        timestamp: startedAt,
                        turn_id: turnId,
                        initialAgents: agent ? {[agentKey]: ""} : {},
                        initialAgentTimes: agent ? {[agentKey]: {startedAt, active: true}} : {},
                    }),
                ];
            });
        } else if (agentKey) {
            // ensure the agent row exists and timing is initialized on FIRST chunk
            const startedAt = new Date(Date.parse(ts || new Date().toISOString()));
            setThinkingItems((prev) =>
                prev.map((it) => {
                    if (it.id !== thinkingItemIdRef.current) return it;
                    const nextAgents = {...(it.agents || {})};
                    const nextTimes = {...(it.agentTimes || {})};
                    if (!Object.prototype.hasOwnProperty.call(nextAgents, agentKey)) {
                        nextAgents[agentKey] = "";
                    }
                    if (!Object.prototype.hasOwnProperty.call(nextTimes, agentKey)) {
                        nextTimes[agentKey] = {startedAt, active: true};
                    }
                    return new AssistantThinkingItem(
                        it.id,
                        it.timestamp,
                        it.turn_id,
                        it.active,
                        it.endedAt,
                        nextAgents,
                        nextTimes
                    );
                })
            );
        }
    };

    const reconcileTurnId = useCallback((serverTurnId?: string) => {
        if (!serverTurnId) return;
        const prevId = activeTurnIdRef.current;
        if (!prevId || prevId === serverTurnId) {
            activeTurnIdRef.current = serverTurnId;
            return;
        }
        setMessages((prev) => prev.map((m) => (m.metadata?.turn_id === prevId ? {
            ...m,
            metadata: {...(m.metadata || {}), turn_id: serverTurnId}
        } : m)));
        setCurrentSteps((prev) => prev.map((s) => (s.turn_id === prevId ? {...s, turn_id: serverTurnId} : s)));
        setThinkingItems((prev) =>
            prev.map((t) =>
                t.turn_id === prevId
                    ? new AssistantThinkingItem(
                        t.id,
                        t.timestamp,
                        serverTurnId,
                        t.active,
                        t.endedAt,
                        {...(t.agents || {})},
                        {...(t.agentTimes || {})}
                    )
                    : t
            )
        );
        activeTurnIdRef.current = serverTurnId;
    }, []);

    /* =========================
       Socket event handlers (v1)
       ========================= */

    const chatEventHandlers: ChatEventHandlers = useMemo(
        () => ({
            onConnect: () => {
            },
            onSessionInfo: (info) => {
                console.log("Server session:", info.session_id, info.user_type);
            },
            onDisconnect: (reason: string) => {
                console.log("Disconnected:", reason);
                setIsProcessing(false);
                if (flushTimerRef.current) {
                    window.clearTimeout(flushTimerRef.current);
                    flushTimerRef.current = null;
                }
                Object.values(thinkingFlushTimerRef.current || {}).forEach((t) => {
                    if (t != null) window.clearTimeout(t as any);
                });
                thinkingFlushTimerRef.current = {};
                thinkingBufferRef.current = {};
                deltaBufferRef.current = "";
                streamingMsgIdRef.current = null;
                sawFirstDeltaRef.current = false;
            },
            onConnectError: (error: Error) => {
                console.error("Connect error:", error);
                setIsProcessing(false);

                const msg = (error?.message || "").toLowerCase();
                const looksAuthy =
                    msg.includes("401") ||
                    msg.includes("unauthorized") ||
                    msg.includes("forbidden") ||
                    msg.includes("rejected by server");

                if (looksAuthy) {
                    // Token is likely expired or invalid for this server — force re-auth.
                    try {
                        authContext.logout?.();
                    } catch (e) {
                        console.error("Failed to initiate re-auth:", e);
                    }
                }
            },

            onChatStart: (env: ChatStartEnvelope) => {
                console.log("chat.start:", env);
                const turnId = env.conversation?.turn_id;
                if (turnId) reconcileTurnId(turnId);
                sawFirstDeltaRef.current = false;
                deltaBufferRef.current = "";

                // reset thinking buffers every new turn
                thinkingItemIdRef.current = null;
                thinkingBufferRef.current = {};
                Object.values(thinkingFlushTimerRef.current || {}).forEach((t) => {
                    if (t != null) clearTimeout(t as any);
                });
                thinkingFlushTimerRef.current = {};
            },

            onChatDelta: (env: ChatDeltaEnvelope) => {
                // strict v1 fields
                const turnId = env.conversation?.turn_id ?? activeTurnIdRef.current ?? undefined;
                if (env.conversation?.turn_id) reconcileTurnId(env.conversation.turn_id);

                const marker = env.delta?.marker ?? "answer";
                const chunkText = env.delta?.text ?? "";
                const ts = env.timestamp ?? new Date().toISOString();
                const agent = (env.event?.agent ?? "").toString();
                const completed = (env.delta as any)?.completed === true; // ← your server flag

                if (!sawFirstDeltaRef.current) {
                    sawFirstDeltaRef.current = true;
                    setIsProcessing(false);
                }

                if (marker === "thinking") {
                    const agentKey = agent || "agent";
                    upsertThinkingItem(agentKey, turnId, ts);

                    // buffer per agent
                    thinkingBufferRef.current[agentKey] = (thinkingBufferRef.current[agentKey] || "") + chunkText;

                    // schedule per-agent flush
                    if (thinkingFlushTimerRef.current[agentKey] == null) {
                        thinkingFlushTimerRef.current[agentKey] = window.setTimeout(() => {
                            flushThinkingBuffered(agentKey);
                        }, 24) as unknown as number;
                    }

                    // If this chunk marks agent completion, stamp endedAt (single shot) and inactivate that agent.
                    if (completed) {
                        const endedAt = new Date(Date.parse(ts || new Date().toISOString()));
                        setThinkingItems((prev) =>
                            prev.map((it) => {
                                if (it.id !== thinkingItemIdRef.current) return it;
                                const nextTimes = {...(it.agentTimes || {})};
                                const rec = nextTimes[agentKey] || {startedAt: endedAt, active: false};
                                // Only set endedAt once
                                nextTimes[agentKey] = rec.endedAt
                                    ? {...rec, active: false}
                                    : {...rec, endedAt, active: false};
                                return new AssistantThinkingItem(
                                    it.id,
                                    it.timestamp,
                                    it.turn_id ?? turnId,
                                    it.active,
                                    it.endedAt,
                                    {...(it.agents || {})},
                                    nextTimes
                                );
                            })
                        );
                    }
                    return;
                }

                // answer streaming — finalize overall thinking (per the global header)
                deactivateThinking(turnId, ts);

                if (streamingMsgIdRef.current == null) {
                    setMessages((prev) => {
                        const last = prev[prev.length - 1];
                        if (last && last.sender === "assistant" && (last.text ?? "") === "" && !last.isError) {
                            streamingMsgIdRef.current = last.id;
                            return prev;
                        }
                        const id = Date.now();
                        streamingMsgIdRef.current = id;
                        return [
                            ...prev,
                            {
                                id,
                                sender: "assistant",
                                text: "",
                                timestamp: new Date(Date.parse(ts || new Date().toISOString())),
                                metadata: {turn_id: turnId || undefined},
                            },
                        ];
                    });
                }

                deltaBufferRef.current += chunkText;
                if (flushTimerRef.current == null) {
                    flushTimerRef.current = window.setTimeout(() => {
                        flushBuffered();
                    }, 24) as unknown as number;
                }
            },

            onChatStep: (env: ChatStepEnvelope) => {
                // console.log("chat.step:", env);
                const serverTid = env.conversation?.turn_id;
                if (serverTid) reconcileTurnId(serverTid);

                const stepUpdate: StepUpdate = {
                    step: env.event?.step,
                    status: env.event?.status as any,
                    title: env.event?.title,
                    timestamp: new Date(Date.parse(env.timestamp || new Date().toISOString())),
                    elapsed_time: (env as any).elapsed_time, // optional, not in v1 spec
                    error: env.data?.error,
                    data: env.data,
                    markdown: (env.event as any)?.markdown,
                    agent: (env.event as any)?.agent,
                    turn_id: serverTid ?? activeTurnIdRef.current ?? undefined,
                };

                setCurrentSteps((prev) => {
                    const existing = prev.find(
                        (s) => s.step === stepUpdate.step && s.turn_id === stepUpdate.turn_id
                    );
                    return existing
                        ? prev.map((s) =>
                            s.step === stepUpdate.step && s.turn_id === stepUpdate.turn_id ? stepUpdate : s
                        )
                        : [...prev, stepUpdate];
                });

                if (env.event?.step === "followups" && env.event?.status === "completed") {
                    setFollowUpQuestion(env.data?.items || []);
                }
            },

            onChatComplete: (env: ChatCompleteEnvelope) => {
                console.log("chat.complete:", env);
                if (flushTimerRef.current != null) {
                    window.clearTimeout(flushTimerRef.current);
                    flushTimerRef.current = null;
                }
                if (deltaBufferRef.current) flushBuffered();

                const serverTid = env.conversation?.turn_id;
                if (serverTid) reconcileTurnId(serverTid);

                const finalText = env.data?.final_answer ?? "";
                const ts = env.timestamp ?? new Date().toISOString();

                const msgId = streamingMsgIdRef.current;
                setMessages((prev) => {
                    if (msgId != null) {
                        const idx = prev.findIndex((m) => m.id === msgId);
                        if (idx !== -1) {
                            const updated = [...prev];
                            const current = updated[idx];
                            updated[idx] = {
                                ...current,
                                text: finalText,
                                timestamp: new Date(Date.parse(ts)),
                                metadata: {
                                    ...(current.metadata || {}),
                                    turn_id: serverTid ?? current.metadata?.turn_id,
                                },
                            };
                            return updated;
                        }
                    }
                    return [
                        ...prev,
                        {
                            id: Date.now() + 1,
                            sender: "assistant",
                            text: finalText,
                            timestamp: new Date(Date.parse(ts)),
                            metadata: {
                                turn_id: serverTid ?? activeTurnIdRef.current ?? undefined,
                            },
                        },
                    ];
                });

                // finalize global thinking (do not force per-agent endedAt)
                deactivateThinking(serverTid, ts);

                streamingMsgIdRef.current = null;
                deltaBufferRef.current = "";
                sawFirstDeltaRef.current = false;
                setIsProcessing(false);
            },

            onChatError: (env: ChatErrorEnvelope) => {
                console.log("chat.error:", env);
                if (flushTimerRef.current != null) {
                    window.clearTimeout(flushTimerRef.current);
                    flushTimerRef.current = null;
                }
                const ts = env.timestamp ?? new Date().toISOString();
                deactivateThinking(env.conversation?.turn_id, ts);

                deltaBufferRef.current = "";
                streamingMsgIdRef.current = null;
                sawFirstDeltaRef.current = false;

                const errText = env.data?.error ? String(env.data.error) : "Unknown error";
                setMessages((prev) => [
                    ...prev,
                    {
                        id: Date.now() + 1,
                        sender: "assistant",
                        text: `I encountered an error: ${errText}`,
                        timestamp: new Date(Date.parse(ts)),
                        isError: true,
                        metadata: {turn_id: env.conversation?.turn_id ?? activeTurnIdRef.current ?? undefined},
                    },
                ]);
                setIsProcessing(false);
            },
        }),
        [flushBuffered, reconcileTurnId]
    );

    /* ========================
       Send message (with turn)
       ======================== */

    const sendMessage = useCallback(
        async (message: string, attachments?: File[]): Promise<void> => {
            if ((!message.trim() && !attachments?.length) || isProcessing) return;

            const clientTurnId = `turn_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
            activeTurnIdRef.current = clientTurnId;

            const userMessage: ChatMessage = {
                id: Date.now(),
                sender: "user",
                text: message.trim(),
                timestamp: new Date(),
                metadata: {turn_id: clientTurnId},
                attachments: attachments
            };
            setMessages((prev) => [...prev, userMessage]);
            setIsProcessing(true);
            setFollowUpQuestion([]);

            const toWire = (msgs: UIMessage[]): WireChatMessage[] =>
                msgs
                    .filter((m) => m.sender === "user" || m.sender === "assistant")
                    .map((m) => ({
                        role: m.sender,
                        content: m.text,
                        timestamp: m.timestamp.toISOString(),
                        id: (m as any).id
                    }));

            try {
                const history = toWire(messages);
                const payload: ChatRequest = {
                    message: userMessage.text,
                    chat_history: history,
                    project,
                    tenant,
                    turn_id: clientTurnId,
                    bundle_id: headerBundle?.id
                };
                sendSocketMessage(payload, attachments);
            } catch (error) {
                console.error("Error sending message via socket:", error);
                setMessages((prev) => [
                    ...prev,
                    {
                        id: Date.now() + 1,
                        sender: "assistant",
                        text: `I couldn't send your message: ${(error as Error).message}`,
                        timestamp: new Date(),
                        isError: true,
                        metadata: {turn_id: clientTurnId},
                    },
                ]);
                setIsProcessing(false);
            }
        },
        [isProcessing, sendSocketMessage, messages, project, tenant, headerBundle]
    );

    /* ===========================
       Grouping for ChatInterface
       =========================== */

    // UI helpers
    const hideKB = () => setShowKB(false);
    const toggleSystemMonitor = () => setShowSystemMonitor(prev => !prev);

    const chatLogItems: ChatLogItem[] = useMemo(() => {
        const items: ChatLogItem[] = [];
        const toItem = (m: ChatMessage) => createChatMessage(m);
        const steps = [...currentSteps];
        const allMessages = [...messages];

        const greetings = allMessages.filter((m) => m.sender === "assistant" && m.timestamp.getTime() === 0);
        greetings.forEach((g) => items.push(toItem(g)));

        const userMsgs = allMessages.filter((m) => m.sender === "user").sort((a, b) => a.timestamp.getTime() - b.timestamp.getTime());

        const nextUserOrInfinity = (idx: number) => (idx + 1 < userMsgs.length ? userMsgs[idx + 1].timestamp.getTime() : Number.POSITIVE_INFINITY);

        userMsgs.forEach((uMsg, idx) => {
            const tid = uMsg.metadata?.turn_id;
            const startT = uMsg.timestamp.getTime();
            const endT = nextUserOrInfinity(idx);

            items.push(toItem(uMsg));

            const turnSteps = steps
                .filter((s) => (s.turn_id ? s.turn_id === tid : s.timestamp.getTime() >= startT && s.timestamp.getTime() < endT))
                .sort((a, b) => a.timestamp.getTime() - b.timestamp.getTime());
            turnSteps.forEach((s) => items.push(createAssistantChatStep(s)));

            const turnThinking = thinkingItems
                .filter((t) => (t.turn_id ? t.turn_id === tid : t.timestamp.getTime() >= startT && t.timestamp.getTime() < endT))
                .sort((a, b) => a.timestamp.getTime() - b.timestamp.getTime());
            turnThinking.forEach((t) => items.push(t));

            const downloadItems = steps
                .filter((s) => ((s.turn_id ? s.turn_id === tid : s.timestamp.getTime() >= startT
                        && s.timestamp.getTime() < endT)
                    && s.step === "file"
                    && s.status === "completed"
                    && !!s.data?.rn
                    && !!s.data?.filename

                ))
                .sort((a, b) => a.timestamp.getTime() - b.timestamp.getTime())
                // .map(createDownloadItem)
                .map((s => {
                    //console.log("DOWNLOAD ITEM", s);
                    return createDownloadItem(s);
                }));
            items.push(...downloadItems);

            const sourceItems = steps
                .filter((s) => ((s.turn_id ? s.turn_id === tid : s.timestamp.getTime() >= startT
                        && s.timestamp.getTime() < endT)
                    && s.step === "citations"
                    && s.status === "completed"
                    && !!s.data?.count
                    && !!s.data?.items

                ))
                .sort((a, b) => a.timestamp.getTime() - b.timestamp.getTime())
                // .map(createDownloadItem)
                .map((s => {
                    //console.log("DOWNLOAD ITEM", s);
                    return createSourceLinks(s);
                }));
            items.push(...sourceItems);

            const assistantMsgs = allMessages
                .filter(
                    (m) =>
                        m.sender === "assistant" &&
                        m.timestamp.getTime() !== 0 &&
                        (m.metadata?.turn_id ? m.metadata?.turn_id === tid : m.timestamp.getTime() >= startT && m.timestamp.getTime() < endT)
                )
                .sort((a, b) => a.timestamp.getTime() - b.timestamp.getTime());
            assistantMsgs.forEach((m) => items.push(toItem(m)));
        });

        const firstUserTime = userMsgs[0]?.timestamp.getTime() ?? Number.POSITIVE_INFINITY;
        const strayStepItems = steps
            .filter((s) => s.timestamp.getTime() < firstUserTime && !s.turn_id)
            .sort((a, b) => a.timestamp.getTime() - b.timestamp.getTime())
            .map(createAssistantChatStep);
        items.push(...strayStepItems);

        const strayAssistants = allMessages
            .filter((m) => m.sender === "assistant" && m.timestamp.getTime() > 0 && !m.metadata?.turn_id && m.timestamp.getTime() < firstUserTime)
            .sort((a, b) => a.timestamp.getTime() - b.timestamp.getTime())
            .map(toItem);
        items.push(...strayAssistants);

        return items;
    }, [messages, currentSteps, thinkingItems]);

    /* ===== UI chrome ===== */

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

    const onDownloadItemClick = (item: DownloadItem) => {
        const fn = async () => {
            const resource = await getResourceByRN(item.rn, authContext)
            const download_url = resource.metadata.download_url
            const data = await downloadBlob(download_url, authContext)
            handleContentDownload(item.filename, data, item.mimeType || "application/octet-stream")
        }
        fn()
    }

    const onLinkItemClick = (item: RichLink) => {
        openUrlSafely(item.url)
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

    const chatContextValue: ChatInterfaceContextValue = {
        chatLogItems: chatLogItems,
        onSendMessage: sendMessage,
        onDownloadItemClick: onDownloadItemClick,
        userInputEnabled: isSocketConnected,
        isProcessing: isProcessing,
        followUpQuestion: followUpQuestion
    }

    return (
        <div id={SingleChatApp.name} className="flex h-screen bg-slate-100">
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
                {/*{renderSimpleHeader()}*/}
                {renderFullHeader()}

                {/* Body: Chat + optionally Steps / KB Results / System Monitor */}
                <div className={`flex-1 flex overflow-hidden transition-all duration-300`}>
                    {/* Chat Column */}
                    <div className={`flex-1 flex flex-col ${showSystemMonitor ? 'mr-4' : ''}`} ref={chatContainerRef}>
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

                        <ChatInterfaceContext value={chatContextValue}>
                            <ChatInterface maxWidth={fullChatWidth * (3 / 5)}/>
                        </ChatInterfaceContext>
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
