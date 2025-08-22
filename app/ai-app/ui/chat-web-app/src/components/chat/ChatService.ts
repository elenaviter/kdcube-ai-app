/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2025 Elena Viter
 */

import { Manager, Socket } from 'socket.io-client';
import {AppConfig} from './ChatConfigProvider';
import {getChatBaseAddress} from "../../AppConfig.ts";
import {AuthContextValue} from "../auth/AuthManager.tsx";

let __chatSingleton: SocketChatService | null = null;

export function getChatServiceSingleton(opts: SocketChatOptions): SocketChatService {
    if (!__chatSingleton) __chatSingleton = new SocketChatService(opts);
    else __chatSingleton.updateOptions(opts);
    (window as any).__chatSvc = __chatSingleton; // handy for debugging
    return __chatSingleton;
}

// Type definitions for chat events
export interface ChatStepData {
    step: string;
    status: 'started' | 'completed' | 'error';
    timestamp: string;
    data?: Record<string, any>;
    elapsed_time?: string;
    error?: string;
}

export interface ChatStartData {
    message: string;
    timestamp: string;
}

export interface ChatDeltaData {
    task_id: string;
    delta: string;
    index: number;
    timestamp: string
}

export interface ChatCompleteData {
    final_answer: string;
    is_our_domain?: boolean;
    classification_reasoning?: string;
    rag_queries?: any[];
    retrieved_docs?: any[];
    reranked_docs?: any[];
    error_message?: string;
    selected_model: string;
    config_info: Record<string, any>;
    timestamp: string;
}

export interface ChatErrorData {
    error: string;
    timestamp: string;
}

export interface ChatMessage {
    role: 'user' | 'assistant';
    content: string;
    timestamp?: string;
    id: number;
}

export type WireChatMessage = ChatMessage;

export type UIMessage = {
    id: number;
    sender: 'user' | 'assistant';
    text: string;
    timestamp: string;
    isError?: boolean;
    metadata?: any;
};

export interface ChatRequest {
    message: string;
    chat_history: WireChatMessage[];
    config: AppConfig;
    project?: string;
    tenant?: string;
}

// Event handler types
export interface ChatEventHandlers {
    onConnect?: () => void;
    onDisconnect?: (reason: string) => void;
    onConnectError?: (error: Error) => void;
    onChatStart?: (data: ChatStartData) => void;
    onChatDelta?:  (data: ChatDeltaData) => void;
    onChatStep?: (data: ChatStepData) => void;
    onChatComplete?: (data: ChatCompleteData) => void;
    onChatError?: (data: ChatErrorData) => void;
    onPong?: (data: { timestamp: string }) => void;
    onSessionInfo?: (info: { session_id: string; user_type: string }) => void;
}

// Connection options
export interface SocketChatOptions {
    baseUrl: string;             // origin only (e.g. http://localhost:5005)
    path?: string;               // defaults to '/socket.io'
    reconnectionAttempts?: number;
    timeout?: number;
    project?: string;
    tenant?: string;
    namespace?: string;          // defaults to '/'
    authContext: AuthContextValue;
}


type EngineKey = string;
const managers = new Map<EngineKey, Manager>();

function getManager(baseUrl: string, path = '/socket.io', opts: Partial<Manager['opts']> = {}): Manager {
    const key: EngineKey = `${baseUrl}|${path}`;
    let m = managers.get(key);
    if (!m) {
        m = new Manager(baseUrl, {
            path,
            transports: ['websocket', 'polling'], // no polling -> avoids “two lines” in WS tab
            upgrade: false,            // no upgrade dance
            autoConnect: false,
            timeout: 10000,
            reconnectionAttempts: 5,
            withCredentials: true,
            ...opts,
        });
        managers.set(key, m);
    } else {
        Object.assign(m.opts, opts);
    }
    return m;
}


// ---------- types ----------
export interface ChatStepData { step: string; status: 'started'|'completed'|'error'; timestamp: string; data?: Record<string, any>; elapsed_time?: string; error?: string; }
export interface ChatStartData { message: string; timestamp: string; }
export interface ChatCompleteData { final_answer: string; is_our_domain?: boolean; classification_reasoning?: string; rag_queries?: any[]; retrieved_docs?: any[]; reranked_docs?: any[]; error_message?: string; selected_model: string; config_info: Record<string, any>; timestamp: string; }
export interface ChatErrorData { error: string; timestamp: string; }

export interface ChatRequest { message: string; chat_history: ChatMessage[]; config: AppConfig; project?: string; tenant?: string; }

export interface ChatEventHandlers {
    onConnect?: () => void;
    onDisconnect?: (reason: string) => void;
    onConnectError?: (error: Error) => void;
    onChatStart?: (data: ChatStartData) => void;
    onChatDelta?:  (data: ChatDeltaData) => void;
    onChatStep?: (data: ChatStepData) => void;
    onChatComplete?: (data: ChatCompleteData) => void;
    onChatError?: (data: ChatErrorData) => void;
    onPong?: (data: { timestamp: string }) => void;
    onSessionInfo?: (info: { session_id: string; user_type: string }) => void;
}


// ---------- service ----------
export class SocketChatService {
    private readonly baseUrl: string;
    private options: Required<SocketChatOptions>;
    private manager: Manager;
    private socket: Socket;
    private isConnecting = false;
    private isConnected = false;
    private eventHandlers: any = {};
    private currentSessionId?: string;
    // de-dupe concurrent connect calls
    private connectingPromise: Promise<Socket> | null = null;

    constructor(options: SocketChatOptions) {
        this.baseUrl = options.baseUrl;
        this.options = {
            baseUrl: this.baseUrl,
            path: options.path ?? '/socket.io',
            reconnectionAttempts: options.reconnectionAttempts ?? 5,
            timeout: options.timeout ?? 10000,
            project: options.project,
            tenant: options.tenant,
            namespace: options.namespace ?? '/', // ← default to ROOT NAMESPACE
        };
        this.manager = getManager(this.baseUrl, this.options.path, {
            reconnectionAttempts: this.options.reconnectionAttempts,
            timeout: this.options.timeout,
        });
        this.socket = this.manager.socket(this.options.namespace, { auth: {} });
    }

    public updateOptions(next: SocketChatOptions) {
        this.options = {
            ...this.options,
            ...next,
            path: next.path ?? this.options.path,
            reconnectionAttempts: next.reconnectionAttempts ?? this.options.reconnectionAttempts,
            timeout: next.timeout ?? this.options.timeout,
            namespace: next.namespace ?? this.options.namespace,
        };
        this.manager = getManager(this.baseUrl, this.options.path, {
            reconnectionAttempts: this.options.reconnectionAttempts,
            timeout: this.options.timeout,
        });
        // IMPORTANT: if you ever change namespace at runtime, recreate the socket:
        if (this.socket.nsp !== this.options.namespace) {
            this.socket.off();
            this.socket.disconnect();
            this.socket = this.manager.socket(this.options.namespace, { auth: {} });
        }
    }

    private bindHandlers() {
        this.socket.off(); // remove previous bindings

        this.socket.on('connect', () => {
            this.isConnected = true;
            this.isConnecting = false;
            console.log('✅ Chat Socket.IO connected:', this.socket.id);
            this.eventHandlers.onConnect?.();
        });
        this.socket.on('disconnect', (reason: string) => {
            this.isConnected = false;
            console.log('⚠️ Chat Socket.IO disconnected:', reason);
            this.eventHandlers.onDisconnect?.(reason);
        });
        this.socket.on('connect_error', (err: any) => {
            this.isConnecting = false;
            console.log('❌ Chat Socket.IO connect error:', err);
            this.eventHandlers.onConnectError?.(err instanceof Error ? err : new Error(String(err?.message || err)));
        });

        this.socket.on('chat_start', (d) => this.eventHandlers.onChatStart?.(d));
        this.socket.on('chat_delta', (d) => this.eventHandlers.onChatDelta?.(d));
        this.socket.on('chat_step', (d) => this.eventHandlers.onChatStep?.(d));
        this.socket.on('chat_complete', (d) => this.eventHandlers.onChatComplete?.(d));
        this.socket.on('chat_error', (d) => this.eventHandlers.onChatError?.(d));
        this.socket.on('pong', (d) => this.eventHandlers.onPong?.(d));
        this.socket.on('session_info', (info: any) => {
            if (info?.session_id) this.currentSessionId = info.session_id;
            this.eventHandlers.onSessionInfo?.({ session_id: info?.session_id, user_type: info?.user_type });
        });
    }

    private async fetchProfile(authContext: AuthContextValue): Promise<{ session_id: string; user_type: string }> {
        const headers: HeadersInit = [
            ['Content-Type', 'application/json']
        ];
        // config
        authContext.appendAuthHeader(headers);
        const r = await fetch(`${getChatBaseAddress()}/profile`, { headers, credentials: 'include' as RequestCredentials });
        if (!r.ok) throw new Error(`Profile fetch failed (${r.status})`);
        const j = await r.json();
        if (!j.session_id) throw new Error('Profile missing session_id');
        this.currentSessionId = j.session_id;
        return { session_id: j.session_id, user_type: j.user_type };
    }

    public async connect(handlers: any = {}, authContext: AuthContextValue): Promise<Socket> {
        if (this.socket.connected) {
            this.eventHandlers = handlers;
            this.bindHandlers();
            return this.socket;
        }
        if (this.connectingPromise) {
            this.eventHandlers = handlers;
            this.bindHandlers();
            return this.connectingPromise;
        }

        this.isConnecting = true;
        this.eventHandlers = handlers;

        this.connectingPromise = (async () => {
            const { session_id } = await this.fetchProfile(authContext);

            const authPayload: any = {
                user_session_id: session_id,
                project: this.options.project,
                tenant: this.options.tenant,
            };
            if (authContext && authContext.getUserAuthToken()) {
                authPayload.bearer_token = authContext.getUserAuthToken();
                authPayload.id_token = authContext.getUserIdToken();
            }

            // make both auth and query available to the server
            this.socket.auth = authPayload;
            (this.socket.io as any).opts.query = { ...(this.socket.io as any).opts?.query, ...authPayload };

            this.bindHandlers();
            this.socket.connect();

            const t = setTimeout(() => {
                if (this.isConnecting) {
                    this.isConnecting = false;
                    this.socket.disconnect();
                }
            }, this.options.timeout);

            // Wait until either connect or connect_error fires once
            await new Promise<void>((resolve, reject) => {
                const ok = () => { clearTimeout(t); this.socket.off('connect_error', bad); resolve(); };
                const bad = (e: any) => { clearTimeout(t); this.socket.off('connect', ok); reject(e instanceof Error ? e : new Error(String(e?.message || e))); };
                this.socket.once('connect', ok);
                this.socket.once('connect_error', bad);
            });

            this.connectingPromise = null;
            return this.socket;
        })();

        return this.connectingPromise;
    }

    public disconnect() {
        this.socket.off();
        this.socket.disconnect();
        this.isConnected = false;
        this.isConnecting = false;
        this.connectingPromise = null;
    }

    public sendChatMessage(req: { message: string; chat_history: any[]; config: AppConfig; project?: string; tenant?: string; }) {
        if (!this.socket.connected) throw new Error('Socket not connected. Call connect() first.');
        this.socket.emit('chat_message', req);
    }

    public ping() {
        if (!this.socket.connected) throw new Error('Socket not connected. Call connect() first.');
        this.socket.emit('ping', { timestamp: new Date().toISOString() });
    }

    public get connected() { return this.isConnected && this.socket.connected; }
    public get socketId() { return this.socket.id; }
    public get sessionId(): string | undefined { return this.currentSessionId; }
}

// Export singleton instance
// export const socketChatService = new SocketChatService({});

// React hook for using the Socket.IO chat service

export interface UseSocketChatReturn {
    isConnected: boolean;
    isConnecting: boolean;
    socketId?: string;
    connect: (handlers: ChatEventHandlers, authContext?: AuthContextValue) => Promise<void>;
    disconnect: () => void;
    sendMessage: (request: ChatRequest) => void;
    ping: () => void;
    connectionError: string | null;
}

