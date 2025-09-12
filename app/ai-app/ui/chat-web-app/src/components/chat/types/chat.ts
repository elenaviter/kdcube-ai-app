/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2025 Elena Viter
 */

// types/chat.ts
export interface ModelInfo {
    id: string;
    name: string;
    provider: string;
    description: string;
    has_classifier: boolean;
}

export interface EmbedderInfo {
    id: string;
    provider: string;
    model: string;
    dimension: number;
    description: string;
}

export interface EmbeddingProvider {
    name: string;
    description: string;
    requires_api_key: boolean;
    requires_endpoint: boolean;
}

export interface BundleInfo {
    id: string;
    name?: string;
    description?: string;
    path: string;
    module?: string;
    singleton?: boolean;
}

export interface StepUpdate {
    step: string;
    title?: string;
    turn_id?: string;
    status: 'started' | 'completed' | 'error' | 'running';
    timestamp: Date;
    error?: string;
    data?: StepData;
    markdown?: string;   // ← from event.markdown
    agent?: string;      // ← from event.agent (optional, useful for UI)
}

interface StepData {
    message?: string;
    markdown?: string; // some backends might still put it here; we prefer top-level
    [key: string]: unknown;
}

/** minimal metadata used by grouping code */
interface ChatMessageMetadata {
    turn_id?: string;
}

export class ChatMessage {
    id: number;
    text: string;
    timestamp: Date;
    isError?: boolean;
    metadata?: ChatMessageMetadata;

    constructor(id: number, text: string, timestamp: Date, metadata?: ChatMessageMetadata) {
        this.id = id;
        this.text = text;
        this.timestamp = timestamp;
        this.metadata = metadata;
    }
}

export class UserChatMessage extends ChatMessage {
    attachments?: File[];

    constructor(id: number, text: string, timestamp: Date, metadata?: ChatMessageMetadata, attachments?: File[]) {
        super(id, text, timestamp, metadata);
        this.attachments = attachments;
    }
}

export class AssistantChatMessage extends ChatMessage {
}

interface ChatMessageInput {
    id: number;
    sender: 'user' | 'assistant';
    text: string;
    timestamp: Date;
    isError?: boolean;
    metadata?: ChatMessageMetadata;
    attachments?: File[];
}

export const createChatMessage = (input: ChatMessageInput): UserChatMessage | AssistantChatMessage => {
    const {id, text, timestamp, metadata, sender, isError, attachments} = input;
    return sender === 'user'
        ? new UserChatMessage(id, text, timestamp, metadata, attachments)
        : Object.assign(new AssistantChatMessage(id, text, timestamp, metadata), {isError});
};

export const createAssistantChatStep = (input: StepUpdate): AssistantChatStep => {
    const {step, status, timestamp, error, data, title, markdown, agent} = input;
    return new AssistantChatStep(step, status, timestamp, error, data, title, markdown, agent);
};

type AssistantChatStepStatus = 'started' | 'completed' | 'error'

export class AssistantChatStep {
    step: string;
    title?: string;
    status: AssistantChatStepStatus;
    timestamp: Date;
    error?: string;
    data?: StepData;
    markdown?: string;
    agent?: string;

    constructor(
        step: string,
        status: AssistantChatStepStatus,
        timestamp: Date,
        error?: string,
        data?: StepData,
        title?: string,
        markdown?: string,
        agent?: string
    ) {
        this.step = step;
        this.timestamp = timestamp;
        this.status = status;
        this.error = error;
        this.data = data;
        this.title = title;
        this.markdown = markdown;
        this.agent = agent;
    }

    getMarkdown() {
        return this.markdown || this.data?.markdown || '';
    }
}

/** Per-agent timing info */
export interface AgentTiming {
    startedAt: Date;
    endedAt?: Date;  // ← set only when the server sends completed: true
    active: boolean; // ← not used for display; keeps internal state
}

/** Thinking holder that supports multiple agent rows + per-agent timings */
export class AssistantThinkingItem {
    id: number;
    timestamp: Date;
    turn_id?: string;
    active: boolean;
    endedAt?: Date;
    /** Map of agent -> markdown text */
    agents: Record<string, string>;
    /** Map of agent -> timing */
    agentTimes: Record<string, AgentTiming>;

    constructor(
        id: number,
        timestamp: Date,
        turn_id?: string,
        active: boolean = true,
        endedAt?: Date,
        agents?: Record<string, string>,
        agentTimes?: Record<string, AgentTiming>,
    ) {
        this.id = id;
        this.timestamp = timestamp;
        this.turn_id = turn_id;
        this.active = active;
        this.endedAt = endedAt;
        this.agents = agents ?? {};
        this.agentTimes = agentTimes ?? {};
    }
}

export class DownloadItem {
    filename: string;
    rn: string;
    mimeType?: string;
    turnId?: string;

    constructor(filename: string, rn: string, mimeType?: string, turnId?: string) {
        this.filename = filename;
        this.rn = rn;
        this.mimeType = mimeType;
        this.turnId = turnId;
    }
}

export const createDownloadItem = (input: StepUpdate): DownloadItem => {
    const {data, turn_id} = input;
    return new DownloadItem(data.filename, data.rn, data?.mime, turn_id)
};

export class RichLink {
    url: string;
    title?: string;
    body?: string;

    constructor(url: string, title?: string, body?: string) {
        this.url = url;
        this.title = title;
        this.body = body;
    }
}

export class SourceLinks {
    links: RichLink[];
    turnId?: string;

    constructor(links: RichLink[], turnId?: string) {
        this.links = links;
        this.turnId = turnId;
    }
}

export const createSourceLinks = (input: StepUpdate): SourceLinks => {
    const {data, turn_id} = input;
    return new SourceLinks(data?.items?.map((item) => {
        return {url: item.url, title: item.title, body: item.body}
    }), turn_id)
};

export const createAssistantThinkingItem = (input: {
    id: number;
    timestamp: Date;
    turn_id?: string;
    initialAgents?: Record<string, string>;
    initialAgentTimes?: Record<string, AgentTiming>;
}): AssistantThinkingItem =>
    new AssistantThinkingItem(
        input.id,
        input.timestamp,
        input.turn_id,
        true,
        undefined,
        input.initialAgents,
        input.initialAgentTimes
    );

export type ChatLogItem =
    UserChatMessage
    | AssistantChatMessage
    | AssistantChatStep
    | AssistantThinkingItem
    | DownloadItem
    | SourceLinks;

