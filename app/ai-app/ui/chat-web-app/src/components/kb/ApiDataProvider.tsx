// ApiDataProvider.tsx
import React, { createContext, useContext } from "react";
import {
    ChatMessage,
} from "./ApiService";


interface UseExample {
    content: string;
    status: string;
    aiResponse?: {
        text: string;
        annotations?: Array<{
            id: string;
            startIndex: number;
            endIndex: number;
            type: 'factual' | 'hallucination' | 'style' | 'other';
            note?: string;
        }>;
    };
}

interface SiteLink {
    id: string;
    title: string;
    url: string;
}

export interface FileDocument {
    id: string;
    name: string;
    size: string;
    type: string;
    path?: string;  // Added path field
    mime?: string;  // Added mime field
}

interface Dataset {
    id: string;
    name: string;
    description: string;
    format: string;
    rows?: number;
}

interface AIChatMessage {
    id: string;
    sender: "user" | "ai";
    text: string;
    timestamp: Date;
}

// Input sources for topics
export interface InputSource {
    id: string;
    title: string;
    url?: string;
    type: 'document' | 'url' | 'dataset';
    relevanceScore: number; // 0-100
}

// Event log entries
export interface EventLogEntry {
    id: string;
    nodeId: number;
    type: 'train' | 'exam' | 'add_data' | 'feedback' | 'improvement_started' | 'improvement_completed';
    description: string;
    timestamp: Date;
    metadata?: any;
}

// Topic data that changes during system life (separate from curriculum)
export interface TopicData {
    nodeId: number;
    status: "new" | "can_be_improved" | "improving" | "ready";
    coverage: number; // 0-100
    spentCost: number;
    projectedCost: number;
    initialCost: number; // Cost if learned from scratch
    updateCost: number; // Cost to improve if already learned
    isImproving: boolean;
    improvingProgress: number; // 0-100
    inputSources: InputSource[];
    eventLog: EventLogEntry[];
    learnedAt?: Date; // When first learned
    lastUpdated?: Date;
}

interface ApiDataProviderContext {
    // Chat-related
    requirementsChatMessages: Array<ChatMessage>;
    sendRequirementsChatMessages?: (msg: string) => void;
    requirementsDescription: string;
    setRequirementsDescription?: (msg: string) => void;

    // Examples-related
    useExamples: Array<UseExample>;
    addUseExample?: (content: string) => void;
    removeUseExample?: (content: string) => void;
    generateAIResponse?: (content: string) => void;
    addAnnotation?: (content: string, startIndex: number, endIndex: number, type: 'factual' | 'hallucination' | 'style' | 'other', note?: string) => void;

    // KB Panel data
    siteLinks: Array<SiteLink>;
    addSiteLink?: (title: string, url: string) => void;
    removeSiteLink?: (id: string) => void;
    fileDocuments: Array<FileDocument>;
    addFileDocument?: (name: string, size: string, type: string, path: string, mime: string) => void;
    removeFileDocument?: (id: string) => void;
    datasets: Array<Dataset>;
    addDataset?: (name: string, description: string, format: string, rows?: number) => void;
    removeDataset?: (id: string) => void;
    googleDriveConnected: boolean;
    setGoogleDriveConnected?: (connected: boolean) => void;

    // AI Chat data
    aiChatMessages: Array<AIChatMessage>;
    sendAIChatMessage?: (message: string) => void;

    // Loading states
    isLoading: boolean;
    error: string | null;

    // Topic data management (separate from curriculum)
    topicDataMap: Map<number, TopicData>;
    getTopicData: (nodeId: number) => TopicData;
    updateTopicData: (nodeId: number, updates: Partial<TopicData>) => void;
    addInputSourceToTopic: (nodeId: number, source: InputSource) => void;
    addEventLogEntry: (nodeId: number, entry: Omit<EventLogEntry, 'id' | 'nodeId' | 'timestamp'>) => void;
    improveTopics: (nodeIds: number[]) => void;
    addDataToTopics: (nodeIds: number[], files: FileList) => void;
    resetTopicData: () => void;
}

const ApiDataProviderContext = createContext<ApiDataProviderContext>({
    requirementsChatMessages: [],
    requirementsDescription: "",
    suggestions: [],
    useExamples: [],
    siteLinks: [],
    fileDocuments: [],
    datasets: [],
    googleDriveConnected: false,
    aiChatMessages: [],
    isLoading: false,
    error: null,
    topicDataMap: new Map(),
    getTopicData: () => ({} as TopicData),
    updateTopicData: () => {},
    addInputSourceToTopic: () => {},
    addEventLogEntry: () => {},
    improveTopics: () => {},
    addDataToTopics: () => {},
    resetTopicData: () => {},
});

export const useApiDataContext = () => {
    const ctx = useContext(ApiDataProviderContext);
    if (!ctx) throw new Error("useApiDataContext must be used inside <ApiDataProvider>");
    return ctx;
};
