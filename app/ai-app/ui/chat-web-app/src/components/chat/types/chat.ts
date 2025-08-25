/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2025 Elena Viter
 */

import React, {useEffect, useRef, useState} from "react";
import {Bot} from "lucide-react";

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
    status: 'started' | 'completed' | 'error';
    timestamp: Date;
    elapsed_time?: string;
    error?: string;
    data?: StepData;
}

interface ChatMessageMetadata {
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
}

export class ChatMessage {
    id: number;
    //sender: 'user' | 'assistant';
    text: string;
    timestamp: Date;
    isError?: boolean;
    metadata?: ChatMessageMetadata

    constructor(id: number, text: string, timestamp: Date, metadata?: ChatMessageMetadata) {
        this.id = id;
        this.text = text;
        this.timestamp = timestamp;
        this.metadata = metadata;
    }
}

export class UserChatMessage extends ChatMessage {
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
}

export const createChatMessage =
    (input: ChatMessageInput): UserChatMessage | AssistantChatMessage => {
        const {id, text, timestamp, metadata, sender} = input;
        switch (sender) {
            case 'user':
                return new UserChatMessage(id, text, timestamp, metadata);
            case 'assistant':
                return new AssistantChatMessage(id, text, timestamp, metadata);
            default:
                throw new Error(`Unknown sender type: ${sender}`);
        }
    }

export const createAssistantChatStep =
    (input: StepUpdate): AssistantChatStep => {
        const { step, status, timestamp, error, elapsed_time, data} = input;
        return new AssistantChatStep(step, status, timestamp, elapsed_time, error, data)
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

type AssistantChatStepStatus = 'started' | 'completed' | 'error'

export class AssistantChatStep {
    step: string;
    status: AssistantChatStepStatus;
    timestamp: Date;
    elapsed_time?: string;
    error?: string;
    data?: StepData;

    constructor(step: string, status: AssistantChatStepStatus, timestamp: Date, elapsed_time?: string, error?: string, data?: StepData) {
        this.step = step;
        this.timestamp = timestamp;
        this.elapsed_time = elapsed_time;
        this.status = status;
        this.error = error;
        this.data = data;
    }
}

export type ChatLogItem = UserChatMessage | AssistantChatMessage | AssistantChatStep;
