/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2025 Elena Viter
 */

export interface ModelInfo {
    id: string; name: string; provider: string; description: string; has_classifier: boolean;
}
export interface EmbedderInfo {
    id: string; provider: string; model: string; dimension: number; description: string;
}
export interface EmbeddingProvider {
    name: string; description: string; requires_api_key: boolean; requires_endpoint: boolean;
}
export interface BundleInfo {
    id: string; name?: string; description?: string;
    path: string; module?: string; singleton?: boolean;
}
export interface StepData { [k: string]: any }
export interface StepUpdate {
    step: string; status: 'started'|'completed'|'error'; timestamp: string;
    elapsed_time?: string; error?: string; data?: StepData;
}
