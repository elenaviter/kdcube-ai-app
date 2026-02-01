/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2025 Elena Viter
 */

// features/apiService.ts
import {io, Socket} from "socket.io-client";
import {BacktrackNavigation, EnhancedSearchResult, SearchPreviewContent,} from "../search/SearchInterfaces";
import {getKBAPIBaseAddress, getKBSocketAddress, getKBSocketSocketIOPath,} from "../../AppConfig.ts";
import {appendDefaultCredentialsHeader} from "../../app/api/utils.ts";

// ------------ Types matching the backend models ------------
export interface ChatMessage {
    id?: string;
    sender: string;
    text: string;
    timestamp?: Date;
    buttons?: Array<{ actionCaption: string; id: string }>;
}


export interface DataElement {
    type: "url" | "file" | "raw_text";
    url?: string;
    parser_type?: string;
    mime?: string;
    filename?: string;
    path?: string;
    metadata?: Record<string, any>;
    text?: string;
    name?: string;
}

export interface KBResource {
    id: string;
    source_id: string;
    source_type: string;
    uri: string;
    filename: string;
    ef_uri: string;
    name: string;
    mime?: string;
    version: string;
    size_bytes?: number;
    timestamp: string;
    processing_status: {
        extraction: boolean;
        segmentation: boolean;
        metadata: boolean;
        summarization: boolean;
    };
    fully_processed: boolean;
    rns?: {
        raw: string;
        extraction: string;
        segmentation: string;
    };
    extraction_info?: any;
}

export interface EnhancedKBSearchRequest {
    query: string;
    resource_id?: string;
    top_k?: number;
    include_backtrack?: boolean;
    include_navigation?: boolean;
    tenant?: string;
    project?: string;
}

export interface EnhancedKBSearchResponse {
    query: string;
    results: EnhancedSearchResult[];
    total_results: number;
    search_metadata: {
        enhanced_search: boolean;
        backtrack_enabled: boolean;
        navigation_enabled: boolean;
        [key: string]: any;
    };
}

export interface KBUploadResponse {
    success: boolean;
    resource_id: string;
    resource_metadata: Record<string, any>;
    message: string;
    user_session_id?: string;
}

export interface KBResourceContent {
    resource_id: string;
    version: string;
    content?: string;
    segments?: Array<Record<string, any>>;
    type: "raw" | "extraction" | "segments";
    segment_count?: number;
    available_files?: string[];
}

export interface RNContentRequest {
    rn: string;
    content_type?: string;
}

export interface RNContentResponse {
    rn: string;
    content_type: string;
    content: any;
    metadata: Record<string, any>;
}

export interface KBAddURLRequest {
    url: string;
    name?: string;
}

// ============================================================

class ApiService {
    private baseUrl: string;
    private kbSocket?: Socket;

    constructor() {
        this.baseUrl = getKBAPIBaseAddress();
    }

    // -------------------- helpers --------------------

    /** Escape special regex characters */
    private escapeRegex(str: string): string {
        return str.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    }

    // -------------------- CHAT --------------------

    async getChatMessages(): Promise<ChatMessage[]> {
        const headers = appendDefaultCredentialsHeader([
            ['Content-Type', 'application/json']
        ]);
        const res = await fetch(`${this.baseUrl}/api/chat/messages`, {headers});
        if (!res.ok) throw new Error("Failed to fetch chat messages");
        return res.json();
    }

    // -------------------- DATA / KB (HTTP) --------------------

    async addDataElement(element: DataElement): Promise<any> {
        const headers = appendDefaultCredentialsHeader([
            ['Content-Type', 'application/json']
        ]);
        const res = await fetch(`${this.baseUrl}/api/data/elements`, {
            method: "POST",
            headers,
            body: JSON.stringify(element),
        });
        if (!res.ok) throw new Error("Failed to add data element");
        return res.json();
    }

    // -------------------- UTIL --------------------

    extractResourceMetadata(rn: string): {
        project: string;
        stage: string;
        resourceId: string;
        version: string;
        filename?: string;
    } {
        const parts = rn.split(":");
        return {
            project: parts[1] || "unknown",
            stage: parts[3] || "unknown",
            resourceId: parts[4] || "unknown",
            version: parts[5] || "1",
            filename: parts[6],
        };
    }

    async getSpendingReport(): Promise<any> {
        const headers = appendDefaultCredentialsHeader();
        const res = await fetch(`${this.baseUrl}/api/spending`, {headers});
        if (!res.ok) throw new Error("Failed to fetch spending report");
        return res.json();
    }

    async getEventLog(): Promise<{ events: any[] }> {
        const headers = appendDefaultCredentialsHeader();
        const res = await fetch(`${this.baseUrl}/api/events`, {headers});
        if (!res.ok) throw new Error("Failed to fetch event log");
        return res.json();
    }

    // -------------------- SOCKET.IO (KB) --------------------

    ensureKBSocket(
        accessToken: string | null | undefined,
        idToken: string | null | undefined,
        project?: string,
        tenant?: string,
        userSessionId?: string
    ): Socket {
        if (this.kbSocket?.connected) return this.kbSocket;

        if (this.kbSocket) {
            this.kbSocket.off();
            this.kbSocket.disconnect();
            this.kbSocket = undefined;
        }

        this.kbSocket = io(getKBSocketAddress(), {
            path: getKBSocketSocketIOPath(),
            transports: ["websocket", "polling"],
            forceNew: false,
            timeout: 5000,
            // In browsers we can't set custom headers; put tokens in the auth payload
            auth: {
                bearer_token: accessToken,
                id_token: idToken,
                project,
                tenant,
                user_session_id: userSessionId,
            },
        });

        this.kbSocket.on("connect", () => console.log("KB socket connected", this.kbSocket?.id));
        this.kbSocket.on("disconnect", (reason) => console.log("KB socket disconnected", reason));
        this.kbSocket.on("connect_error", (err) => console.error("KB socket connect_error", err));

        return this.kbSocket;
    }

    async waitForKBConnected(timeoutMs = 5000): Promise<void> {
        if (this.kbSocket?.connected) return;
        if (!this.kbSocket) throw new Error("KB socket not initialized");
        await new Promise<void>((resolve, reject) => {
            const onConnect = () => {
                cleanup();
                resolve();
            };
            const onError = (e: any) => {
                cleanup();
                reject(e);
            };
            const timer = setTimeout(() => {
                cleanup();
                reject(new Error("Socket connection timeout"));
            }, timeoutMs);

            const cleanup = () => {
                clearTimeout(timer);
                this.kbSocket?.off("connect", onConnect);
                this.kbSocket?.off("connect_error", onError);
            };

            this.kbSocket.once("connect", onConnect);
            this.kbSocket.once("connect_error", onError);
        });
    }

    subscribeResourceProgress(resourceId: string, handler: (msg: any) => void): () => void {
        if (!this.kbSocket) throw new Error("KB socket not connected");
        const channel = `resource_processing_progress:${resourceId}`;
        const listener = (msg: any) => handler(msg);
        this.kbSocket.off(channel, listener);
        this.kbSocket.on(channel, listener);
        return () => this.kbSocket?.off(channel, listener);
    }

    disconnectKBSocket() {
        if (!this.kbSocket) return;
        this.kbSocket.off();
        this.kbSocket.disconnect();
        this.kbSocket = undefined;
    }

    // -------------------- KB FILE / URL --------------------

    async uploadFileToKB(
        project: string,
        tenant: string,
        file: File,
        onProgress?: (progress: number) => void
    ): Promise<KBUploadResponse> {
        const formData = new FormData();
        formData.append("file", file);

        return new Promise((resolve, reject) => {
            const xhr = new XMLHttpRequest();

            if (onProgress) {
                xhr.upload.addEventListener("progress", (e) => {
                    if (e.lengthComputable) {
                        const progress = Math.round((e.loaded * 100) / e.total);
                        onProgress(progress);
                    }
                });
            }

            xhr.addEventListener("load", () => {
                if (xhr.status >= 200 && xhr.status < 300) {
                    try {
                        resolve(JSON.parse(xhr.responseText));
                    } catch {
                        reject(new Error("Invalid response format"));
                    }
                } else {
                    try {
                        const err = JSON.parse(xhr.responseText);
                        reject(new Error(err.detail || "Upload failed"));
                    } catch {
                        reject(new Error(`Upload failed with status ${xhr.status}`));
                    }
                }
            });

            xhr.addEventListener("error", () => reject(new Error("Network error during upload")));

            xhr.open("POST", `${this.baseUrl}/api/kb/${project}/upload`);

            // apply Authorization + X-ID-Token with the same helper
            const tmpPairs = appendDefaultCredentialsHeader( []) as [string, string][];
            for (const [k, v] of tmpPairs) xhr.setRequestHeader(k, v);

            xhr.send(formData);
        });
    }

    async addURLToKB(project: string, tenant: string, request: KBAddURLRequest): Promise<KBUploadResponse> {
        const headers = appendDefaultCredentialsHeader( [["Content-Type", "application/json"]]);
        const res = await fetch(`${this.baseUrl}/api/kb/${project}/add-url`, {
            method: "POST",
            headers,
            body: JSON.stringify(request),
        });
        if (!res.ok) throw new Error("Failed to add URL to KB");
        return res.json();
    }

    async processKBURLWithSocket(
        project: string,
        tenant: string,
        resourceMetadata: any,
        socketId: string,
        processingMode?: string
    ): Promise<any> {
        const headers = appendDefaultCredentialsHeader( [["Content-Type", "application/json"]]);
        const res = await fetch(`${this.baseUrl}/api/kb/${project}/add-url/process`, {
            method: "POST",
            headers,
            body: JSON.stringify({
                resource_metadata: resourceMetadata,
                socket_id: socketId,
                processing_mode: processingMode ?? "retrieval_only",
            }),
        });
        if (!res.ok) throw new Error("Failed to start URL processing");
        return res.json();
    }

    async processKBFileWithSocket(
        project: string,
        tenant: string,
        resource_metadata: any,
        socketId: string,
    ): Promise<any> {
        const headers = appendDefaultCredentialsHeader( [["Content-Type", "application/json"]]);
        const res = await fetch(`${this.baseUrl}/api/kb/${project}/upload/process`, {
            method: "POST",
            headers,
            body: JSON.stringify({resource_metadata, socket_id: socketId}),
        });
        if (!res.ok) throw new Error("Failed to start KB processing");
        return res.json();
    }

    async listKBResources(
        project: string,
        tenant: string,
        resourceType?: string
    ): Promise<{ resources: KBResource[]; total_count: number; kb_stats: any }> {
        const headers = appendDefaultCredentialsHeader();
        const res = await fetch(`${this.baseUrl}/api/kb/${project}/resources`, {headers});
        if (!res.ok) throw new Error("Failed to list KB resources");
        const result = await res.json();
        const resources = result.resources.filter((r: KBResource) =>
            resourceType ? r.source_type === resourceType : true
        );
        return {resources, total_count: result.total_count, kb_stats: result.kb_stats};
    }

    async getKBResourceContent(
        project: string,
        tenant: string,
        resourceId: string,
        version?: string,
        contentType: "raw" | "extraction" | "segments" = "raw"
    ): Promise<KBResourceContent> {
        const headers = appendDefaultCredentialsHeader();
        const params = new URLSearchParams({content_type: contentType});
        if (version) params.append("version", version);

        const res = await fetch(`${this.baseUrl}/api/kb/${project}/resource/${resourceId}/content?${params}`, {
            headers,
        });
        if (!res.ok) throw new Error("Failed to get KB resource content");
        return res.json();
    }

    async deleteKBResource(project: string, tenant: string, resourceId: string): Promise<any> {
        const headers = appendDefaultCredentialsHeader();
        const res = await fetch(`${this.baseUrl}/api/kb/${project}/resource/${resourceId}`, {
            method: "DELETE",
            headers,
        });
        if (!res.ok) throw new Error("Failed to delete KB resource");
        return res.json();
    }

    // TODO: request by RN instead
    getKBResourceDownloadUrl(project: string, tenant: string, resourceId: string, version?: string): string {
        const params = new URLSearchParams();

        if (version) params.append("version", version);
        const qs = params.toString();
        return `${this.baseUrl}/api/kb/${project}/resource/${resourceId}/download${qs ? "?" + qs : ""}`;
    }


    // -------------------- RN content helpers --------------------

    async getContentByRN(request: RNContentRequest): Promise<RNContentResponse> {
        const headers = appendDefaultCredentialsHeader( [["Content-Type", "application/json"]]);
        const res = await fetch(`${this.baseUrl}/api/kb/content/by-rn`, {
            method: "POST",
            headers,
            body: JSON.stringify(request),
        });
        if (!res.ok) throw new Error(`Failed to get content by RN: ${res.status}`);
        return res.json();
    }

    async getKBHealth(): Promise<any> {
        const res = await fetch(`${this.baseUrl}/api/kb/health`);
        if (!res.ok) throw new Error("Failed to get KB health");
        return res.json();
    }

    // -------------------- KB search --------------------

    async searchKBEnhanced(
        request: EnhancedKBSearchRequest
    ): Promise<EnhancedKBSearchResponse> {
        const project = request.project || "default-project";
        const headers = appendDefaultCredentialsHeader( [["Content-Type", "application/json"]]);
        const res = await fetch(`${this.baseUrl}/api/kb/${project}/search/enhanced`, {
            method: "POST",
            headers,
            body: JSON.stringify({
                ...request,
                include_backtrack: true,
                include_navigation: true,
            }),
        });

        if (!res.ok) throw new Error(`Search failed with status ${res.status}`);

        const data = (await res.json()) as EnhancedKBSearchResponse;
        const processedResults = this.processSearchResults(data.results, request.query);
        return {...data, results: processedResults};
    }

    private processSearchResults(results: any[], query: string): EnhancedSearchResult[] {
        return results.map((r) => ({
            query: r.query || query,
            relevance_score: r.relevance_score || 0,
            heading: r.heading || "",
            subheading: r.subheading || "",
            backtrack: {
                raw: {
                    citations: r.backtrack?.raw?.citations || [query],
                    rn: r.backtrack?.raw?.rn || "",
                },
                extraction: {
                    related_rns: r.backtrack?.extraction?.related_rns || [],
                    rn: r.backtrack?.extraction?.rn || "",
                },
                segmentation: {
                    rn: r.backtrack?.segmentation?.rn || "",
                    navigation: this.processNavigation(r.backtrack?.segmentation?.navigation || []),
                },
            },
        }));
    }

    private processNavigation(navigation: any[]): BacktrackNavigation[] {
        return navigation.map((nav) => ({
            start_line: nav.start_line || 0,
            end_line: nav.end_line || 0,
            start_pos: nav.start_pos || 0,
            end_pos: nav.end_pos || 0,
            citations: nav.citations || [],
            text: nav.text,
            heading: nav.heading,
            subheading: nav.subheading,
        }));
    }

    async getContentWithHighlighting(
        rn: string,
        citations: string[],
        navigation?: BacktrackNavigation[]
    ): Promise<{ content: string; highlighted_content: string; navigation_applied: boolean }> {
        try {
            const headers = appendDefaultCredentialsHeader( [["Content-Type", "application/json"]]);
            const res = await fetch(`${this.baseUrl}/api/kb/content/highlighted`, {
                method: "POST",
                headers,
                body: JSON.stringify({
                    rn,
                    citations,
                    navigation,
                    highlight_format: '<mark class="bg-yellow-200 px-1 rounded">{}</mark>',
                }),
            });

            if (res.ok) return res.json();

            // Fallback: fetch content and highlight locally
            const contentResp = await this.getContentByRN({rn, content_type: "auto"});
            const content = contentResp.content;
            let highlighted = content;

            citations.forEach((c) => {
                const regex = new RegExp(`(${this.escapeRegex(c)})`, "gi");
                highlighted = highlighted.replace(
                    regex,
                    '<mark class="bg-yellow-200 px-1 rounded">$1</mark>'
                );
            });

            return {content, highlighted_content: highlighted, navigation_applied: false};
        } catch (e) {
            console.error("Failed to get highlighted content:", e);
            throw e;
        }
    }

    async getSegmentContent(
        rn: string,
        segment_index: number,
        highlight_citations?: string[]
    ): Promise<{
        segment_content: string;
        highlighted_content: string;
        navigation_info: BacktrackNavigation;
        context_before?: string;
        context_after?: string;
    }> {
        const headers = appendDefaultCredentialsHeader( [["Content-Type", "application/json"]]);
        const res = await fetch(`${this.baseUrl}/api/kb/content/segment`, {
            method: "POST",
            headers,
            body: JSON.stringify({
                rn,
                segment_index,
                highlight_citations,
                include_context: true,
                context_lines: 3,
            }),
        });
        if (!res.ok) throw new Error(`Failed to get segment content: ${res.status}`);
        return res.json();
    }

    async getEnhancedPreview(
        result: EnhancedSearchResult,
        view_type: "original" | "extraction"
    ): Promise<SearchPreviewContent> {
        const targetRN = view_type === "original" ? result.backtrack.raw.rn : result.backtrack.extraction.rn;
        if (!targetRN) throw new Error(`No ${view_type} RN available for this result`);

        const contentResp = await this.getContentByRN(
            {rn: targetRN, content_type: view_type === "original" ? "raw" : "extraction"}
        );

        const highlightedResp = await this.getContentWithHighlighting(
            targetRN,
            result.backtrack.raw.citations,
            view_type === "extraction" ? result.backtrack.segmentation.navigation : undefined
        );

        const rnParts = targetRN.split(":");
        const resourceId = rnParts[4] || "unknown";
        const version = rnParts[5] || "1";

        return {
            type: view_type,
            resource_id: resourceId,
            version,
            rn: targetRN,
            content: contentResp.content,
            highlightedContent: highlightedResp.highlighted_content,
            mimeType:
                view_type === "extraction" ? "text/markdown" : contentResp.metadata.mime || "text/plain",
            filename: contentResp.metadata.filename || "Unknown",
            navigation: view_type === "extraction" ? result.backtrack.segmentation.navigation : undefined,
            citations: result.backtrack.raw.citations,
        };
    }
}

// Singleton
export const apiService = new ApiService();