/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2025 Elena Viter
 */

// api/apiService.ts - Uses authContext.appendAuthHeader(headers) everywhere
import { io, Socket } from "socket.io-client";
import {
  BacktrackNavigation,
  EnhancedSearchResult,
  SearchPreviewContent,
} from "../search/SearchInterfaces";
import { AuthContextValue } from "../auth/AuthManager.tsx";
import {
  getKBAPIBaseAddress,
  getKBSocketAddress,
  getKBSocketSocketIOPath,
  getExtraIdTokenHeaderName,
} from "../../AppConfig.ts";

// ------------ Types matching the backend models ------------
export interface ChatMessage {
  id?: string;
  sender: string;
  text: string;
  timestamp?: Date;
  buttons?: Array<{ actionCaption: string; id: string }>;
}

export interface FileUploadResponse {
  success: boolean;
  file_info: {
    filename: string;
    saved_filename: string;
    path: string;
    size: number;
    size_formatted: string;
    mime_type: string;
    upload_time: string;
  };
}

export interface FileListResponse {
  files: Array<{
    filename: string;
    path: string;
    size: number;
    size_formatted: string;
    modified: string;
    mime_type?: string;
  }>;
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

export interface DataSummary {
  metadata: any[];
  summary: string;
  last_updated: string;
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

export interface KBSearchRequest {
  query: string;
  resource_id?: string;
  top_k?: number;
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

  /** Make a Headers object and apply auth via context */
  private makeHeaders(
    authContext?: AuthContextValue,
    base?: HeadersInit
  ): Headers {
    const h = new Headers(base as any);
    if (authContext) authContext.appendAuthHeader(h);
    return h;
  }

  /** Derive access + id tokens by calling appendAuthHeader into a temp list */
  private getTokensFromContext(
    authContext: AuthContextValue
  ): { accessToken?: string; idToken?: string } {
    const pairs: [string, string][] = [];
    authContext.appendAuthHeader(pairs);
    const idHdr = getExtraIdTokenHeaderName();

    let accessToken: string | undefined;
    let idToken: string | undefined;

    for (const [k, v] of pairs) {
      if (k.toLowerCase() === "authorization") {
        const m = /^Bearer\s+(.+)$/i.exec(v);
        if (m) accessToken = m[1];
      } else if (k === idHdr) {
        idToken = v;
      }
    }
    return { accessToken, idToken };
  }

  /** Escape special regex characters */
  private escapeRegex(str: string): string {
    return str.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  }

  // -------------------- CHAT --------------------

  async getChatMessages(authContext: AuthContextValue): Promise<ChatMessage[]> {
    const headers = this.makeHeaders(authContext, { "Content-Type": "application/json" });
    const res = await fetch(`${this.baseUrl}/api/chat/messages`, { headers });
    if (!res.ok) throw new Error("Failed to fetch chat messages");
    return res.json();
  }

  async postChatMessage(text: string, authContext: AuthContextValue): Promise<ChatMessage> {
    const headers = this.makeHeaders(authContext, { "Content-Type": "application/json" });
    const res = await fetch(`${this.baseUrl}/api/chat/messages`, {
      method: "POST",
      headers,
      body: JSON.stringify({ text }),
    });
    if (!res.ok) throw new Error("Failed to post chat message");
    return res.json();
  }

  // -------------------- DATA / KB (HTTP) --------------------

  async addDataElement(element: DataElement, authContext: AuthContextValue): Promise<any> {
    const headers = this.makeHeaders(authContext, { "Content-Type": "application/json" });
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

  async getSpendingReport(authContext: AuthContextValue): Promise<any> {
    const headers = this.makeHeaders(authContext);
    const res = await fetch(`${this.baseUrl}/api/spending`, { headers });
    if (!res.ok) throw new Error("Failed to fetch spending report");
    return res.json();
  }

  async getEventLog(authContext: AuthContextValue): Promise<{ events: any[] }> {
    const headers = this.makeHeaders(authContext);
    const res = await fetch(`${this.baseUrl}/api/events`, { headers });
    if (!res.ok) throw new Error("Failed to fetch event log");
    return res.json();
  }

  // -------------------- SOCKET.IO (KB) --------------------

  ensureKBSocket(
    authContext: AuthContextValue,
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

    const { accessToken, idToken } = this.getTokensFromContext(authContext);

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
    file: File,
    authContext: AuthContextValue,
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

      xhr.open("POST", `${this.baseUrl}/api/kb/upload`);

      // apply Authorization + X-ID-Token with the same helper
      const tmpPairs: [string, string][] = [];
      authContext.appendAuthHeader(tmpPairs);
      for (const [k, v] of tmpPairs) xhr.setRequestHeader(k, v);

      xhr.send(formData);
    });
  }

  async addURLToKB(request: KBAddURLRequest, authContext: AuthContextValue): Promise<KBUploadResponse> {
    const headers = this.makeHeaders(authContext, { "Content-Type": "application/json" });
    const res = await fetch(`${this.baseUrl}/api/kb/add-url`, {
      method: "POST",
      headers,
      body: JSON.stringify(request),
    });
    if (!res.ok) throw new Error("Failed to add URL to KB");
    return res.json();
  }

  async processKBURLWithSocket(
    authContext: AuthContextValue,
    resourceMetadata: any,
    socketId: string,
    processingMode?: string
  ): Promise<any> {
    const headers = this.makeHeaders(authContext, { "Content-Type": "application/json" });
    const res = await fetch(`${this.baseUrl}/api/kb/add-url/process`, {
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
    authContext: AuthContextValue,
    resource_metadata: any,
    socketId: string
  ): Promise<any> {
    const headers = this.makeHeaders(authContext, { "Content-Type": "application/json" });
    const res = await fetch(`${this.baseUrl}/api/kb/upload/process`, {
      method: "POST",
      headers,
      body: JSON.stringify({ resource_metadata, socket_id: socketId }),
    });
    if (!res.ok) throw new Error("Failed to start KB processing");
    return res.json();
  }

  async listKBResources(
    authContext: AuthContextValue,
    resourceType?: string
  ): Promise<{ resources: KBResource[]; total_count: number; kb_stats: any }> {
    const headers = this.makeHeaders(authContext);
    const res = await fetch(`${this.baseUrl}/api/kb/resources`, { headers });
    if (!res.ok) throw new Error("Failed to list KB resources");
    const result = await res.json();
    const resources = result.resources.filter((r: KBResource) =>
      resourceType ? r.source_type === resourceType : true
    );
    return { resources, total_count: result.total_count, kb_stats: result.kb_stats };
  }

  async getKBResourceContent(
    authContext: AuthContextValue,
    resourceId: string,
    version?: string,
    contentType: "raw" | "extraction" | "segments" = "raw"
  ): Promise<KBResourceContent> {
    const headers = this.makeHeaders(authContext);
    const params = new URLSearchParams({ content_type: contentType });
    if (version) params.append("version", version);

    const res = await fetch(`${this.baseUrl}/api/kb/resource/${resourceId}/content?${params}`, {
      headers,
    });
    if (!res.ok) throw new Error("Failed to get KB resource content");
    return res.json();
  }

  async deleteKBResource(authContext: AuthContextValue, resourceId: string): Promise<any> {
    const headers = this.makeHeaders(authContext);
    const res = await fetch(`${this.baseUrl}/api/kb/resource/${resourceId}`, {
      method: "DELETE",
      headers,
    });
    if (!res.ok) throw new Error("Failed to delete KB resource");
    return res.json();
  }

  getKBResourceDownloadUrl(resourceId: string, version?: string): string {
    const params = new URLSearchParams();

    if (version) params.append("version", version);
    const qs = params.toString();
    return `${this.baseUrl}/api/kb/resource/${resourceId}/download${qs ? "?" + qs : ""}`;
    }

  async getKBResourceContentForPreview(
    authContext: AuthContextValue,
    resourceId: string,
    version?: string
  ): Promise<{ content: any; mimeType: string; name: string }> {
    const data = await this.getKBResourceContent(authContext, resourceId, version);
    const { resources } = await this.listKBResources(authContext);
    const resource = resources.find((r) => r.id === resourceId);
    return {
      content: data.content,
      mimeType: resource?.mime || "application/octet-stream",
      name: resource?.name || "Unknown",
    };
  }

  // -------------------- RN content helpers --------------------

  async getContentByRN(request: RNContentRequest, authContext: AuthContextValue): Promise<RNContentResponse> {
    const headers = this.makeHeaders(authContext, { "Content-Type": "application/json" });
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
    request: EnhancedKBSearchRequest,
    authContext: AuthContextValue
  ): Promise<EnhancedKBSearchResponse> {
    const project = request.project || "default-project";
    const headers = this.makeHeaders(authContext, { "Content-Type": "application/json" });
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
    return { ...data, results: processedResults };
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
    authContext: AuthContextValue,
    navigation?: BacktrackNavigation[]
  ): Promise<{ content: string; highlighted_content: string; navigation_applied: boolean }> {
    try {
      const headers = this.makeHeaders(authContext, { "Content-Type": "application/json" });
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
      const contentResp = await this.getContentByRN({ rn, content_type: "auto" }, authContext);
      const content = contentResp.content;
      let highlighted = content;

      citations.forEach((c) => {
        const regex = new RegExp(`(${this.escapeRegex(c)})`, "gi");
        highlighted = highlighted.replace(
          regex,
          '<mark class="bg-yellow-200 px-1 rounded">$1</mark>'
        );
      });

      return { content, highlighted_content: highlighted, navigation_applied: false };
    } catch (e) {
      console.error("Failed to get highlighted content:", e);
      throw e;
    }
  }

  async getSegmentContent(
    authContext: AuthContextValue,
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
    const headers = this.makeHeaders(authContext, { "Content-Type": "application/json" });
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
    view_type: "original" | "extraction",
    authContext: AuthContextValue
  ): Promise<SearchPreviewContent> {
    const targetRN = view_type === "original" ? result.backtrack.raw.rn : result.backtrack.extraction.rn;
    if (!targetRN) throw new Error(`No ${view_type} RN available for this result`);

    const contentResp = await this.getContentByRN(
      { rn: targetRN, content_type: view_type === "original" ? "raw" : "extraction" },
      authContext
    );

    const highlightedResp = await this.getContentWithHighlighting(
      targetRN,
      result.backtrack.raw.citations,
      authContext,
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

// Named exports (keep surface area explicit)
export const {
  getChatMessages,
  postChatMessage,
  addDataElement,
  getSpendingReport,
  getEventLog,
  ensureKBSocket,
  waitForKBConnected,
  subscribeResourceProgress,
  disconnectKBSocket,
  uploadFileToKB,
  addURLToKB,
  processKBURLWithSocket,
  processKBFileWithSocket,
  listKBResources,
  getKBResourceContent,
  deleteKBResource,
  getKBResourceDownloadUrl,
  getKBResourceContentForPreview,
  extractResourceMetadata,
  getContentByRN,
  getKBHealth,
  searchKBEnhanced,
  getContentWithHighlighting,
  getSegmentContent,
  getEnhancedPreview,
} = apiService;
