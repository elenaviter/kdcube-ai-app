import {Indexed, Timestamped} from "../../types/common.ts";
import {RichLink, RNFile, TurnStep} from "../chatController/chatBase.ts";

export interface ChatSettings {
    showMetadata: boolean;
}

export interface WorkingScope {
    project: string;
    tenant: string;
}

export interface UserAttachmentDescription {
    name: string;
    size: number;
}

export interface UserAttachment extends UserAttachmentDescription {
    fileKey: string;
}

export interface ConversationState {
    conversationId: string | null | undefined;
    turns: Record<string, ChatTurn>;
    turnOrder: string[]
    conversationTitle?: string | null;
}

export interface ChatState extends WorkingScope, ConversationState {
    stayConnected: boolean;
    connected: boolean;
    locked: boolean;
    userMessage: string;
    userAttachments: UserAttachment[];
}

export interface UserMessage extends Timestamped {
    text: string,
    attachments: UserAttachmentDescription[],
}

export interface AssistantMessage {
    content: string
}

export interface AgentTiming {
    startedAt: number;
    endedAt?: number;  // ← set only when the server sends completed: true
    active: boolean; // ← not used for display; keeps internal state
}

export interface TurnEvent<C> extends Timestamped, Indexed {
    eventType: string;
    agent: string;
    completed: boolean;
    data: C;
}

export interface ThinkingEventData {
    text: string;
}

export interface ThinkingEvent extends TurnEvent<ThinkingEventData> {
    eventType: "thinking";
}

export interface AnswerEventData {
    text: string;
}

export interface AnswerEvent extends TurnEvent<AnswerEventData> {
    eventType: "answer";
}

export interface CanvasEventData {
    name: string;
    title?: string | null;
    content: string;
    contentType: string;
    subType: string | null;
}

export interface CanvasEvent extends TurnEvent<CanvasEventData> {
    eventType: "canvas";
}

export interface TimelineTextEventData {
    text: string;
}

export interface TimelineTextEvent extends TurnEvent<TimelineTextEventData> {
    eventType: "timeline_text";
}

// case "web_search.filtered_results":
// case "web_search.html_view":
// extra["searchId"] = env.extra.search_id as string;
// name = name + extra["searchId"]
// break
// case "code_exec.code":
// extra["language"] = env.extra.language as string;
// // eslint-disable-next-line no-fallthrough
// case "code_exec.program.name":
// case "code_exec.objective":
// case "code_exec.contract":
// case "code_exec.status":
// extra["executionId"] = env.extra.execution_id as string;
// break

export interface SubsystemEventData {
    subtype: string;
    name: string;
    text: string;
}

export interface WebSearchSubsystemEventData extends SubsystemEventData {
    searchId: string;
}

export interface WebSearchFilteredResultsSubsystemEventData extends WebSearchSubsystemEventData {
    subtype: "web_search.filtered_results"
}

export interface WebSearchHTMLViewSubsystemEventData extends WebSearchSubsystemEventData {
    subtype: "web_search.html_view"
}

export interface CodeExecSubsystemEventData extends SubsystemEventData {
    executionId: string;
}

export interface CodeExecCodeSubsystemEventData extends CodeExecSubsystemEventData {
    subtype: "code_exec.code"
    language: string;
}

export interface CodeExecProgramNameSubsystemEventData extends CodeExecSubsystemEventData {
    subtype: "code_exec.program.name"
}

export interface CodeExecObjectiveSubsystemEventData extends CodeExecSubsystemEventData {
    subtype: "code_exec.objective"
}

export interface CodeExecContractSubsystemEventData extends CodeExecSubsystemEventData {
    subtype: "code_exec.contract"
}

export interface CodeExecStatusSubsystemEventData extends CodeExecSubsystemEventData {
    subtype: "code_exec.status"
}

export type CodeExecMetaEventData = CodeExecProgramNameSubsystemEventData
    | CodeExecObjectiveSubsystemEventData
    | CodeExecStatusSubsystemEventData
    | CodeExecContractSubsystemEventData

export interface SubsystemEvent extends TurnEvent<SubsystemEventData> {
    eventType: "subsystem";
}

export interface TurnArtifact<C> {
    content: C;
    artifactType: string;
}

export interface ThinkingItemData extends Timestamped {
    endedAt?: number;
    agents: Record<string, string>;
    agentTimes: Record<string, AgentTiming>;
}

export interface TurnThinkingItem extends TurnArtifact<ThinkingItemData> {
    artifactType: "thinking";
}

export interface TurnCitation extends TurnArtifact<RichLink> {
    artifactType: "citation";
}

export interface TurnFile extends TurnArtifact<RNFile> {
    artifactType: "file";
}

export interface CanvasItemData extends Timestamped {
    name: string;
    description?: string | null;
    content: unknown;
    contentType: string;
    complete?: boolean;
    subType?: "webSearch" | string | null;
    title?: string | null;
}

export interface TurnCanvasItem extends TurnArtifact<CanvasItemData> {
    artifactType: "canvas";
}

export interface TimelineTextItem extends TurnArtifact<string> {
    artifactType: "timeline_text";
}

export interface UserMessageRequest {
    message?: string;
    files?: File[] | null;
}

export type TurnState = "new" | "inProgress" | "finished" | "error"

export interface ChatTurn {
    id: string;
    state: TurnState;
    userMessage: UserMessage;
    answer?: string | null;
    events: TurnEvent<unknown>[];
    steps: Record<string, TurnStep>;
    artifacts: TurnArtifact<unknown>[];
    error?: string | null;
    followUpQuestions: string[];
}

export interface NewChatTurnRequest {
    id: string;
    state: TurnState;
    userMessage: string;
    attachments: UserAttachmentDescription[];
}

export interface TurnError {
    id: string;
    error: string | Error;
}

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
    title?: string | null;
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

    [key: string]: unknown;
}

export interface ChatMessageData {
    id: number;
    sender: "user" | "assistant";
    text: string;
    timestamp: Date;
    isError?: boolean;
    isGreeting?: boolean; //only relevant for assistant message
    attachments?: File[] //only relevant for user message
    metadata?: ChatMessageMetadata
}

interface ChatMessageMetadata {
    turn_id?: string;
}