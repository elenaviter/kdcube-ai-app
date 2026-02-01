export interface ConversationDescriptorDTO {
    conversation_id: string
    last_activity_at: string | null | undefined
    started_at: string | null | undefined
    title: string | null | undefined
}

export interface ConversationDescriptor {
    id: string
    lastActivity: number | null | undefined
    started: number | null | undefined
    title: string | null | undefined
}

export interface ArtifactDataDTO {
    payload: unknown,
    text: string
}

export interface AssistantFileData {
    payload: {
        filename: string,
        rn: string,
        mime: string | null | undefined,
        description: string | null | undefined,
    },
    text: string
}

export interface ThinkingStreamItem {
    agent: string
    text: string
    ts_first: number
    ts_last: number
}

export interface ThinkingStreamData {
    payload: {
        items: ThinkingStreamItem[]
    },
}

export type TurnArtifactType =
    'chat:user'
    | 'chat:assistant'
    | 'artifact:solver.program.files'
    | 'artifact:assistant.file'
    | 'artifact:user.attachment'
    | 'artifact:conv.thinking.stream'
    | 'artifact:solver.program.citables'
    | 'artifact:conv.artifacts.stream'
    | 'artifact:conv.user_shortcuts'
    | 'artifact:conv.timeline_text.stream'

export interface TurnArtifactDTO {
    ts: string
    type: TurnArtifactType | string
    data: ArtifactDataDTO
}

export interface TurnDTO {
    turn_id: string
    artifacts: TurnArtifactDTO[]
}

export interface ConversationDTO {
    conversation_id: string
    turns: TurnDTO[]
}

export interface ConversationsState {
    conversationDescriptors: ConversationDescriptor[] | null
    conversationsDescriptorsLoading: boolean
    conversationsDescriptorsLoadingError: string | null
    conversationLoading: string | null
    conversationStatusUpdateRequired: boolean
}