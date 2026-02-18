import {Artifact} from "../../chat/chatTypes.ts";

export interface ConversationStatusArtifactData {
    status: string;
}

export const ConversationStatusArtifactType = "conversation.turn.status";

export interface ConversationStatusArtifact extends Artifact<ConversationStatusArtifactData> {
    artifactType: typeof ConversationStatusArtifactType;
    complete?: boolean;
}