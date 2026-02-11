import {Artifact} from "../../chat/chatTypes.ts";

export interface TimelineTextArtifactData {
    name: string;
    text: string;
}

export const TimelineTextArtifactType = "timeline_text"

export interface TimelineTextArtifact extends Artifact<TimelineTextArtifactData> {
    artifactType: typeof TimelineTextArtifactType
}