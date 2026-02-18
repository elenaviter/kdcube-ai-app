import {Artifact} from "../../chat/chatTypes.ts";

export interface WebFetchDataItem {
    url: string,
    status: "success" | "timeout" | "paywall" | "error",
    mime: string,
    favicon?: string
    content_length: number,
    published_time_iso: string,
    modified_time_iso: string,
}

export interface WebFetchArtifactData {
    name: string;
    title?: string;
    objective?: string;
    items: WebFetchDataItem[];
}

export const WebFetchArtifactType = "web_fetch.results";

export interface WebFetchArtifact extends Artifact<WebFetchArtifactData> {
    artifactType: typeof WebFetchArtifactType;
    complete?: boolean;
}