import {Artifact} from "../../chat/chatTypes.ts";

export interface WebSearchLink {
    url: string,
    title?: string | null,
    body?: string | null,
    favicon?: string | null,
    provider: string;
    weightedScore: number;
}

export interface WebSearchData {
    name: string;
    searchId: string;
    title?: string | null;
    items: WebSearchLink[];
    objective?: string;
    queries?: string[];
    reportContent?: string | null;
}

export const WebSearchArtifactType = "web_search_results"

export interface WebSearchArtifact extends Artifact<WebSearchData> {
    artifactType: typeof WebSearchArtifactType;
}