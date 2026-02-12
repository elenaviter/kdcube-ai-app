import {CanvasItemLink} from "../../canvas/canvasContext.tsx";
import {WebSearchArtifact, WebSearchArtifactType} from "./types.ts";
import {UnknownArtifact} from "../../chat/chatTypes.ts";

export const isWebSearchArtifactLink = (link: CanvasItemLink | null) => {
    return link?.itemType === WebSearchArtifactType
}

export const matchesWebSearchWithId = (link: CanvasItemLink | null, searchId: string) => {
    return searchId === link!.searchId
}

export const matchesWebSearchArtifact = (link: CanvasItemLink | null, item: UnknownArtifact) => {
    if (link?.itemType !== WebSearchArtifactType) return false;
    return matchesWebSearchWithId(link, (item as WebSearchArtifact).content.searchId)
}

export const getWebSearchArtifactLink = (artifact: UnknownArtifact): CanvasItemLink => {
    if (artifact.artifactType !== WebSearchArtifactType) {
        throw new Error("not a WebSearchArtifactType");
    }
    return {
        itemType: WebSearchArtifactType,
        searchId: (artifact as WebSearchArtifact).content.searchId,
    }
}