import {CanvasItemLink} from "../../canvas/canvasContext.tsx";
import {WebSearchArtifact, WebSearchArtifactType} from "./types.ts";

export const isWebSearchArtifactLink = (link: CanvasItemLink | null) => {
    return link?.itemType === WebSearchArtifactType
}
export const matchesWebSearchWithId = (link: CanvasItemLink | null, searchId: string) => {
    return searchId === link!.searchId
}
export const matchesWebSearchArtifact = (link: CanvasItemLink | null, item: WebSearchArtifact) => {
    return matchesWebSearchWithId(link, item.content.searchId)
}