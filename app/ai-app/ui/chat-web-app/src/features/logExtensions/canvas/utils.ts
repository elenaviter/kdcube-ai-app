import {CanvasItemLink} from "../../canvas/canvasContext.tsx";
import {CanvasArtifact, CanvasArtifactType} from "./types.ts";

export const isCanvasItemLink = (link: CanvasItemLink | null) => {
    return link?.itemType === CanvasArtifactType
}
export const matchesCanvasWithName = (link: CanvasItemLink | null, name: string) => {
    return name === link?.name
}
export const matchesCanvasArtifact = (link: CanvasItemLink | null, item: CanvasArtifact) => {
    return matchesCanvasWithName(link, item.content.name)
}