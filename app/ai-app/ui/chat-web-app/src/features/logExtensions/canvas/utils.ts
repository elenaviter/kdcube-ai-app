import {CanvasItemLink} from "../../canvas/canvasContext.tsx";
import {CanvasArtifact, CanvasArtifactType} from "./types.ts";
import {UnknownArtifact} from "../../chat/chatTypes.ts";

export const isCanvasItemLink = (link: CanvasItemLink | null) => {
    return link?.itemType === CanvasArtifactType
}

export const matchesCanvasWithName = (link: CanvasItemLink | null, name: string) => {
    return name === link?.name
}

export const matchesCanvasArtifact = (link: CanvasItemLink | null, item: UnknownArtifact) => {
    if (link?.itemType !== CanvasArtifactType) return false;
    return matchesCanvasWithName(link, (item as CanvasArtifact).content.name)
}

export const getCanvasArtifactLink = (artifact: UnknownArtifact): CanvasItemLink => {
    if (artifact.artifactType !== CanvasArtifactType) {
        throw new Error("not a CanvasArtifactType");
    }
    return {
        itemType: CanvasArtifactType,
        name: (artifact as CanvasArtifact).content.name,
    }
}