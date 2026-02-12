import {CanvasItemLink} from "../../canvas/canvasContext.tsx";
import {CodeExecArtifact, CodeExecArtifactType} from "./types.ts";
import {UnknownArtifact} from "../../chat/chatTypes.ts"

export const isCodeExecArtifactLink = (link: CanvasItemLink | null) => {
    return link?.itemType === CodeExecArtifactType
}
export const matchesCodeExecWithId = (link: CanvasItemLink | null, executionId: string) => {
    return executionId === link!.executionId
}
export const matchesCodeExecArtifact = (link: CanvasItemLink | null, item: UnknownArtifact) => {
    if (link?.itemType !== CodeExecArtifactType) return false;
    return matchesCodeExecWithId(link, (item as CodeExecArtifact).content.executionId)
}

export const getCodeExecArtifactLink = (artifact: UnknownArtifact): CanvasItemLink => {
    if (artifact.artifactType !== CodeExecArtifactType) {
        throw new Error("not a CodeExecArtifactType");
    }
    return {
        itemType: CodeExecArtifactType,
        executionId: (artifact as CodeExecArtifact).content.executionId,
    }
}