import {CanvasItemLink} from "../../canvas/canvasContext.tsx";
import {CodeExecArtifact, CodeExecArtifactType} from "./types.ts";

export const isCodeExecArtifactLink = (link: CanvasItemLink | null) => {
    return link?.itemType === CodeExecArtifactType
}
export const matchesCodeExecWithId = (link: CanvasItemLink | null, executionId: string) => {
    return executionId === link!.executionId
}
export const matchesCodeExecArtifact = (link: CanvasItemLink | null, item: CodeExecArtifact) => {
    return matchesCodeExecWithId(link, item.content.executionId)
}