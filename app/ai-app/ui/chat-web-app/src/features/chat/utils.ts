import {TurnThinkingItem, TurnArtifact, TurnCitation, TurnFile, TurnCanvasItem} from "./chatTypes.ts";

export const getTurnThinkingItems = (items: TurnArtifact<unknown>[]) => items.filter(it => it.artifactType === "thinking") as TurnThinkingItem[]
export const getTurnCitationItems = (items: TurnArtifact<unknown>[]) => items.filter(it => it.artifactType === "citation") as TurnCitation[]
export const getTurnFileItems = (items: TurnArtifact<unknown>[]) => items.filter(it => it.artifactType === "file") as TurnFile[]
export const getTurnCanvasItems = (items: TurnArtifact<unknown>[]) => items.filter(it => it.artifactType === "file") as TurnCanvasItem[]