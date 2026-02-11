import {createContext, useContext} from "react";
import {CodeExecArtifact} from "../logExtensions/codeExec/types.ts";
import {CanvasArtifact} from "../logExtensions/canvas/types.ts";


export interface CanvasItemLink {
    itemType: CanvasArtifact["artifactType"] | CodeExecArtifact["artifactType"] | string
    [key: string]: unknown
}

export interface ChatCanvasContextValue {
    showItem: (item: CanvasItemLink | null) => void;
    itemLink: CanvasItemLink | null;
}

export const ChatCanvasContext = createContext<ChatCanvasContextValue>({
    showItem: () => {
        throw "not implemented";
    },
    itemLink: null,
});

const useChatCanvasContext = () => {
    return useContext(ChatCanvasContext);
}

export default useChatCanvasContext;