import {RefObject, useMemo} from "react";
import useChatCanvasContext, {CanvasItemLink} from "./canvasContext.tsx";
import {useAppSelector} from "../../app/store.ts";
import {selectTurns} from "../chat/chatStateSlice.ts";
import {
    UnknownArtifact
} from "../chat/chatTypes.ts";
import CanvasItem from "../logExtensions/canvas/CanvasItem.tsx";
import CodeExecCanvasItem from "../logExtensions/codeExec/CodeExecCanvasItem.tsx";
import {CodeExecArtifact, CodeExecArtifactType} from "../logExtensions/codeExec/types.ts";
import {CanvasArtifact, CanvasArtifactType} from "../logExtensions/canvas/types.ts";
import {matchesCanvasArtifact} from "../logExtensions/canvas/utils.ts";
import {matchesCodeExecArtifact} from "../logExtensions/codeExec/utils.ts";
import {WebSearchArtifact, WebSearchArtifactType} from "../logExtensions/webSearch/types.ts";
import {matchesWebSearchArtifact} from "../logExtensions/webSearch/utils.ts";
import WebSearchCanvasItem from "../logExtensions/webSearch/WebSearchCanvasItem.tsx";

interface ChatCanvasProps {
    ref?: RefObject<HTMLDivElement | null>;
}

const matchesLink = (link: CanvasItemLink | null,  artifact:UnknownArtifact) => {
    if (link?.itemType !== artifact.artifactType) {
        return false
    }
    switch (artifact.artifactType) {
        case CanvasArtifactType:
            return matchesCanvasArtifact(link, artifact as CanvasArtifact)
        case CodeExecArtifactType:
            return matchesCodeExecArtifact(link, artifact as CodeExecArtifact)
        case WebSearchArtifactType:
            return matchesWebSearchArtifact(link, artifact as WebSearchArtifact)
    }
}

const ChatCanvas = ({ref}: ChatCanvasProps) => {
    const {itemLink} = useChatCanvasContext()

    const turns = useAppSelector(selectTurns)
    const artifacts = useMemo(() => {
        return Object.values(turns).reduce((acc, cur) => {
            acc = acc.concat(cur.artifacts)
            return acc
        }, [] as UnknownArtifact[])
    }, [turns])

    const item = useMemo(() => {
        return artifacts.find(a => matchesLink(itemLink, a))
    }, [artifacts, itemLink])

    const content = useMemo(() => {
        if (!item) return null
        switch (item.artifactType) {
            case CanvasArtifactType:
                return <CanvasItem item={item as CanvasArtifact}/>
            case CodeExecArtifactType:
                return <CodeExecCanvasItem item={item as CodeExecArtifact}/>
            case WebSearchArtifactType:
                return <WebSearchCanvasItem item={item as WebSearchArtifact}/>
            default:
                return "Unknown artifact type"
        }

    }, [item])

    return useMemo(() => {
        return <div ref={ref} className={"h-full w-[35vw]"}>
            {content}
        </div>
    }, [content, ref])
}

export default ChatCanvas