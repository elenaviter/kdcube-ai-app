import {useMemo} from "react";
import ReactMarkdown from "react-markdown";
import {
    markdownComponentsTight,
    rehypePlugins,
    remarkPlugins
} from "../../../components/chat/ChatInterface/markdownRenderUtils.tsx";
import {appendCodeMarkdown, cleanupCode} from "../../canvas/utils.ts";
import {CodeExecArtifact} from "./types.ts";

interface CodeExecItemProps {
    item: CodeExecArtifact
}

const CodeExecCanvasItem = ({item}: CodeExecItemProps) => {
    return useMemo(() => {
        return <div className={"p-2 border-gray-200 border-l-1 bg-white h-full w-full overflow-y-auto"}>
            <ReactMarkdown
                remarkPlugins={remarkPlugins}
                rehypePlugins={rehypePlugins}
                components={markdownComponentsTight}
                skipHtml={false}
            >
                {appendCodeMarkdown(cleanupCode(item.content!.program!.content), item.content!.program!.language)}
            </ReactMarkdown>
        </div>
    }, [item])
}

export default CodeExecCanvasItem;