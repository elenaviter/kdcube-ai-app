import hljs from "highlight.js";
import {useMemo, useState} from "react";
import ReactMarkdown from "react-markdown";
import {
    markdownComponentsTight,
    rehypePlugins,
    remarkPlugins
} from "../../../components/chat/ChatInterface/markdownRenderUtils.tsx";
import {closeUpMarkdown} from "../../../components/WordStreamingEffects.tsx";
import MermaidDiagram from "../../../components/MermaidDiagram.tsx";
import {appendCodeMarkdown, cleanupCode} from "../../canvas/utils.ts";
import {CanvasArtifact} from "./types.ts";

const getCanvasContentType = (format: string | null | undefined) => {
    if (!format) {
        return null;
    }
    format = format.toLowerCase();

    if (["markdown", "mermaid", "csv"].includes(format)) return format;
    if (hljs.listLanguages().includes(format)) return "code";
    if (format === "html") return "srcdoc";
    return null
}

interface CanvasItemProps {
    item: CanvasArtifact
}

const CanvasItem = ({item}: CanvasItemProps) => {
    const [showItemSource, setShowItemSource] = useState<boolean>(false)

    const contentType = useMemo(() => {
        return item ? getCanvasContentType(item.content.contentType) : null
    }, [item])

    const itemCompleted = useMemo(() => {
        return !!item?.complete
    }, [item])

    const showSourceSwitch = useMemo(() => {
        return itemCompleted && contentType && ["srcdoc", "mermaid", "csv"].includes(contentType)
    }, [itemCompleted, contentType])

    const itemSource = useMemo(() => {
        if (!item) return null;
        return String(item.content.content)
    }, [item])

    const itemRender = useMemo(() => {
        if (!item) return null;
        switch (contentType) {
            case "markdown":
                return <ReactMarkdown
                    remarkPlugins={remarkPlugins}
                    rehypePlugins={rehypePlugins}
                    components={markdownComponentsTight}
                    skipHtml={false}
                >
                    {closeUpMarkdown(item.content.content as string)}
                </ReactMarkdown>
            case "mermaid":
                return <MermaidDiagram chart={item.content.content as string}/>
            case "code":
                return <ReactMarkdown
                    remarkPlugins={remarkPlugins}
                    rehypePlugins={rehypePlugins}
                    components={markdownComponentsTight}
                    skipHtml={false}
                >
                    {appendCodeMarkdown(cleanupCode(item.content.content as string), item.content.contentType)}
                </ReactMarkdown>
            default:
                return <div>not supported</div>;
        }

    }, [contentType, item])

    return useMemo(() => {
        return <div className={"p-2 border-gray-200 border-l-1 bg-white h-full w-full overflow-y-auto"}>
            {showItemSource || !itemCompleted || !contentType ? itemSource : itemRender}
        </div>
    }, [contentType, itemCompleted, itemRender, itemSource, showItemSource])
}

export default CanvasItem