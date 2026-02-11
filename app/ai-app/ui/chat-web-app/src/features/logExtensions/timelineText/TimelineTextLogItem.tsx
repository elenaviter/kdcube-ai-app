import {useMemo} from "react";
import {
    markdownComponents,
    rehypePlugins,
    remarkPlugins
} from "../../../components/chat/ChatInterface/markdownRenderUtils.tsx";
import {closeUpMarkdown, useWordStreamEffect} from "../../../components/WordStreamingEffects.tsx";
import ReactMarkdown from "react-markdown";
import {TimelineTextArtifact} from "./types.ts";

interface TimelineTextLogItemProps {
    item: TimelineTextArtifact,
    historical: boolean
}

const TimelineTextLogItem = ({item, historical}: TimelineTextLogItemProps) => {
    const streamedText = useWordStreamEffect(
        item.content.text ?? "",
        !historical,
        50
    );

    return useMemo(() => {
        return <ReactMarkdown
            remarkPlugins={remarkPlugins}
            rehypePlugins={rehypePlugins}
            components={markdownComponents}
            skipHtml={false}
        >
            {closeUpMarkdown(streamedText)}
        </ReactMarkdown>
    }, [streamedText])
}

export default TimelineTextLogItem