import {useMemo} from "react";
import {
    markdownComponents,
    rehypePlugins,
    remarkPlugins
} from "../../../components/chat/ChatInterface/markdownRenderUtils.tsx";
import {closeUpMarkdown, useWordStreamEffect} from "../../../components/WordStreamingEffects.tsx";
import ReactMarkdown from "react-markdown";
import {TimelineTextArtifact, TimelineTextArtifactType} from "./types.ts";
import {ChatLogComponentProps} from "../../extensions/logExtesnions.ts";

const TimelineTextLogItem = ({item, historical}: ChatLogComponentProps) => {
    if (item.artifactType !== TimelineTextArtifactType) {
        throw new Error("not a TimelineTextArtifact")
    }

    const timelineTextItem = item as TimelineTextArtifact;

    const streamedText = useWordStreamEffect(
        timelineTextItem.content.text ?? "",
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