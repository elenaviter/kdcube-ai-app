import {WebSearchArtifact} from "./types.ts";
import {useMemo} from "react";

interface WebSearchCanvasItemProps {
    item: WebSearchArtifact
}

const WebSearchCanvasItem = ({item}:WebSearchCanvasItemProps) => {
    return useMemo(() => {
        if (!item.content.reportContent) return null;
        return <div className={"w-full h-full overflow-hidden"}>
            <iframe
                srcDoc={item.content.reportContent}
                className={"w-full h-full border-0"}
            />
        </div>
    }, [item.content.reportContent])
}

export default WebSearchCanvasItem;