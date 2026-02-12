import useChatCanvasContext from "../../canvas/canvasContext.tsx";
import {useCallback, useMemo} from "react";
import {getFileIconClass} from "../../../components/FileIcons.tsx";
import {File} from "lucide-react";
import IconContainer from "../../../components/IconContainer.tsx";
import {CanvasArtifact, CanvasArtifactType} from "./types.ts";
import {getCanvasArtifactLink, isCanvasItemLink, matchesCanvasArtifact} from "./utils.ts";
import {ChatLogComponentProps} from "../../extensions/logExtesnions.ts";

export const CanvasLogItem = ({item}: ChatLogComponentProps) => {
    if (item.artifactType !== CanvasArtifactType) {
        throw new Error("not a canvas artifact");
    }

    const canvasItem = item as CanvasArtifact

    const {showItem, itemLink: selectedItem} = useChatCanvasContext()
    const selected = useMemo(() => {
        return isCanvasItemLink(selectedItem) && matchesCanvasArtifact(selectedItem, canvasItem)
    }, [canvasItem, selectedItem])
    const onClick = useCallback(() => {
        if (selected) {
            showItem(null)
        } else {
            showItem(getCanvasArtifactLink(canvasItem))
        }
    }, [canvasItem, selected, showItem])
    return useMemo(() => {
        const Icon = getFileIconClass(canvasItem.content.contentType) ?? File
        return <button
            onClick={onClick}
            className={`w-full text-left p-2 min-w-0 flex flex-row rounded-lg mb-2 border  hover:bg-gray-100 cursor-pointer ${selected ? "border-gray-400" : "border-gray-200"}`}>
            <IconContainer icon={Icon} size={1.5}/>
            {canvasItem.content.title || canvasItem.content.name}
        </button>
    }, [canvasItem.content.contentType, canvasItem.content.title, canvasItem.content.name, onClick, selected])
}