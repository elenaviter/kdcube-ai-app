import useChatCanvasContext from "../../canvas/canvasContext.tsx";
import {useCallback, useMemo} from "react";
import {getFileIconClass} from "../../../components/FileIcons.tsx";
import {File} from "lucide-react";
import IconContainer from "../../../components/IconContainer.tsx";
import {CanvasArtifact, CanvasArtifactType} from "./types.ts";
import {isCanvasItemLink, matchesCanvasArtifact} from "./utils.ts";

interface CanvasItemProps {
    item: CanvasArtifact
}

export const CanvasLogItem = ({item}: CanvasItemProps) => {
    const {showItem, itemLink: selectedItem} = useChatCanvasContext()
    const selected = useMemo(() => {
        return isCanvasItemLink(selectedItem) && matchesCanvasArtifact(selectedItem, item)
    }, [item, selectedItem])
    const onClick = useCallback(() => {
        if (selected) {
            showItem(null)
        } else {
            showItem({
                itemType: CanvasArtifactType,
                name: item.content.name,
            })
        }
    }, [item, selected, showItem])
    return useMemo(() => {
        const Icon = getFileIconClass(item.content.contentType) ?? File
        return <button
            onClick={onClick}
            className={`w-full text-left p-2 min-w-0 flex flex-row rounded-lg mb-2 border  hover:bg-gray-100 cursor-pointer ${selected ? "border-gray-400" : "border-gray-200"}`}>
            <IconContainer icon={Icon} size={1.5}/>
            {item.content.title || item.content.name}
        </button>
    }, [item.content.contentType, item.content.title, item.content.name, onClick, selected])
}