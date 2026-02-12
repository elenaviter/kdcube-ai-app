import useChatCanvasContext from "../../canvas/canvasContext.tsx";
import {useCallback, useMemo, useState} from "react";
import IconContainer from "../../../components/IconContainer.tsx";
import {Ellipsis, Globe} from "lucide-react";
import IconLoader from "../../../components/IconLoader.tsx";
import {WebSearchArtifact, WebSearchArtifactType} from "./types.ts";
import {getWebSearchArtifactLink, isWebSearchArtifactLink, matchesWebSearchArtifact} from "./utils.ts";
import {ChatLogComponentProps} from "../../extensions/logExtesnions.ts";

const WebSearchLogItem = ({item}: ChatLogComponentProps) => {
    const maxVisibleLinks = 3;

    if (item.artifactType !== WebSearchArtifactType) {
        throw new Error("not a WebSearchArtifact");
    }

    const webSearchItem = item as WebSearchArtifact;

    const {itemLink: selectedItem, showItem} = useChatCanvasContext()
    const selected = useMemo(() => {
        return isWebSearchArtifactLink(selectedItem) && matchesWebSearchArtifact(selectedItem, webSearchItem)
    }, [webSearchItem, selectedItem])

    const [expanded, setExpanded] = useState<boolean>(false)


    const hasReport = useMemo(() => {
        return !!webSearchItem.content.reportContent
    }, [webSearchItem.content.reportContent])

    const expandable = useMemo(() => {
        return webSearchItem.content.items.length > maxVisibleLinks
    }, [webSearchItem.content.items, maxVisibleLinks])

    const sortedLinks = useMemo(() => {
        return webSearchItem.content.items.concat().sort((a, b) => {
            return a.weightedScore - b.weightedScore
        })
    }, [webSearchItem.content.items])

    const visibleLinks = useMemo(() => {
        return expanded || !expandable ? sortedLinks : sortedLinks.slice(0, maxVisibleLinks + 1)
    }, [expanded, expandable, sortedLinks, maxVisibleLinks])

    const onClick = useCallback(() => {
        if (selected) {
            showItem(null)
        } else {
            showItem(getWebSearchArtifactLink(webSearchItem))
        }
    }, [webSearchItem, selected, showItem])

    return useMemo(() => {
        return <div
            id={`webSearch_${webSearchItem.content.searchId}`}
            className={`relative w-full text-left p-2 min-w-0 flex flex-row rounded-lg mb-2 border ${selected ? "border-gray-400" : "border-gray-200"}`}>
            <div className={"flex flex-row w-full duration-200 transition-all"}>
                <IconContainer icon={Globe} size={1.5} className={"mr-0.5"}/>
                <div className={"flex-1 min-w-0 mb-4 [&_*]:cursor-auto"}>
                    <div className={"flex flex-row w-full"}>
                        {webSearchItem.content.title ?? webSearchItem.content.name}
                        {hasReport && <button
                            onClick={onClick}
                            className={"ml-auto border rounded-md text-xs border-gray-200 px-2 py-1 hover:bg-white cursor-pointer"}>Report</button>}
                    </div>
                    {webSearchItem.content.objective &&
                        <div className={"text-sm"}>{webSearchItem.content.objective}</div>}
                    {webSearchItem.content.queries && webSearchItem.content.queries.length > 0 && <div className={"text-sm"}>
                        <h2>Queries:</h2>
                        <ul className={"list-disc pl-5"}>
                            {webSearchItem.content.queries.map((q) => {
                                return <li key={q}>{q}</li>
                            })}
                        </ul>
                    </div>}
                    {visibleLinks.length > 0 && <div className={"text-sm border-gray-200 border p-2 rounded-md mt-1"}>
                        {visibleLinks.map((link, i) => {
                            return <a
                                key={i}
                                href={link.url}
                                target={"_blank"}
                                className={"!cursor-pointer underline flex flex-row items-center gap-1"}
                            >
                                <IconLoader url={link.favicon} size={1}/>
                                {link.title ?? link.url}
                            </a>
                        })}
                        {expandable && (expanded ? <div>

                        </div> : <button
                            className={"!cursor-pointer hover:text-black mt-1 flex flex-row items-center gap-1"}
                            onClick={() => {
                                setExpanded(true)
                            }}
                        >
                            <IconContainer icon={Ellipsis} size={1}/>
                            {sortedLinks.length - visibleLinks.length} more
                        </button>)}
                    </div>}
                </div>
            </div>
        </div>
    }, [expandable, expanded, hasReport, webSearchItem.content.name, webSearchItem.content.objective, webSearchItem.content.queries, webSearchItem.content.searchId, webSearchItem.content.title, onClick, selected, sortedLinks.length, visibleLinks])
}

export default WebSearchLogItem;