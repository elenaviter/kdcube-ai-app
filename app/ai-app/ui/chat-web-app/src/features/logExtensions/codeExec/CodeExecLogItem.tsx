import {useEffect, useMemo, useState} from "react";
import {Maximize2, SquareTerminal} from "lucide-react";
import IconContainer from "../../../components/IconContainer.tsx";
import useChatCanvasContext from "../../canvas/canvasContext.tsx";
import {appendCodeMarkdown, cleanupCode} from "../../canvas/utils.ts";
import {
    markdownComponentsTight,
    rehypePlugins,
    remarkPlugins
} from "../../../components/chat/ChatInterface/markdownRenderUtils.tsx";
import ReactMarkdown from "react-markdown";
import {CodeExecArtifact, CodeExecArtifactType} from "./types.ts";
import {isCodeExecArtifactLink, matchesCodeExecArtifact} from "./utils.ts";
import {ChatLogComponentProps} from "../../extensions/logExtesnions.ts";

const CodeExecLogItem = ({item}: ChatLogComponentProps) => {
    if (item.artifactType !== CodeExecArtifactType) {
        throw new Error("not a CodeExecArtifact");
    }

    const codeExecItem = item as CodeExecArtifact;

    const {itemLink: selectedItem} = useChatCanvasContext()
    const selected = useMemo(() => {
        return isCodeExecArtifactLink(selectedItem) && matchesCodeExecArtifact(selectedItem, codeExecItem)
    }, [codeExecItem, selectedItem])

    const [expanded, setExpanded] = useState<boolean>(false)

    const detailsMemo = useMemo(() => {
        if (!codeExecItem.content.contract && !codeExecItem.content.program) return null
        return <>
            {codeExecItem.content.contract &&
                <div className={"my-1 w-full"}>
                    {codeExecItem.content.contract.content.map((a, i: number) => {
                        return <div className={"text-xs mb-1"} key={i}><span
                            className={"border bg-gray-50 rounded-sm px-1 py-0.5"}>{a.filename}</span> - {a.description}
                        </div>
                    })}
                </div>}
            {codeExecItem.content.program && <div className={"w-full [&_code]:max-h-[50vh]"}>
                <ReactMarkdown
                    remarkPlugins={remarkPlugins}
                    rehypePlugins={rehypePlugins}
                    components={markdownComponentsTight}
                    skipHtml={false}
                >
                    {appendCodeMarkdown(cleanupCode(codeExecItem.content.program.content), codeExecItem.content.program.language)}
                </ReactMarkdown>
            </div>}
        </>
    }, [codeExecItem.content.contract, codeExecItem.content.program])

    const hasDetails = useMemo(() => {
        return detailsMemo !== null
    }, [detailsMemo])

    useEffect(() => {
        if (!hasDetails) setExpanded(false);
    }, [hasDetails])

    return useMemo(() => {
        let executionStatus = ""
        if (codeExecItem.content.status) {
            switch (codeExecItem.content.status.content.status) {
                case "error":
                    executionStatus = " - Error"
                    break;
                case "gen":
                    executionStatus = " - Generating"
                    break;
                case "exec":
                    executionStatus = " - Executing"
                    break;
            }
        }

        return <div
            className={`relative w-full text-left p-2 min-w-0 flex flex-row rounded-lg mb-2 border ${selected ? "border-gray-400" : "border-gray-200"}`}>
            {hasDetails &&
                <button onClick={() => {
                    if (hasDetails) setExpanded(!expanded)
                }} className={"absolute top-2 right-2 cursor-pointer"}>
                    <IconContainer icon={Maximize2} size={1}
                                   className={"text-gray-500"}/>
                </button>}
            <div className={"flex flex-row w-full duration-200 transition-all"}>
                <IconContainer icon={SquareTerminal} size={1.5} className={"mr-0.5"}/>
                <div className={"flex-1 min-w-0 mr-4 mb-4 [&_*]:cursor-auto"}>
                    <div className={"flex flex-row w-fit"}>
                        {codeExecItem.content.name ? codeExecItem.content.name.content : "Program"}{executionStatus}
                    </div>
                    {codeExecItem.content.objective &&
                        <div className={"text-sm"}>{codeExecItem.content.objective?.content}</div>}
                    {expanded && detailsMemo}
                </div>
            </div>
        </div>
    }, [detailsMemo, expanded, hasDetails, codeExecItem.content.name, codeExecItem.content.objective, codeExecItem.content.status, selected])
}

export default CodeExecLogItem;