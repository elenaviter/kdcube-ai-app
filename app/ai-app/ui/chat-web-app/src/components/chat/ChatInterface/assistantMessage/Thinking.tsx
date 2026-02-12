import {ThinkingArtifact} from "../../../../features/chat/chatTypes.ts";
import {useCallback, useMemo, useState} from "react";
import {ChevronDown, ChevronUp} from "lucide-react";
import ReactMarkdown from "react-markdown";
import {markdownComponentsTight, rehypePlugins, remarkPlugins} from "../markdownRenderUtils.tsx";
import {closeUpMarkdown} from "../../../WordStreamingEffects.tsx";

const formatSeconds = (sec: number): string => {
    if (!isFinite(sec) || sec < 0) sec = 0;
    if (sec < 60) return `${Math.round(sec * 10) / 10}s`;
    const m = Math.floor(sec / 60);
    const s = Math.round((sec % 60) * 10) / 10;
    const sStr = (s < 10 ? "0" : "") + s.toFixed(1);
    return `${m}:${sStr}`;
};

interface ThinkingItemProps {
    item: ThinkingArtifact
}

export const Thinking = ({item}: ThinkingItemProps) => {
    const [expanded, setExpanded] = useState<boolean>(false)

    const active = !!item.content.endedAt;
    const endedAt = item.content.endedAt;
    const endMs = active ? Date.now() : endedAt ?? Date.now();
    const durSec = Math.max(0, Math.round(((endMs - item.content.timestamp) / 1000) * 10) / 10);
    const agentKeys = Object.keys(item.content.agents || {})

    const getAgentSecondsIfCompleted = useCallback((agent: string): number | null => {
        const meta = item.content.agentTimes?.[agent];
        if (!meta?.endedAt || !meta?.startedAt) return null; // ← only show after completed
        const s = (meta.endedAt - meta.startedAt) / 1000;
        return Math.max(0, Math.round(s * 10) / 10);
    }, [item.content.agentTimes]);

    return useMemo(() => {
        if (Object.keys(agentKeys).length === 0) {
            return null
        }
        return (
            <div
                className="flex flex-col rounded-lg mb-2 border border-gray-200 [&>div:not(:last-child)]:border-b [&>div:not(:last-child)]:border-gray-200">
                <button
                    className={`flex items-center justify-between px-3 py-2 cursor-pointer select-none`}
                    onClick={() => {
                        setExpanded(!expanded)
                    }}
                    title={expanded ? "Collapse" : "Expand"}
                >
                    <div className="flex items-center gap-2">
                        <span className={`text-sm font-medium ${active ? "thinking-animated" : "text-gray-700"}`}>
                            {active ? "Thinking" : `Thought for ${formatSeconds(durSec)}`}
                        </span>
                    </div>
                    <span className="text-gray-500">{expanded ? <ChevronUp size={16}/> :
                        <ChevronDown size={16}/>}</span>
                </button>
                {expanded && (
                    <div className="px-3 py-2 text-xs text-slate-700">
                        {agentKeys.length === 0 ? (
                            <div className="text-gray-500 italic">No thoughts yet…</div>
                        ) : (
                            <div className="space-y-3">
                                {agentKeys.map((agent) => {
                                    const secs = getAgentSecondsIfCompleted(agent);
                                    return (
                                        <div key={agent}>
                                            <div
                                                className="px-2 py-1 font-medium uppercase tracking-wide text-gray-600 rounded-t-md flex items-center justify-between">
                                                <span>{agent || "agent"}</span>
                                                {/* Only show when completed */}
                                                {secs != null ? (
                                                    <span className="text-gray-500">{formatSeconds(secs)}</span>
                                                ) : null}
                                            </div>
                                            <div>
                                                <ReactMarkdown
                                                    remarkPlugins={remarkPlugins}
                                                    rehypePlugins={rehypePlugins}
                                                    components={markdownComponentsTight}
                                                    skipHtml={false}
                                                >
                                                    {closeUpMarkdown(item.content.agents[agent] || "")}
                                                </ReactMarkdown>
                                            </div>
                                        </div>
                                    );
                                })}
                            </div>
                        )}
                    </div>
                )}
            </div>
        )
    }, [active, agentKeys, durSec, expanded, getAgentSecondsIfCompleted, item.content.agents])
}