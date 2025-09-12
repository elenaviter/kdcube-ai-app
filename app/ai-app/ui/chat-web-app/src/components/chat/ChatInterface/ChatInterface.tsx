// ChatInterface.tsx
import {
    AlertCircle,
    CheckCircle2,
    ChevronDown,
    ChevronUp,
    Circle,
    CircleChevronUp,
    CirclePlus,
    ClipboardCopy,
    Database,
    File,
    FileText,
    Loader,
    MessageSquare,
    Play,
    Search, Send,
    User,
    X,
    Zap,
} from "lucide-react";
import React, {createContext, CSSProperties, Fragment, useContext, useEffect, useMemo, useRef, useState} from "react";

import ReactMarkdown from "react-markdown";

import {
    AssistantChatStep,
    AssistantThinkingItem,
    ChatLogItem,
    ChatMessage,
    DownloadItem,
    UserChatMessage
} from "../types/chat.ts";
import {Hint} from "../../Hints.tsx";
import {copyMarkdownToClipboard} from "../../Clipboard.ts";

import {getFileIcon} from "../../FileIcons.tsx";
import {markdownComponents, markdownComponentsTight, rehypePlugins, remarkPlugins} from "./markdownRenderUtils.tsx";
import {selectFileAdvanced} from "../../shared.ts";

interface ChatInterfaceProps {
    lockMessage?: string;
    inputPlaceholder?: string;
    showMetadata?: boolean;
    maxWidth?: number | string;
}

/* ---------- helpers (no hooks) ---------- */
const getStepName = (step: AssistantChatStep): string =>
    step.title || step.step.replace("_", " ").replace(/\b\w/g, (l) => l.toUpperCase());

const getStepIcon = (step: AssistantChatStep, iconSize = 14, className = "m-auto"): React.ReactNode => {
    switch (step.status) {
        case "started":
            return <Loader size={iconSize} className={`animate-spin ${className}`}/>;
        case "error":
            return <AlertCircle size={iconSize} className={className}/>;
    }
    switch (step.step) {
        case "classifier":
            return <Zap size={iconSize} className={className}/>;
        case "query_writer":
            return <FileText size={iconSize} className={className}/>;
        case "rag_retrieval":
            return <Database size={iconSize} className={className}/>;
        case "reranking":
            return <Search size={iconSize} className={className}/>;
        case "answer_generator":
            return <MessageSquare size={iconSize} className={className}/>;
        case "workflow_start":
            return <Play size={iconSize} className={className}/>;
        case "workflow_complete":
            return <CheckCircle2 size={iconSize} className={className}/>;
        default:
            return <Circle size={iconSize} className={className}/>;
    }
};

const getStepColor = (step: AssistantChatStep): string => {
    switch (step.status) {
        case "completed":
            return "text-green-600 ";
        case "started":
            return "text-blue-600";
        case "error":
            return "text-red-600";
        default:
            return "text-gray-600";
    }
};

const formatSeconds = (sec: number): string => {
    if (!isFinite(sec) || sec < 0) sec = 0;
    if (sec < 60) return `${Math.round(sec * 10) / 10}s`;
    const m = Math.floor(sec / 60);
    const s = Math.round((sec % 60) * 10) / 10;
    const sStr = (s < 10 ? "0" : "") + s.toFixed(1);
    return `${m}:${sStr}`;
};

const DownloadItemsPanel = ({items, onClick}: { items: DownloadItem[], onClick?: (item: DownloadItem) => void; }) => {
    if (!items || !items.length) return null;
    return (
        <div className="flex justify-start">
            <div className="w-full flex flex-row flex-wrap">
                {items.map((item, index) => (
                    <div key={index}>
                        <button
                            className="m-2 text-gray-700 flex items-center text-sm cursor-pointer hover:text-black hover:underline"
                            onClick={() => onClick && onClick(item)}>
                            <span className="inline-block mr-1">{getFileIcon(item.filename, 24, item.mimeType)}</span>
                            <span className="inline-block">{item.filename}</span>
                        </button>
                    </div>
                ))}
            </div>
        </div>
    )
}

/* ---------- child component to keep hooks outside loops ---------- */
const StepItem: React.FC<{
    step: AssistantChatStep;
    isLast: boolean;
    defaultExpanded: boolean;
}> = ({step, isLast, defaultExpanded}) => {
    const [isExpanded, setIsExpanded] = useState(defaultExpanded);
    useEffect(() => setIsExpanded(defaultExpanded), [defaultExpanded]);

    const markdown = step.getMarkdown();

    return (
        <Fragment>
            <div className={`flex flex-row text-sm ${getStepColor(step)}`}>
                <div className="flex w-6 h-6">{getStepIcon(step)}</div>
                <span className="my-auto font-bold">{getStepName(step)}</span>
                {markdown && (
                    <div className="flex w-4 h-6 cursor-pointer" onClick={() => setIsExpanded(v => !v)}>
                        {isExpanded ? <ChevronUp size={16} className="m-auto"/> :
                            <ChevronDown size={16} className="m-auto"/>}
                    </div>
                )}
                <div/>
            </div>

            <div className={`flex flex-row text-sm ${getStepColor(step)}`}>
                <div className={`w-3 ml-3${isLast ? "" : " border-l-2 border-dotted"}`}/>
                {markdown && isExpanded && (
                    <div
                        className="
              mb-1 not-prose
              leading-[1.4]
              [&_*]:leading-[1.4]
              [&_p]:my-1 [&_p:last-child]:mb-0
              [&_ul]:my-1 [&_ol]:my-1
              [&_li]:my-0.5
              [&_blockquote]:my-1
            "
                    >
                        <ReactMarkdown
                            remarkPlugins={remarkPlugins}
                            rehypePlugins={rehypePlugins as any}
                            components={markdownComponentsTight}
                            linkTarget="_blank"
                            skipHtml={false}
                        >
                            {markdown}
                        </ReactMarkdown>
                    </div>
                )}
                {!isLast && <div className="h-2"/>}
            </div>
        </Fragment>
    );
};
/** ---------- THINKING ITEM (per turn, rows per agent, markdown, show duration only after completed) ---------- */
const ThinkingItem: React.FC<{ item: AssistantThinkingItem }> = ({item}) => {
    const [open, setOpen] = useState(false); // collapsed by default
    const toggle = () => setOpen(v => !v);
    const onKeyDown = (e: React.KeyboardEvent) => {
        if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            toggle();
        }
    };

    // keep the global header ticking while overall thinking is active
    const [, setTick] = useState(0);
    useEffect(() => {
        if (!item.active) return;
        const t = setInterval(() => setTick(v => (v + 1) % 1000), 500);
        return () => clearInterval(t);
    }, [item.active]);

    const active = item.active !== false;
    const endedAt = item.endedAt;
    const endMs = active ? Date.now() : endedAt?.getTime() ?? Date.now();
    const durSec = Math.max(0, Math.round(((endMs - item.timestamp.getTime()) / 1000) * 10) / 10);

    const agentKeys = useMemo(() => Object.keys(item.agents || {}), [item.agents]);

    const getAgentSecondsIfCompleted = (agent: string): number | null => {
        const meta = item.agentTimes?.[agent];
        if (!meta?.endedAt || !meta?.startedAt) return null; // ← only show after completed
        const s = (meta.endedAt.getTime() - meta.startedAt.getTime()) / 1000;
        return Math.max(0, Math.round(s * 10) / 10);
    };

    return (
        <div className="w-full">
            {/* tiny CSS for animated gradient text (scoped) */}
            <style>{`
        @keyframes sheen {
          0% { background-position: 0% 50%; }
          100% { background-position: 200% 50%; }
        }
        .thinking-animated {
          background: linear-gradient(90deg, #9ca3af, #6b7280, #9ca3af);
          background-size: 200% 100%;
          -webkit-background-clip: text;
          background-clip: text;
          color: transparent;
          animation: sheen 2s linear infinite;
        }
      `}</style>

            <div
                className={`flex items-center justify-between px-3 py-2 rounded-lg border border-gray-400 cursor-pointer select-none
                    ${active ? "bg-gray-100" : "bg-gray-50"}`}
                onClick={toggle}
                onKeyDown={onKeyDown}
                role="button"
                tabIndex={0}
                aria-expanded={open}
                aria-controls={`thinking-body-${item.id}`}
                title={open ? "Collapse" : "Expand"}
            >
                <div className="flex items-center gap-2">
                    <span className={`text-sm font-medium ${active ? "thinking-animated" : "text-gray-700"}`}>
                        {active ? "Thinking" : `Thought for ${formatSeconds(durSec)}`}
                    </span>
                </div>
                <span className="text-gray-500">
                    {open ? <ChevronUp size={16}/> : <ChevronDown size={16}/>}
                </span>
            </div>

            {open && (
                <div
                    id={`thinking-body-${item.id}`}
                    className="px-3 py-2 text-xs text-slate-700 bg-gray-50 border border-t-0 border-gray-400 rounded-b-lg"
                >
                    {agentKeys.length === 0 ? (
                        <div className="text-gray-500 italic">No thoughts yet…</div>
                    ) : (
                        <div className="space-y-3">
                            {agentKeys.map((agent) => {
                                const secs = getAgentSecondsIfCompleted(agent);
                                return (
                                    <div key={agent} className="rounded-md border border-gray-400 bg-white">
                                        <div
                                            className="px-2 py-1 text-[10px] font-medium uppercase tracking-wide text-gray-600 bg-gray-100 rounded-t-md flex items-center justify-between">
                                            <span>{agent || "agent"}</span>
                                            {/* Only show when completed */}
                                            {secs != null ? (
                                                <span className="text-[10px] text-gray-500">{formatSeconds(secs)}</span>
                                            ) : null}
                                        </div>
                                        <div
                                            className="
    px-3 py-2 text-[12px] not-prose
    leading-[1.4] [&_*]:leading-[1.4]
    [&_p]:my-1 [&_p:last-child]:mb-0
    [&_ul]:my-1 [&_ol]:my-1
    [&_li]:my-0.5
    [&_blockquote]:my-1
  "
                                        >
                                            <ReactMarkdown
                                                remarkPlugins={remarkPlugins}
                                                rehypePlugins={rehypePlugins as any}
                                                components={markdownComponentsTight}
                                                linkTarget="_blank"
                                                skipHtml={false}
                                            >
                                                {item.agents[agent] || ""}
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
    );
};

const AssistantMessage = (
    {message, showMetadata}:
    {
        message: ChatMessage,
        showMetadata: boolean,
    }
) => {
    const mdRed = useRef<HTMLDivElement>(null);
    const copyClick = () => {
        copyMarkdownToClipboard(message.text, mdRed.current?.innerHTML);
    }

    return (
        <div key={message.id} className="flex justify-start">
            <div className="flex flex-col w-full">
                <div
                    className={`px-3 pt-2 ${message.isError ? "text-red-800" : "text-gray-800"} prose max-w-none prose-p:my-2 prose-ul:my-2 prose-ol:my-2`}
                    ref={mdRed}>
                    <ReactMarkdown
                        remarkPlugins={remarkPlugins}
                        rehypePlugins={rehypePlugins}
                        components={markdownComponents}
                        linkTarget="_blank"
                        skipHtml={false}
                    >
                        {message.text}
                    </ReactMarkdown>
                </div>

                {showMetadata && message.metadata && (
                    <div className="mt-2 text-xs text-gray-500 space-y-1">
                        {message.metadata.is_our_domain !== undefined && (
                            <div className="flex items-center">
                                <span className="mr-2">Classification:</span>
                                <span
                                    className={`px-2 py-1 rounded text-xs ${
                                        message.metadata.is_our_domain ? "bg-green-100 text-green-700" : "bg-gray-100 text-gray-700"
                                    }`}
                                >
                                        {message.metadata.is_our_domain ? "Our Domain" : "Not Our Domain"}
                                    </span>
                            </div>
                        )}
                        {message.metadata.retrieved_docs && message.metadata.retrieved_docs > 0 && (
                            <div className="flex items-center">
                                <Database size={12} className="mr-1"/>
                                <span>KB Search: {message.metadata.retrieved_docs} documents retrieved</span>
                            </div>
                        )}
                    </div>
                )}

                <div
                    className="flex flex-row space-x-2 w-full justify-start transition-all duration-300 ease-out pl-3 pb-3"
                >
                    <Hint content="Copied" trigger="click" autohideDelay={2000} className={"text-nowrap"}>
                        <Hint content="Copy to clipboard">
                            <button className="cursor-pointer" onClick={copyClick}>
                                <ClipboardCopy size={16} className="mt-0.5 text-gray-400 hover:text-gray-600"/>
                            </button>
                        </Hint>
                    </Hint>
                </div>
            </div>
        </div>
    )
}

export interface ChatInterfaceContextValue {
    chatLogItems?: ChatLogItem[];
    isLocked?: boolean;
    userInputEnabled?: boolean;
    isProcessing?: boolean;
    followUpQuestion?: string[];
    onSendMessage?: (message: string, files?: File[]) => Promise<void>;
    onDownloadItemClick?: (downloadItem: DownloadItem) => void;
}

export const ChatInterfaceContext = createContext<ChatInterfaceContextValue>({})

const ChatInterface = ({
                           inputPlaceholder = "Ask me anything...",
                           lockMessage,
                           showMetadata = false,
                           maxWidth,
                       }: ChatInterfaceProps) => {
    const [userInput, setUserInput] = useState<string>("");
    const [userInputFiles, setUserInputFiles] = useState<File[]>([]);
    const logContainerRef = useRef<HTMLDivElement | null>(null);
    const userInputFieldRef = useRef<HTMLTextAreaElement | null>(null);
    const contentRef = useRef<HTMLDivElement | null>(null);

    // auto-scroll when near bottom
    const autoScroll = useRef(true);

    const {
        chatLogItems,
        followUpQuestion,
        isProcessing,
        isLocked,
        userInputEnabled,
        onSendMessage,
        onDownloadItemClick
    } = useContext(ChatInterfaceContext);

    const addInputFiles = (files: File[]) => {
        setUserInputFiles((prevState) => {
            const newState = [...prevState]
            files.forEach((newFile) => {
                const i = newState.findIndex((file) => file.name === newFile.name)
                if (i >= 0) {
                    newState[i] = newFile;
                } else {
                    newState.push(newFile);
                }
            })
            return newState;
        })
    }

    const removeInputFiles = (files: File[]) => {
        setUserInputFiles((prevState) => {
            const newState = [...prevState]
            files.forEach((newFile) => {
                const i = newState.findIndex((file) => file.name === newFile.name)
                if (i >= 0) {
                    newState.splice(i, 1)
                }
            })
            return newState;
        })
    }

    useEffect(() => {
        if (autoScroll.current && logContainerRef.current) {
            logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
        }
    }, [chatLogItems, isProcessing]);

    const onScroll = () => {
        const el = logContainerRef.current;
        if (!el) return;
        const threshold = 60;
        const atBottom = el.scrollHeight - (el.scrollTop + el.clientHeight) < threshold;
        autoScroll.current = atBottom;
    };

    const sendMessage = (message?: string) => {
        message = message || userInput.trim();
        if (!message && userInputFiles.length < 1) return;
        onSendMessage(message, userInputFiles).then(() => {
            setUserInput("")
            setUserInputFiles([])
        });
    };

    const renderMessage = (message: ChatMessage) => {
        const isUserMessage = message instanceof UserChatMessage;

        const renderUserMessage = () => (
            <div key={message.id} className="flex justify-end">
                <div className="flex flex-row p-3 rounded-2xl bg-gray-200 text-black">
                    <div className="flex flex-col">
                        {(message as UserChatMessage).attachments && (<div className="flex flex-row gap-1 flex-wrap">
                            {(message as UserChatMessage).attachments?.map((attachment: File) => (
                                <div
                                    className="flex items-center border-2 px-2 py-1 rounded-xl border-gray-300 bg-gray-100">{getFileIcon(attachment.name, 18, undefined, "mr-1")}{attachment.name}</div>
                            ))}
                        </div>)}
                        {message.text &&
                            <p className="text-sm leading-relaxed whitespace-pre-wrap pt-1">{message.text}</p>}
                    </div>
                    <div
                        className="w-8 h-8 rounded-full bg-gray-300 ml-3 flex items-center justify-center flex-shrink-0">
                        <User size={16} className="text-gray-600"/>
                    </div>
                </div>
            </div>
        );

        return isUserMessage ? renderUserMessage() :
            <AssistantMessage message={message} showMetadata={showMetadata}/>;
    };

    const renderDownloadItem = (items: DownloadItem[], i: number) => {
        return [<DownloadItemsPanel items={items} key={i} onClick={onDownloadItemClick}/>]
    }

    const renderChatMessageGroup = (messageGroup: ChatMessage[]) => messageGroup.map(renderMessage);

    const renderAssistantChatStepGroup = (messageGroup: AssistantChatStep[], groupIndex: number) => {
        return [
            <div key={`chat-log-item-group-${groupIndex}`} className="flex flex-col pl-2">
                {messageGroup.map((v, i) => (
                    <StepItem
                        key={`${groupIndex}-${i}-${v.step}`}
                        step={v}
                        isLast={i === messageGroup.length - 1}
                        defaultExpanded={i === messageGroup.length - 1}
                    />
                ))}
            </div>,
        ];
    };

    const renderAssistantThinkingGroup = (group: AssistantThinkingItem[], groupIndex: number) => {
        return [
            <div key={`thinking-group-${groupIndex}`} className="flex flex-col">
                {group.map((it, i) => (
                    <ThinkingItem key={`thinking-${groupIndex}-${i}-${it.id}`} item={it}/>
                ))}
            </div>,
        ];
    };

    const renderChatLogItems = (items: ChatLogItem[]) => {
        // group consecutive items of same type
        const groups: ChatLogItem[][] = [];
        let currentType: any;
        let group: ChatLogItem[] = [];
        for (const item of items) {
            if (!currentType || !(item instanceof currentType)) {
                if (group.length) groups.push(group);
                group = [];
                currentType = item.constructor;
            }
            group.push(item);
        }
        if (group.length) groups.push(group);

        const out: React.ReactNode[] = [];
        groups.forEach((g, i) => {
            if (g[0] instanceof ChatMessage) out.push(...renderChatMessageGroup(g as ChatMessage[]));
            else if (g[0] instanceof AssistantChatStep) out.push(...renderAssistantChatStepGroup(g as AssistantChatStep[], i));
            else if (g[0] instanceof AssistantThinkingItem) out.push(...renderAssistantThinkingGroup(g as AssistantThinkingItem[], i));
            else if (g[0] instanceof DownloadItem) out.push(...renderDownloadItem(g as DownloadItem[], i));
        });
        return out;
    };

    const renderFollowUpQuestions = () => {
        const disabled = !userInputEnabled || isProcessing;
        const stateClass = disabled ? "cursor-auto text-gray-400" : "cursor-pointer hover:border-gray-400 hover:bg-slate-400"
        if (!isProcessing && followUpQuestion) {
            return (
                <div className="flex flex-row items-start w-full flex-wrap space-x-1 space-y-4">
                    {followUpQuestion.map((q, i) => {
                        return (<button key={`follow-up-question-${i}`} onClick={() => {
                            sendMessage(q)
                        }} disabled={disabled}>
                            <span
                                className={`text-nowrap rounded-2xl p-2 border border-gray-400 bg-slate-100  ${stateClass}`}>{q}</span>
                        </button>)
                    })}
                </div>
            )
        }
        return null
    }

    const renderProcessing = () => {
        return isProcessing ? (
            <div className="flex items-center text-gray-500 text-xs mt-2">
                <Loader size={14} className="animate-spin mr-2"/>
                <span>Working…</span>
            </div>
        ) : null
    }

    const renderChatLog = () => {
        const elementStyle: CSSProperties = {}
        if (maxWidth)
            elementStyle.width = typeof maxWidth === 'number' ? `${maxWidth}px` : maxWidth;
        return (
            <div
                className="h-full w-full"
                id="ChatLog"
            >
                <div
                    className="h-full w-full overflow-x-auto"
                    ref={logContainerRef}
                    onScroll={onScroll}
                >
                    <div className="border-r border-l border-gray-200 mx-auto min-h-full bg-slate-50" style={elementStyle}>
                        <div className="px-10 py-4" >
                            {renderChatLogItems(chatLogItems)}
                            {renderProcessing()}
                            {renderFollowUpQuestions()}
                        </div>
                        <div className="pb-22"/>
                    </div>
                </div>

            </div>
        )
    };

    const renderUserInput = () => {
        const inputDisabled = !userInputEnabled || isLocked;
        // const inputDisabled = false;
        const sendButtonDisabled = inputDisabled || isProcessing || (!userInput.trim() && userInputFiles.length == 0);
        const elementStyle: CSSProperties = {}
        if (maxWidth)
            elementStyle.width = typeof maxWidth === 'number' ? `${maxWidth}px` : maxWidth;

        return (
            <div
                id="UserInput"
                className="absolute -left-2 z-30 bottom-0 w-full"
                onClick={() => userInputFieldRef.current?.focus()}
            >
                <div className="pointer-events-none mx-auto px-8" style={elementStyle}>
                    <div
                        className={`flex flex-col mx-auto border rounded-t-xl border-gray-400 shadow-sm pointer-events-auto ${isLocked ? "bg-yellow-50" : "bg-white"}`}
                    >
                        {userInputFiles && userInputFiles.length > 0 &&
                            <div className="flex flex-row flex-wrap p-3 gap-1">
                                {
                                    userInputFiles.map((file, i) => {
                                        return (<div
                                            className="flex border-2 border-gray-400 bg-gray-50 rounded-2xl px-3 py-1 items-center"
                                            key={`input-file-${i}`}
                                        >
                                            <span>{file.name}</span>
                                            <button
                                                className="pl-1 text-gray-400 hover:text-gray-600 cursor-pointer"
                                                onClick={() => {
                                                    removeInputFiles([file])
                                                }}
                                            >
                                                <X size={12}/>
                                            </button>
                                        </div>)
                                    })
                                }
                            </div>}
                        <div className="flex max-h-72 min-h-12 w-full">
                            {isLocked ? (
                                <div className="flex-1 m-3 flex flex-col items-center">
                                    <span
                                        className="font-semibold text-gray-400">{lockMessage || "Daily token limit reached. Please try again later."} </span>
                                </div>
                            ) : (
                                <textarea
                                    value={userInput}
                                    onChange={(e) => setUserInput(e.target.value)}
                                    onKeyDown={(e) => {
                                        console.log(e.key, userInputEnabled)
                                        if (userInputEnabled && e.key === "Enter" && !e.shiftKey) {
                                            e.preventDefault();
                                            sendMessage();
                                        }
                                    }}
                                    placeholder={inputPlaceholder}
                                    disabled={inputDisabled}
                                    className="flex-1 m-3 resize-none grow field-sizing-content focus:outline-none overflow-y-auto"
                                    rows={2}
                                    ref={userInputFieldRef}
                                />
                            )}

                        </div>
                        <div className="flex">
                            <div className="pl-2"/>
                            <button
                                onClick={() => {
                                    selectFileAdvanced({multiple: true}).then((res) => {
                                        addInputFiles(res)
                                    })
                                }}
                                disabled={inputDisabled}
                                className=" mb-3 rounded-lg font-medium text-gray-600 hover:text-gray-900 disabled:text-gray-300"
                                aria-label="Add file"
                                title="Add file"
                            >
                                <CirclePlus size={18} className={`${inputDisabled ?
                                    (isProcessing ? "cursor-wait" : "cursor-auto") :
                                    "cursor-pointer"}`}/>
                            </button>
                            <button
                                onClick={() => {
                                    sendMessage()
                                }}
                                disabled={sendButtonDisabled}
                                className="mb-3 mr-3 rounded-lg font-medium text-gray-600 hover:text-gray-900 disabled:text-gray-300 ml-auto"
                                aria-label="Send message"
                                title="Send"
                            >
                                <Send size={18} className={`${sendButtonDisabled ?
                                    (userInput.trim() || userInputFiles.length > 0 ? "cursor-wait" : "cursor-auto") :
                                    "cursor-pointer"}`}/>
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        );
    };

    return (
        <div id={ChatInterface.name}
             ref={contentRef}
             className="flex-1 flex flex-col bg-slate-100 min-h-0 min-w-0 transition-all duration-100 ease-out w-full relative"
        >
            {renderChatLog()}
            {renderUserInput()}
        </div>
    );
};

export default ChatInterface;
