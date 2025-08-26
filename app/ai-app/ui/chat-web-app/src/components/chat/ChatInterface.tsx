/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2025 Elena Viter
 */

import {
    AlertCircle,
    CheckCircle2,
    ChevronDown,
    ChevronUp,
    Circle,
    CircleChevronUp,
    Database,
    FileText,
    Loader,
    MessageSquare,
    Play,
    Search,
    User,
    Zap
} from "lucide-react";
import React, {Fragment, useEffect, useRef, useState} from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";
import {AssistantChatStep, ChatLogItem, ChatMessage, UserChatMessage} from "./types/chat.ts";

interface ChatInterfaceProps {
    chatLogItems: ChatLogItem[];
    onSendMessage: (message: string) => Promise<void>;
    userInputEnabled?: boolean;
    isProcessing?: boolean;
    inputPlaceholder?: string;
    aiModelName?: string;
    showMetadata?: boolean;
}

const ChatInterface = ({
                           chatLogItems,
                           onSendMessage,
                           aiModelName,
                           isProcessing,
                           inputPlaceholder = "Ask me anything...",
                           userInputEnabled = true,
                           showMetadata = false,
                       }: ChatInterfaceProps) => {

    const [userInput, setUserInput] = useState<string>('');
    const chatRef = useRef<HTMLDivElement | null>(null);
    const [expandedItems, setExpandedItems] = useState<Map<ChatLogItem, boolean>>(new Map());
    useEffect(() => {
        if (chatRef.current) chatRef.current.scrollTop = chatRef.current.scrollHeight;
    }, [chatLogItems]);

    const sendMessage = () => {
        onSendMessage(userInput)
            .then(() => {
                setUserInput("");
            })
            .catch(() => {

            })
            .finally(() => {

            });

    }

    const onUserInputKeyDown = (e: React.KeyboardEvent): void => {
        if (userInputEnabled && e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    }

    const renderMessage = (message: ChatMessage) => {
        const isUserMessage = message instanceof UserChatMessage;
        const renderUserMessage = () => {
            return (<div
                key={message.id}
                className='flex justify-end'
            >
                <div className="flex flex-row p-3 rounded-2xl bg-gray-200 text-black">
                    <p className="text-sm leading-relaxed whitespace-pre-wrap pt-1">{message.text}</p>
                    <div
                        className="w-8 h-8 rounded-full bg-gray-300 ml-3 flex items-center justify-center flex-shrink-0">
                        <User size={16} className="text-gray-600"/>
                    </div>
                </div>
            </div>)
        }
        const renderAssistantMessage = () => {
            return (
                <div
                    key={message.id}
                    className='flex justify-start'
                >

                    <div className="flex flex-col">
                        <div
                            className={`p-3 ${message.isError
                                ? 'text-red-800'
                                : 'text-gray-800'
                            }`}
                        >
                            <ReactMarkdown
                                remarkPlugins={[remarkGfm]}
                                // @ts-expect-error it's ok
                                rehypePlugins={[rehypeHighlight]}
                                components={{
                                    code({inline, className, children, ...props}) {
                                        return inline ? (
                                            <code
                                                className="px-1 py-0.5 rounded bg-gray-100" {...props}>{children}</code>
                                        ) : (
                                            <pre className="p-3 rounded bg-gray-900 text-gray-100 overflow-auto">
            <code className={className} {...props}>{children}</code>
          </pre>
                                        );
                                    },
                                    a({children, ...props}) {
                                        return <a className="text-blue-600 underline" target="_blank"
                                                  rel="noreferrer" {...props}>{children}</a>;
                                    }
                                }}
                            >
                                {message.text}
                            </ReactMarkdown>
                        </div>

                        {showMetadata && message.metadata && (
                            <div className="mt-2 text-xs text-gray-500 space-y-1">
                                {message.metadata.is_our_domain !== undefined && (
                                    <div className="flex items-center">
                                        <span className="mr-2">Classification:</span>
                                        <span className={`px-2 py-1 rounded text-xs ${
                                            message.metadata.is_our_domain
                                                ? 'bg-green-100 text-green-700'
                                                : 'bg-gray-100 text-gray-700'
                                        }`}>
                                                            {message.metadata.is_our_domain ? 'Our Domain' : 'Not Our Domain'}
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
                    </div>


                </div>
            )
        }

        return isUserMessage ? renderUserMessage() : renderAssistantMessage()
    }

    const getStepName = (step: AssistantChatStep): string => {
        return step.title || step.step.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase());
    }

    const getStepIcon = (step: AssistantChatStep, iconSize = 14, className = "m-auto"): React.ReactNode => {
        switch (step.status) {
            case 'started' :
                return <Loader size={iconSize} className={`animate-spin ${className}`}/>
            case 'error' :
                return <AlertCircle size={iconSize} className={className}/>
        }
        switch (step.step) {
            case 'classifier':
                return <Zap size={iconSize} className={className}/>;
            case 'query_writer':
                return <FileText size={iconSize} className={className}/>;
            case 'rag_retrieval':
                return <Database size={iconSize} className={className}/>;
            case 'reranking':
                return <Search size={iconSize} className={className}/>;
            case 'answer_generator':
                return <MessageSquare size={iconSize} className={className}/>;
            case 'workflow_start':
                return <Play size={iconSize} className={className}/>;
            case 'workflow_complete':
                return <CheckCircle2 size={iconSize} className={className}/>;
            default:
                return <Circle size={iconSize} className={className}/>;
        }
    }

    const getStepColor = (step: AssistantChatStep): string => {
        switch (step.status) {
            case 'completed':
                return 'text-green-600 ';
            case 'started':
                return 'text-blue-600';
            case 'error':
                return 'text-red-600';
            default:
                return 'text-gray-600';
        }
    }

    const renderChatMessageGroup = (messageGroup: ChatMessage[]) => {
        const result = []
        for (const message of messageGroup) {
            result.push(renderMessage(message));
        }
        return result
    }

    const renderStepMarkdown = (markdown: string) => {
        return (
            <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                // @ts-expect-error it's ok
                rehypePlugins={[rehypeHighlight]}
                components={{
                    code({inline, className, children, ...props}) {
                        return inline ? (
                            <code
                                className="px-1 py-0.5 rounded bg-gray-100" {...props}>{children}</code>
                        ) : (
                            <pre className="p-3 rounded bg-gray-900 text-gray-100 overflow-auto">
            <code className={className} {...props}>{children}</code>
          </pre>
                        );
                    },
                    a({children, ...props}) {
                        return <a className="text-blue-600 underline" target="_blank"
                                  rel="noreferrer" {...props}>{children}</a>;
                    }
                }}
            >
                {markdown}
            </ReactMarkdown>
        )
    }

    const renderAssistantChatStepGroup = (messageGroup: AssistantChatStep[], groupIndex: number) => {
        const result = messageGroup.map((v, i) => {
            const markdown = v.getMarkdown()
            const isLastItem = i === messageGroup.length - 1
            const isExpanded = expandedItems.has(v) ? expandedItems.get(v) : chatLogItems[chatLogItems.length - 1] === v
            const onExpandClick = () => {
                setExpandedItems(prev => new Map(prev).set(v, !isExpanded));
            }
            return (
                <Fragment key={i}>
                    <div className={`flex flex-row text-sm ${getStepColor(v)}`}>
                        <div className="flex w-6 h-6">
                            {getStepIcon(v)}
                        </div>
                        <span className="my-auto font-bold">
                                {getStepName(v)}
                            </span>
                        {markdown && (
                            <div className="flex w-4 h-6 cursor-pointer" onClick={onExpandClick}>
                                {isExpanded ?
                                    <ChevronUp size={16} className="m-auto"/> :
                                    <ChevronDown size={16} className="m-auto"/>
                                }
                            </div>
                        )}
                        <div/>
                    </div>
                    <div className={`flex flex-row text-sm ${getStepColor(v)}`}>
                        <div className={`w-3 ml-3${isLastItem ? "" : " border-l-2 border-dotted"}`}/>
                        {markdown && isExpanded && <span className="mb-1">{renderStepMarkdown(markdown)}</span>}
                        {!isLastItem && <div className="h-2"/>}
                    </div>
                </Fragment>
            )
        })
        return [(<div key={`chat-log-item-group-${groupIndex}`} className="flex flex-col pl-2">{result}</div>)]
    }

    const renderChatLogItems = (items: ChatLogItem[]) => {
        const groupMessages = () => {
            const groups: ChatLogItem[][] = []
            let currentGroupType;
            let currentGroup: ChatLogItem[] = [];
            for (const item of items) {
                if (!currentGroupType || !(item instanceof currentGroupType)) {
                    if (currentGroup.length)
                        groups.push(currentGroup)
                    currentGroup = []
                    currentGroupType = item.constructor
                }
                currentGroup.push(item)
            }
            groups.push(currentGroup)
            return groups
        }

        const result = []
        const groups = groupMessages()
        for (let i = 0; i < groups.length; i++) {
            const group = groups[i]
            if (group[0] instanceof ChatMessage) {
                result.push(...renderChatMessageGroup(group as ChatMessage[]))
            } else if (group[0] instanceof AssistantChatStep) {
                result.push(...renderAssistantChatStepGroup(group as AssistantChatStep[], i))
            }
        }

        return result
    }

    const renderChatLog = () => {
        return (<div
            ref={chatRef}
            className="flex flex-1 overflow-y-auto px-6 py-4 space-y-4"
            id="ChatLog"
        >
            <div className="w-3/5 mx-auto">
                {renderChatLogItems(chatLogItems)}

                {isProcessing && (
                    <div className="flex justify-start">
                        <div
                            className="w-8 h-8 rounded-full bg-gradient-to-br from-blue-500 to-purple-600 mr-3 flex items-center justify-center flex-shrink-0">
                            <Loader size={16} className="text-white animate-spin"/>
                        </div>
                        <div
                            className="bg-white text-gray-800 border border-gray-200 shadow-sm p-4 rounded-lg">
                            <div className="flex items-center text-gray-600">
                                <Loader size={16} className="animate-spin mr-2"/>
                                <span
                                    className="text-sm">Processing with {aiModelName || 'AI model'}...</span>
                            </div>
                        </div>
                    </div>
                )}
                <div className="h-28"/>
            </div>
        </div>)
    }

    const userInputFieldRef = useRef(null)

    const renderUserInput = () => {
        const inputDisabled = !userInputEnabled
        const sendButtonDisabled = inputDisabled || isProcessing || !userInput.trim()
        
        return (
            <div id="UserInput"
                 className="flex items-center border-gray-200 bottom-0 absolute w-full cursor-text"
                 onClick={() => {userInputFieldRef.current?.focus()}}
            >
                <div
                    className="flex flex-col items-end w-3/5 mx-auto mb-6 border rounded-2xl border-gray-200 bg-white shadow-sm">
                    <div className="flex max-h-72 min-h-6 w-full">
                        <textarea
                            value={userInput}
                            onChange={(e) => setUserInput(e.target.value)}
                            onKeyDown={onUserInputKeyDown}
                            placeholder={inputPlaceholder}
                            disabled={inputDisabled}
                            className="flex-1 m-3 border-gray-300 resize-none grow field-sizing-content focus:outline-none overflow-y-auto"
                            rows={2}
                            ref={userInputFieldRef}
                        />

                    </div>
                    <button
                        onClick={sendMessage}
                        disabled={sendButtonDisabled}
                        className={`mb-3 mr-3 rounded-lg font-medium transition-colors mt-auto`}
                    >
                        <CircleChevronUp size={18} className="cursor-pointer"/>
                    </button>
                </div>
            </div>
        )
    }

    return (
        <div id={ChatInterface.name} className='flex-1 flex flex-col min-h-1/3 bg-gray-50'>
            {renderChatLog()}
            {renderUserInput()}
        </div>
    )

}

export default ChatInterface