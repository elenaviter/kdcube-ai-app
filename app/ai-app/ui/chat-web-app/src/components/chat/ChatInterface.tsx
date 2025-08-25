/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2025 Elena Viter
 */

import {
    AlertCircle,
    Bot,
    CheckCircle2,
    Circle,
    Database,
    EllipsisVertical,
    FileText,
    Loader,
    MessageSquare,
    Play,
    Search,
    Send,
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
}

const ChatInterface = ({
                           chatLogItems,
                           onSendMessage,
                           aiModelName,
                           isProcessing,
                           inputPlaceholder = "Ask me anything...",
                           userInputEnabled = true
                       }: ChatInterfaceProps) => {

    const [userInput, setUserInput] = useState<string>('');
    const chatRef = useRef<HTMLDivElement | null>(null);
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
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    }

    const renderMessage = (message: ChatMessage) => {
        const isUserMessage = message instanceof UserChatMessage;
        return (
            <div
                key={message.id}
                className={`flex ${isUserMessage ? 'justify-end' : 'justify-start'}`}
            >
                {!isUserMessage && (
                    <div
                        className="w-8 h-8 rounded-full bg-gradient-to-br from-blue-500 to-purple-600 mr-3 flex items-center justify-center flex-shrink-0">
                        <Bot size={16} className="text-white"/>
                    </div>
                )}

                <div className="flex flex-col max-w-3xl">
                    <div
                        className={`p-4 rounded-lg ${
                            isUserMessage
                                ? 'bg-blue-500 text-white'
                                : message.isError
                                    ? 'bg-red-50 text-red-800 border border-red-200'
                                    : 'bg-white text-gray-800 border border-gray-200 shadow-sm'
                        }`}
                    >
                        {!isUserMessage ? (
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
                        ) : (
                            <p className="text-sm leading-relaxed whitespace-pre-wrap">{message.text}</p>
                        )}
                    </div>

                    {/* Message metadata */}
                    {message.metadata && (
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

                {isUserMessage && (
                    <div
                        className="w-8 h-8 rounded-full bg-gray-300 ml-3 flex items-center justify-center flex-shrink-0">
                        <User size={16} className="text-gray-600"/>
                    </div>
                )}
            </div>
        )
    }

    const getStepName = (stepName: string): string => {
        const names: Record<string, string> = {
            'classifier': 'Domain Classification',
            'query_writer': 'Query Generation',
            'rag_retrieval': 'Document Retrieval',
            'reranking': 'Document Reranking',
            'answer_generator': 'Answer Generation',
            'workflow_start': 'Starting Workflow',
            'workflow_complete': 'Workflow Complete',
            'workflow_error': 'Workflow Error'
        };
        return names[stepName] || stepName.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase());
    }

    const getStepIcon = (step: AssistantChatStep, iconSize = 14, className = "m-auto"): React.ReactNode => {
        switch (step.status) {
            case 'started' :
                return <Loader size={16} className={`animate-spin ${className}`}/>
            case 'error' :
                return <AlertCircle size={16} className={className}/>
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

    const renderAssistantChatStepGroup = (messageGroup: AssistantChatStep[]) => {
        const result = messageGroup.map((v, i) => {
            return (
                <Fragment key={i}>
                    <div className={`flex flex-row ${getStepColor(v)}`}>
                        <div className="flex w-6 h-6">
                            {getStepIcon(v)}
                        </div>
                        <span className="text-sm my-auto">
                            {getStepName(v.step)}
                        </span>
                    </div>
                    {(i < messageGroup.length - 1) && (
                        <div className="flex flex-row">
                            <div className="flex w-6 h-3">
                                <EllipsisVertical size={10} className="m-auto"/>
                            </div>
                        </div>
                    )}
                </Fragment>
            )
        })
        return [(<div className="flex flex-col ml-12">{result}</div>)]
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

        for (const group of groupMessages()) {
            if (group[0] instanceof ChatMessage) {
                result.push(...renderChatMessageGroup(group as ChatMessage[]))
            } else if (group[0] instanceof AssistantChatStep) {
                result.push(...renderAssistantChatStepGroup(group as AssistantChatStep[]))
            }
        }

        return result

        // if (item instanceof ChatMessage) {
        //     return renderMessage(item as ChatMessage);
        // } else if (item instanceof AssistantChatStep) {
        //     return renderAssistantChatStep(item as AssistantChatStep);
        // }
        // throw `Unknown chat log item: ${item}`;
    }

    const renderChatLog = () => {
        return (<div
            ref={chatRef}
            className="flex-1 overflow-y-auto px-6 py-4 space-y-4"
            id="ChatLog"
        >
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
        </div>)
    }

    const renderUserInput = () => {
        const inputDisabled = isProcessing || !userInputEnabled
        const sendButtonDisabled = inputDisabled || !userInput.trim()
        return (
            <div id="UserInput" className="px-6 py-4 bg-white border-t border-gray-200">
                <div className="flex space-x-3">
                <textarea
                    value={userInput}
                    onChange={(e) => setUserInput(e.target.value)}
                    onKeyDown={onUserInputKeyDown}
                    placeholder={inputPlaceholder}
                    disabled={inputDisabled}
                    className="flex-1 p-3 border border-gray-300 rounded-lg resize-none focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 disabled:bg-gray-100 disabled:text-gray-500"
                    rows={2}
                />
                    <button
                        onClick={sendMessage}
                        disabled={sendButtonDisabled}
                        className={`px-6 py-3 rounded-lg font-medium transition-colors ${
                            (sendButtonDisabled ? 'bg-gray-300 text-gray-500 cursor-not-allowed' : 'bg-blue-500 text-white hover:bg-blue-600')
                        }`}
                    >
                        <Send size={18}/>
                    </button>
                </div>
            </div>
        )
    }

    return (
        <div id={ChatInterface.name} className='flex-1 flex flex-col min-h-1/3'>
            {renderChatLog()}
            {renderUserInput()}
        </div>
    )

}

export default ChatInterface