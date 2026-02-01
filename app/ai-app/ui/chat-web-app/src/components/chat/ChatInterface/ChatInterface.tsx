/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2025 Elena Viter
 */

// ChatInterface.tsx
import {CirclePlus, Loader, Send, X,} from "lucide-react";
import {CSSProperties, Fragment, useCallback, useEffect, useMemo, useRef} from "react";
import {selectFileAdvanced} from "../../shared.ts";
import {useAppDispatch, useAppSelector} from "../../../app/store.ts";
import {
    addUserAttachments,
    removeUserAttachment,
    selectChatStayConnected,
    selectCurrentTurn,
    selectLocked,
    selectTurnOrder,
    selectTurns,
    selectUserAttachments,
    selectUserMessage,
    setUserMessage
} from "../../../features/chat/chatStateSlice.ts";
import {connectChat, sendChatMessage} from "../../../features/chat/chatServiceMiddleware.ts";
import {UserAttachment} from "../../../features/chat/chatTypes.ts";
import {getTurnCitationItems, getTurnFileItems, getTurnThinkingItems} from "../../../features/chat/utils.ts";
import {AssistantMessageComponent} from "./AssistantMessageComponent.tsx";
import {UserMessageComponent} from "./UserMessageComponent.tsx";

interface ChatInterfaceProps {
    lockMessage?: string;
    inputPlaceholder?: string;
    showMetadata?: boolean;
    maxWidth?: number | string;
}

const ChatInterface = ({
                           inputPlaceholder = "Ask me anything...",
                           lockMessage,
                           maxWidth,
                       }: ChatInterfaceProps) => {
    const dispatch = useAppDispatch();

    const turns = useAppSelector(selectTurns)
    const turnOrder = useAppSelector(selectTurnOrder)
    const currentTurn = useAppSelector(selectCurrentTurn)
    const isLocked = useAppSelector(selectLocked)
    const stayConnected = useAppSelector(selectChatStayConnected)
    const userInput = useAppSelector(selectUserMessage)
    const userAttachments = useAppSelector(selectUserAttachments)

    const setUserInputValue = useCallback((userInput: string) => {
        dispatch(setUserMessage(userInput));
    }, [dispatch])

    useEffect(() => {
        if (!stayConnected)
            dispatch(connectChat())
    }, [dispatch, stayConnected]);

    const inProgress = useMemo(() => !!currentTurn, [currentTurn])
    const followUpQuestion = useMemo(() => {
        return turnOrder.length > 0 ? turns[turnOrder[turnOrder.length - 1]].followUpQuestions : null
    }, [turnOrder, turns])


    // const [userInput, setUserInput] = useState<string>("");
    // const [userInputFiles, setUserInputFiles] = useState<File[]>([]);

    const logContainerRef = useRef<HTMLDivElement | null>(null);
    const userInputFieldRef = useRef<HTMLTextAreaElement | null>(null);
    const contentRef = useRef<HTMLDivElement | null>(null);

    // auto-scroll when near bottom
    const autoScroll = useRef(true);


    const addInputFiles = useCallback((files: File[]) => {
        dispatch(addUserAttachments(files))
    }, [dispatch])

    const removeInputFiles = useCallback((attachment: UserAttachment) => {
        dispatch(removeUserAttachment(attachment.fileKey))
    }, [dispatch])

    useEffect(() => {
        if (autoScroll.current && logContainerRef.current) {
            logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
        }
    }, [turns, inProgress]);

    const onScroll = () => {
        const el = logContainerRef.current;
        if (!el) return;
        const threshold = 60;
        const atBottom = el.scrollHeight - (el.scrollTop + el.clientHeight) < threshold;
        autoScroll.current = atBottom;
    };

    const sendMessage = useCallback((message?: string) => {
        dispatch(sendChatMessage({
            message
        }))
    }, [dispatch]);

    const followUpQuestionsRender = useMemo(() => {
        const disabled = inProgress;
        if (!inProgress && followUpQuestion && followUpQuestion.length > 0) {
            return (
                <div className="flex flex-row items-start w-full flex-wrap space-x-1 space-y-1 pl-3">
                    {followUpQuestion.map((q, i) => {
                        return (<button key={`follow-up-question-${i}`}
                                        className="px-3 py-1 text-xs bg-white text-gray-700 border border-gray-200 rounded-full hover:bg-gray-50 hover:border-gray-300 disabled:opacity-50"
                                        onClick={() => {
                                            sendMessage(q)
                                        }} disabled={disabled}>
                            {q}
                        </button>)
                    })}
                </div>
            )
        }
        return null
    }, [inProgress, followUpQuestion, sendMessage]);

    const processingRender = useMemo(() => {
        return inProgress ? (
            <div className="flex items-center text-gray-500 mt-2 ml-4">
                <Loader size={16} className="animate-spin mr-2"/>
                <span>Working…</span>
            </div>
        ) : null
    }, [inProgress])

    const turnsRender = useMemo(() => {
        return (<>
            {turnOrder.map(turnId => {
                const turn = turns[turnId];

                return (<Fragment key={turnId}>
                    <UserMessageComponent message={turn.userMessage} turnId={turnId}/>
                    <AssistantMessageComponent
                        message={turn.answer}
                        thinkingItems={getTurnThinkingItems(turn.artifacts)}
                        isGreeting={false}
                        citations={getTurnCitationItems(turn.artifacts)}
                        steps={Object.values(turn.steps)}
                        files={getTurnFileItems(turn.artifacts)}
                        isError={turn.state === 'error'}
                    />
                </Fragment>)
            })}
        </>)
    }, [turnOrder, turns])

    const chatLogMemo = useMemo(() => {
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
                    <div className="border-r border-l border-gray-200 mx-auto min-h-full bg-slate-50"
                         style={elementStyle}>
                        <div className="px-10 py-4">
                            {turnsRender}
                            {processingRender}
                            {followUpQuestionsRender}
                        </div>
                        <div className="pb-22"/>
                    </div>
                </div>

            </div>
        )
    }, [followUpQuestionsRender, maxWidth, processingRender, turnsRender])

    const userInputMemo = useMemo(() => {
        const inputDisabled = inProgress;
        // const inputDisabled = false;
        const sendButtonDisabled = inputDisabled || (!userInput.trim() && userAttachments.length == 0);
        const elementStyle: CSSProperties = {}
        if (maxWidth)
            elementStyle.width = typeof maxWidth === 'number' ? `${maxWidth}px` : maxWidth;

        return (
            <div
                id="UserInput"
                className="absolute -left-2 z-10 bottom-0 w-full"
                onClick={() => userInputFieldRef.current?.focus()}
            >
                <div className="pointer-events-none mx-auto px-8" style={elementStyle}>
                    <div
                        className={`flex flex-col mx-auto border rounded-t-xl border-gray-400 shadow-sm pointer-events-auto ${isLocked ? "bg-yellow-50" : "bg-white"}`}
                    >
                        {userAttachments && userAttachments.length > 0 &&
                            <div className="flex flex-row flex-wrap p-3 gap-1">
                                {
                                    userAttachments.map((file, i) => {
                                        return (<div
                                            className="flex border-2 border-gray-400 bg-gray-50 rounded-2xl px-3 py-1 items-center"
                                            key={`input-file-${i}`}
                                        >
                                            <span>{file.name}</span>
                                            <button
                                                className="pl-1 text-gray-400 hover:text-gray-600 cursor-pointer"
                                                onClick={() => {
                                                    removeInputFiles(file)
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
                                    onChange={(e) => setUserInputValue(e.target.value)}
                                    onKeyDown={(e) => {
                                        if (inputDisabled && e.key === "Enter" && !e.shiftKey) {
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
                                    (inProgress ? "cursor-wait" : "cursor-auto") :
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
                                    (userInput.trim() || userAttachments.length > 0 ? "cursor-wait" : "cursor-auto") :
                                    "cursor-pointer"}`}/>
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        );
    }, [addInputFiles, inProgress, inputPlaceholder, isLocked, lockMessage, maxWidth, removeInputFiles, sendMessage, setUserInputValue, userAttachments, userInput]);

    return useMemo(() => {
        return <div id={ChatInterface.name}
                    ref={contentRef}
                    className="flex-1 flex flex-col bg-slate-100 min-h-0 min-w-0 transition-all duration-100 ease-out w-full relative"
        >
            {chatLogMemo}
            {userInputMemo}
        </div>
    }, [chatLogMemo, userInputMemo])
};

export default ChatInterface;
