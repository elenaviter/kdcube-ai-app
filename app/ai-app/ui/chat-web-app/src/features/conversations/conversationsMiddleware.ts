import {Middleware, UnknownAction} from "@reduxjs/toolkit";
import {AppStore} from "../../app/store.ts";
import {chatConnected, loadConversation, newConversation, setConversationId} from "../chat/chatStateSlice.ts";
import {fetchConversation, getConversations} from "./conversationsAPI.ts";
import {
    conversationStatusUpdateRequired,
    setConversationDescriptors,
    setConversationDescriptorsLoading,
    setConversationDescriptorsLoadingError,
    setConversationLoading
} from "./conversationsSlice.ts";
import {
    ArtifactStreamData, ArtifactStreamReducer,
    AssistantFileData,
    CitationsData,
    ConversationDescriptor,
    FollowUpsData,
    ThinkingStreamData
} from "./conversationsTypes.ts";
import {
    AgentTiming,
    ChatTurn,
    TurnCitation,
    TurnFile,
    TurnThinkingItem,
    UnknownArtifact,
    UserMessage
} from "../chat/chatTypes.ts";
import {RichLink, RNFile} from "../chatController/chatBase.ts";
import {requestConversationStatus} from "../chat/chatServiceMiddleware.ts";
import {CodeExecArtifactStreamReducer} from "../logExtensions/codeExec/CodeExecArtifactStreamReducer.ts";
import {WebSearchArtifactStreamReducer} from "../logExtensions/webSearch/WebSearchArtifactStreamReducer.ts";
import {CanvasArtifactStreamReducer} from "../logExtensions/canvas/CanvasArtifactStreamReducer.ts";
import {IgnoredArtifactStreamReducer} from "../logExtensions/ignored/IgnoredArtifactStreamReducer.ts";

const LOAD_CONVERSATION_LIST = "conversations/loadConversationList"

interface LoadConversationListAction extends UnknownAction {
    type: typeof LOAD_CONVERSATION_LIST;
}

export const loadConversationList = (): LoadConversationListAction => {
    return {
        type: LOAD_CONVERSATION_LIST,
    }
}

const LOAD_CONVERSATION = "conversations/loadConversation"

interface LoadConversationsAction extends UnknownAction {
    type: typeof LOAD_CONVERSATION;
    payload: string | null;
}

export const loadConversations = (conversationId: string | null): LoadConversationsAction => {
    return {
        type: LOAD_CONVERSATION,
        payload: conversationId
    }
}

type ConversationURLAction = LoadConversationsAction
    | LoadConversationListAction
    | ReturnType<typeof setConversationId>
    | ReturnType<typeof chatConnected>;

const conversationsMiddleware = (): Middleware => {
    const artifactStreamReducers: ArtifactStreamReducer[] = [
        new IgnoredArtifactStreamReducer(),
        new CanvasArtifactStreamReducer(),
        new CodeExecArtifactStreamReducer(),
        new WebSearchArtifactStreamReducer()
    ];

    return (store) => (next) => (action) => {
        const actionHandlers = async (store: AppStore, action: ConversationURLAction) => {
            const dispatch = store.dispatch
            switch (action.type) {
                case LOAD_CONVERSATION_LIST: {
                    const state = store.getState()
                    if (state.conversations.conversationsDescriptorsLoading)
                        break
                    dispatch(setConversationDescriptorsLoading())
                    getConversations(state.chatState.tenant, state.chatState.project).then((conversations) => {
                        const list = conversations.map((it): ConversationDescriptor => {
                            return {
                                id: it.conversation_id,
                                started: it.started_at ? Date.parse(it.started_at) : null,
                                lastActivity: it.last_activity_at ? Date.parse(it.last_activity_at) : null,
                                title: it.title
                            }
                        })
                        dispatch(setConversationDescriptors(list))
                    }).catch(error => {
                        console.error(error)
                        dispatch(setConversationDescriptorsLoadingError(error.message))
                    })
                    break
                }
                case LOAD_CONVERSATION: {
                    const state = store.getState()
                    const conversationId = (action as LoadConversationsAction).payload;
                    if (conversationId === state.chatState.conversationId || (conversationId !== state.chatState.conversationId && state.conversations.conversationLoading))
                        break;
                    if (conversationId === null) {
                        dispatch(newConversation())
                    } else {
                        dispatch(setConversationLoading(conversationId))
                        fetchConversation(state.chatState.tenant, state.chatState.project, conversationId).then((conversation) => {
                            dispatch(loadConversation({
                                turnOrder: conversation.turns.map(it => it.turn_id),
                                turns: conversation.turns.reduce((previousValue, currentValue, i, arr) => {
                                    let userMessage: UserMessage | null = null;
                                    let answer: string | null = null;
                                    const followUpQuestions: string[] = []
                                    const turnArtifacts: UnknownArtifact[] = []

                                    currentValue.artifacts.forEach(it => {
                                        switch (it.type) {
                                            case "chat:user":
                                                userMessage = {
                                                    text: it.data.text,
                                                    timestamp: Date.parse(it.ts),
                                                    attachments: userMessage ? userMessage.attachments : []
                                                }
                                                break
                                            case "artifact:user.attachment":
                                                userMessage = {
                                                    text: userMessage ? userMessage.text : "",
                                                    timestamp: userMessage ? userMessage.timestamp : Date.parse(it.ts),
                                                    attachments: []
                                                }
                                                break
                                            case "chat:assistant":
                                                answer = it.data.text
                                                break
                                            case "artifact:assistant.file": {
                                                const dto = it.data as AssistantFileData;
                                                const tf: TurnFile = {
                                                    artifactType: "file",
                                                    timestamp: Date.now(), //todo: use actual date
                                                    content: {
                                                        filename: dto.payload.filename,
                                                        rn: dto.payload.rn,
                                                        mime: dto.payload.mime,
                                                        description: dto.payload.description,
                                                    } as RNFile
                                                }
                                                turnArtifacts.push(tf)
                                                break
                                            }
                                            case "artifact:conv.thinking.stream": {
                                                const dto = it.data as ThinkingStreamData;

                                                const startTime = Math.min(...dto.payload.items.map(item => item.ts_first))
                                                const finishTime = Math.min(...dto.payload.items.map(item => item.ts_last))

                                                const agents: Record<string, string> = {}
                                                const agentTimes: Record<string, AgentTiming> = {}

                                                dto.payload.items.forEach(it => {
                                                    agents[it.agent] = it.text
                                                    agentTimes[it.agent] = {
                                                        startedAt: it.ts_first,
                                                        endedAt: it.ts_last,
                                                        active: false,
                                                    }
                                                })
                                                const r: TurnThinkingItem = {
                                                    artifactType: "thinking",
                                                    timestamp: Date.now(), //todo: use actual date
                                                    content: {
                                                        agentTimes,
                                                        agents,
                                                        timestamp: startTime,
                                                        endedAt: finishTime,
                                                    }
                                                }
                                                turnArtifacts.push(r)
                                                break
                                            }
                                            case "artifact:solver.program.citables": {
                                                const dto = it.data as CitationsData;
                                                dto.payload.items.forEach(v => {
                                                        const t: RichLink = {
                                                            url: v.url,
                                                            title: v.title,
                                                            favicon: v.favicon,
                                                            body: v.text
                                                        }
                                                        const r: TurnCitation = {
                                                            artifactType: "citation",
                                                            content: t,
                                                            timestamp: Date.now(), //todo: use actual date
                                                        }
                                                        turnArtifacts.push(r)
                                                    }
                                                )
                                                break
                                            }
                                            case "artifact:conv.artifacts.stream": {
                                                const dto = it.data as ArtifactStreamData;
                                                dto.payload.items.forEach(a => {
                                                    let processed = false
                                                    artifactStreamReducers.forEach(r => {
                                                        processed = processed || r.process(a)
                                                    })
                                                    if (!processed) {
                                                        console.warn("unknown artifact stream", a)
                                                    }
                                                })
                                                break
                                            }
                                            case "artifact:turn.log.reaction":
                                                break
                                            case "artifact:conv.user_shortcuts": {
                                                const dto = it.data as FollowUpsData;
                                                followUpQuestions.push(...dto.payload.items);
                                                break
                                            }
                                            // case "artifact:conv.timeline_text.stream":
                                            //     break
                                            default:
                                                console.warn("unknown artifact type", it);
                                        }
                                    })

                                    artifactStreamReducers.forEach(r => {
                                        turnArtifacts.push(...r.flush())
                                    })

                                    previousValue[currentValue.turn_id] = {
                                        id: currentValue.turn_id,
                                        state: i < arr.length - 2 ? "finished" : "inProgress",
                                        userMessage: userMessage ?? {text: "ERROR", attachments: [], timestamp: 0},
                                        events: [],
                                        artifacts: turnArtifacts,
                                        steps: {},
                                        followUpQuestions,
                                        answer,
                                        historical: true
                                    }
                                    return previousValue
                                }, {} as Record<string, ChatTurn>),
                                conversationId: conversation.conversation_id
                                //todo: conversationTitle:
                            }))

                            if (state.chatState.connected) {
                                dispatch(requestConversationStatus(conversation.conversation_id))
                            } else {
                                dispatch(conversationStatusUpdateRequired())
                            }

                        }).catch(error => {
                            console.error(error)
                            dispatch(newConversation())
                        })
                    }
                    break;
                }
                case chatConnected.type: {
                    const state = store.getState()
                    if (state.conversations.conversationLoading && state.conversations.conversationStatusUpdateRequired) {
                        dispatch(requestConversationStatus(state.conversations.conversationLoading))
                    }
                    break
                }
            }
        }

        next(action);
        actionHandlers(store as AppStore, action as ConversationURLAction).catch(console.error);
    }
}

export default conversationsMiddleware;