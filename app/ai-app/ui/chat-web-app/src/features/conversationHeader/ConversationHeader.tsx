import {useCallback, useEffect, useMemo, useRef, useState} from "react";
import {useAppDispatch, useAppSelector} from "../../app/store.ts";
import {
    selectConversationTitle,
    selectCurrentTurn,
    selectIsNewConversation,
    selectTurnOrder,
    setUserMessage
} from "../chat/chatStateSlice.ts";
import AnimatedExpander from "../../components/AnimatedExpander.tsx";
import {
    QuestionCategory,
    QuestionsPanelItem,
    useGetSuggestedQuestionsQuery
} from "../suggestedQuestions/suggestedQuestions.ts";
import {selectProject, selectTenant} from "../chat/chatSettingsSlice.ts";

const ConversationHeader = () => {
    const conversationTitle = useAppSelector(selectConversationTitle);
    const contentContainerRef = useRef<HTMLDivElement>(null);
    const dispatch = useAppDispatch();

    const tenant = useAppSelector(selectTenant);
    const project = useAppSelector(selectProject);
    const {data, isSuccess} = useGetSuggestedQuestionsQuery({tenant, project});
    const isNewConversation = useAppSelector(selectIsNewConversation);

    const [topicPath, setTopicPath] = useState<string[]>([]);

    const pushTopic = useCallback((topic: string) => {
        setTopicPath(prevState => [...prevState, topic]);
    }, [])

    const popTopic = useCallback(() => {
        setTopicPath(prevState => {
            if (prevState.length > 0) {
                return prevState.slice(0, prevState.length - 1);
            }
            return prevState;
        });
    }, [])

    useEffect(() => {
        setTopicPath([])
    }, [data]);

    const questionItems = useMemo(() => {
        if (!data) return null;
        if (topicPath.length === 0) {
            return data;
        }
        let items: QuestionsPanelItem[] = data;
        topicPath.forEach((topic) => {
            const nextItems = (items.find(item => item.type === "category" && item.id === topic) as QuestionCategory | undefined)?.items;
            if (!nextItems) {
                console.warn("invalid topic path")
                return null
            }
            items = nextItems;
        })
        return items;

    }, [data, topicPath])

    const content = useMemo(() => {
        if (conversationTitle) {
            return <h1 className={"text-lg"}>{conversationTitle}</h1>
        } else if (isNewConversation && isSuccess && questionItems && questionItems.length > 0) {
            return <div className={"flex flex-row flex-wrap gap-1"}>
                {topicPath.length > 0 && <button
                    onClick={() => popTopic()}
                    className={"text-xs px-2 py-1 border border-gray-200 rounded-md hover:bg-gray-100 cursor-pointer"}>Back</button>}
                {questionItems.map(question => {
                    return <button
                        className={"text-xs px-2 py-1 border border-gray-200 rounded-md hover:bg-gray-100 cursor-pointer"}
                        key={`${question.type}_${question.text}`}
                        onClick={question.type === "category" ? () => pushTopic(question.id) : () => dispatch(setUserMessage(question.text))}
                    >{question.text}</button>
                })}
            </div>
        }
        return null
    }, [conversationTitle, dispatch, isNewConversation, isSuccess, popTopic, pushTopic, questionItems, topicPath.length]);

    return useMemo(() => {
        return <AnimatedExpander contentRef={contentContainerRef} expanded={!!content} direction="vertical">
            <div ref={contentContainerRef} className={"bg-white border-b border-gray-200 py-3 px-4 w-full min-w-0"}>
                {content}
            </div>
        </AnimatedExpander>
    }, [content])
};

export default ConversationHeader;