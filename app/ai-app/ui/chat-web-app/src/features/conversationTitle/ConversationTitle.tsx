import {useMemo, useRef} from "react";
import {useAppSelector} from "../../app/store.ts";
import {selectConversationTitle} from "../chat/chatStateSlice.ts";
import AnimatedExpander from "../../components/AnimatedExpander.tsx";

const ConversationTitle = () => {
    const conversationTitle = useAppSelector(selectConversationTitle)
    const content = useRef<HTMLDivElement>(null);

    return useMemo(() => {
        return <AnimatedExpander contentRef={content} expanded={!!conversationTitle} direction="vertical">
            <div ref={content} className={"bg-white border-b border-gray-200 py-3 px-4 w-full min-w-0"}>
                <h1 className={"text-lg"}>{conversationTitle}</h1>
            </div>
        </AnimatedExpander>
    }, [conversationTitle])
}

export default ConversationTitle;