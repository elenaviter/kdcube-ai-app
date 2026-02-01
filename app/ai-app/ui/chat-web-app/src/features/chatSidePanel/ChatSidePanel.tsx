import {ReactNode, useCallback, useEffect, useMemo, useRef, useState} from "react";
import {CirclePlus, MessagesSquare, Search} from "lucide-react";
import IconContainer from "../../components/IconContainer.tsx";
import AnimatedExpander from "../../components/AnimatedExpander.tsx";
import {motion} from "motion/react";
import {useAppDispatch, useAppSelector} from "../../app/store.ts";
import {
    selectConversationDescriptors,
    selectConversationDescriptorsLoading,
    selectConversationDescriptorsLoadingError
} from "../conversations/conversationsSlice.ts";
import {formatDateToLocalString} from "../../utils/dateTimeUtils.ts";
import {ConversationDescriptor} from "../conversations/conversationsTypes.ts";
import {timeSortPredicate} from "../../utils/utils.ts";
import {getChatPagePath} from "../../AppConfig.ts";
import {useNavigate} from "react-router-dom";
import {loadConversationList} from "../conversations/conversationsMiddleware.ts";
import {newConversation} from "../chat/chatStateSlice.ts";

interface MenuButtonProps {
    children: ReactNode | ReactNode[];
    onClick?: () => void;
}

const MenuButton = ({children, onClick}: MenuButtonProps) => {
    return useMemo(() => {
        return <button
            onClick={e => {
                e.preventDefault();
                e.stopPropagation();
                onClick?.()
            }}
            className={"hover:bg-gray-200 transition-all duration-200 p-1 rounded-md cursor-pointer"}>
            {children}
        </button>
    }, [children, onClick])
}

interface ConversationMenuItemProps {
    conversation: ConversationDescriptor;
}

const ConversationMenuItem = ({conversation}: ConversationMenuItemProps) => {
    const navigate = useNavigate();

    return useMemo(() => {
        const href = getChatPagePath() + "/" + conversation.id;
        return <div className={"mx-2.5 mt-2 px-2 py-1 text-md hover:bg-gray-100 rounded-md cursor-pointer"}>
            <a
                href={href}
                onClick={e => {
                    e.preventDefault();
                    e.stopPropagation();
                    navigate(href)
                }}
            >
                <p>{conversation.title ?? "Unnamed conversation"}</p>
                {conversation.lastActivity && (<p className={"text-sm"}>
                    {formatDateToLocalString(new Date(conversation.lastActivity))}
                </p>)}
            </a>
        </div>
    }, [conversation, navigate]);
}

const ConversationsPanel = () => {
    const dispatch = useAppDispatch();
    const conversations = useAppSelector(selectConversationDescriptors);

    const processedConversations = useMemo(() => {
        return conversations ? conversations.concat().sort((a, b) => timeSortPredicate(a.lastActivity, b.lastActivity)).reverse() : null;

    }, [conversations]);

    const conversationsLoading = useAppSelector(selectConversationDescriptorsLoading);
    const conversationsLoadingError = useAppSelector(selectConversationDescriptorsLoadingError);

    const [searchFor, setSearchFor] = useState("");

    useEffect(() => {
        dispatch(loadConversationList())
    }, [dispatch]);

    return useMemo(() => {
        return <div className={"w-full h-full flex flex-col"}>
            <h1 className={"text-xl mx-auto mt-2 ml-2.5"}>Conversations</h1>
            {conversationsLoading && !conversations && (
                <div>loading</div>
            )}
            {conversationsLoadingError && (
                <div>error</div>
            )}
            {processedConversations && processedConversations.length > 0 && (
                <>
                    <div
                        className="p-1.5 mx-2 border bg-white border-gray-200 transition-all rounded-md flex flex-row items-center focus-within:border-gray-800">
                        <IconContainer icon={Search} size={1}/>
                        <input name={"convSearch"} type={"text"} className={"flex-1 ml-2 focus:outline-none"}
                               placeholder={"Search"} value={searchFor} onChange={e => setSearchFor(e.target.value)}/>
                    </div>
                    {processedConversations.map((conversation) => (
                        <ConversationMenuItem conversation={conversation} key={conversation.id}/>
                    ))}
                </>
            )}
        </div>
    }, [conversationsLoading, conversations, conversationsLoadingError, processedConversations, searchFor])
}

type Panels = "conversations" | null

const ChatSidePanel = () => {
    const dispatch = useAppDispatch();

    const parentRef = useRef<HTMLDivElement>(null);
    const [visiblePanel, setVisiblePanel] = useState<Panels>(null);
    const sidePanelContentRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        const onOutsideClick = (ev: MouseEvent) => {
            if (!parentRef.current?.contains(ev.target as Element))
                setVisiblePanel(null)
        }
        window.addEventListener("click", onOutsideClick);
        return () => {
            window.removeEventListener("click", onOutsideClick)
        }
    }, [visiblePanel]);

    const onPanelButtonClick = useCallback((panel: Panels) => {
        if (panel === null || panel !== visiblePanel) {
            setVisiblePanel(panel)
        } else {
            setVisiblePanel(null)
        }
    }, [visiblePanel])

    return useMemo(() => {
        return <div
            ref={parentRef}
            className={"h-full w-12 relative overflow-visible"}>
            <div
                className={"h-full flex flex-col items-center bg-gray-50 border-r border-gray-200 pt-1 px-1 text-gray-700 gap-1"}>
                <MenuButton onClick={() => dispatch(newConversation())}>
                    <IconContainer icon={CirclePlus} size={1.5}/>
                </MenuButton>
                <MenuButton
                    onClick={() => {
                        onPanelButtonClick("conversations");
                    }}
                >
                    <IconContainer icon={MessagesSquare} size={1.5}/>
                </MenuButton>
            </div>
            <div className={"absolute h-full top-0 left-12 z-20 shadow-md border-r border-gray-200 bg-white"}>
                <AnimatedExpander contentRef={sidePanelContentRef} className={"h-full"}
                                  expanded={visiblePanel !== null}>
                    <motion.div
                        className={"h-full w-128"} ref={sidePanelContentRef}
                        initial={{
                            opacity: visiblePanel === "conversations" ? 0 : 1,
                        }}
                        animate={{
                            opacity: visiblePanel === "conversations" ? 1 : 0,
                        }}
                    >
                        <ConversationsPanel/>
                    </motion.div>
                </AnimatedExpander>
            </div>
        </div>
    }, [onPanelButtonClick, visiblePanel])
}

export default ChatSidePanel;