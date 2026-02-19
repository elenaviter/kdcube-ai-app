import {useCallback, useMemo} from "react";
import {Bot, Loader, LogOut, Wifi, WifiOff} from "lucide-react";
import {useAppDispatch, useAppSelector} from "../../app/store.ts";
import {selectChatConnected, selectChatStayConnected} from "../../features/chat/chatStateSlice.ts";
import {logOut} from "../../features/auth/authMiddleware.ts";

const ChatHeader = () => {
    const dispatch = useAppDispatch()
    const stayConnected = useAppSelector(selectChatStayConnected)
    const connected = useAppSelector(selectChatConnected)

    const connectionStatus = useMemo(() => {
        if (stayConnected && !connected) return {
            icon: <Loader size={14} className="animate-spin"/>,
            text: 'Connecting...',
            color: 'text-yellow-600 bg-yellow-50'
        };
        if (connected) return {icon: <Wifi size={14}/>, text: 'Connected', color: 'text-green-600 bg-green-50'};
        return {icon: <WifiOff size={14}/>, text: 'Disconnected', color: 'text-red-600 bg-red-50'};
    }, [stayConnected, connected]);

    const handleLogout = useCallback(() => {
        dispatch(logOut())
    }, [dispatch]);

    return  useMemo(() => {
        return (
            <div className="bg-white border-b border-gray-200 px-6 py-4">
                <div className="flex items-center justify-between">
                    <div className="flex items-center">
                        <div
                            className="w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg mr-3 flex items-center justify-center">

                            <Bot size={20} className="text-white"/>
                        </div>
                        <div>
                            <h1 className="text-xl font-semibold text-gray-900">
                                AI Assistant
                            </h1>
                        </div>
                    </div>

                    <div className="flex items-center gap-2">
                        <div className={`flex items-center px-3 py-1 rounded-lg text-sm ${connectionStatus.color}`}>
                            {connectionStatus.icon}
                            <span className="ml-2 font-medium">{connectionStatus.text}</span>
                        </div>

                        {/*<button*/}
                        {/*    onClick={() => setShowKB(!showKB)}*/}
                        {/*    className="relative flex items-center px-3 py-2 rounded-lg bg-gray-100 text-gray-600 hover:bg-gray-200"*/}
                        {/*    title="View KB"*/}
                        {/*>*/}
                        {/*    <Database size={16} className="mr-1"/><span className="text-sm">KB</span>*/}
                        {/*</button>*/}

                        {/*<button*/}
                        {/*    onClick={handleShowKbResults}*/}
                        {/*    className={`relative flex items-center px-3 py-2 rounded-lg transition-colors ${*/}
                        {/*        kbSearchHistory.length > 0 ? 'bg-blue-100 text-blue-700 hover:bg-blue-200' : 'bg-gray-100 text-gray-600 hover:bg-gray-200'*/}
                        {/*    }`}*/}
                        {/*    title="View KB Search Results"*/}
                        {/*>*/}
                        {/*    <Search size={16} className="mr-1"/>*/}
                        {/*    <span className="text-sm">KB Search</span>*/}
                        {/*    {kbSearchHistory.length > 0 && (*/}
                        {/*        <span*/}
                        {/*            className="ml-1 text-xs bg-blue-200 text-blue-800 px-1 rounded">{kbSearchHistory.length}</span>*/}
                        {/*    )}*/}
                        {/*    {newKbSearchCount > 0 && (*/}
                        {/*        <span*/}
                        {/*            className="absolute -top-1 -right-1 w-2 h-2 bg-red-500 rounded-full animate-pulse"/>*/}
                        {/*    )}*/}
                        {/*</button>*/}

                        <button
                            onClick={handleLogout}
                            className="flex items-center px-3 py-2 bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-lg"
                            title="Sign out"
                        >
                            <LogOut size={16} className="mr-1"/><span className="text-sm">Logout</span>
                        </button>
                    </div>
                </div>
            </div>
        )
    }, [connectionStatus.color, connectionStatus.icon, connectionStatus.text, handleLogout])
}

export default ChatHeader;