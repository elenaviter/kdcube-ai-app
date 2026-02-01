import {useEffect} from "react";
import {useAppDispatch, useAppSelector} from "../../app/store.ts";
import {connectChat} from "../../features/chat/chatServiceMiddleware.ts";
import {selectUserProfile, selectUserProfileError, selectUserProfileStatus} from "../../features/profile/profile.ts";
import {selectChatConnected, selectChatStayConnected} from "../../features/chat/chatStateSlice.ts";
import {logOut} from "../../features/auth/authMiddleware.ts";
import {selectAppUser} from "../../features/auth/authSlice.ts";

const ProfileInfo = () => {
    const userProfileStatus = useAppSelector(selectUserProfileStatus)
    const userProfileError = useAppSelector(selectUserProfileError)
    const userProfile = useAppSelector(selectUserProfile)

    return <div>
        <p>userProfileStatus: {userProfileStatus}</p>
        <p>userProfileError: {String(userProfileError)}</p>
        <p>userProfile: </p>
        <pre>{userProfile && JSON.stringify(userProfile, null, 4)}</pre>
    </div>
}

const AppUserInfo = () => {
    const appUser = useAppSelector(selectAppUser)
    console.log(appUser)
    return <div>
        <p>appUser.name: {String(appUser?.name)}</p>
    </div>
}

const ChatState = () => {
    const chatConnected = useAppSelector(selectChatConnected)
    const chatStayConnected = useAppSelector(selectChatStayConnected)

    return <div>
        <p>chatStayConnected: {String(chatStayConnected)}</p>
        <p>chatConnected: {String(chatConnected)}</p>
    </div>
}

const Dummy = () => {
    const dispatch = useAppDispatch()

    useEffect(() => {
        dispatch(connectChat())
    }, []);

    return <div className={"m-1 p-1"}>
        <button className={"p-1 border cursor-pointer hover:bg-gray-100"} onClick={() => {
            dispatch(logOut())
        }}>Logout
        </button>
        <ProfileInfo/>
        <AppUserInfo/>
        <ChatState/>
    </div>
}

export default Dummy