import {createSlice, PayloadAction} from "@reduxjs/toolkit";
import {ChatSettings} from "./chatTypes.ts";
import {RootState} from "../../app/store.ts";

const chatSettingsSlice = createSlice({
    name: "chatSettings",
    initialState: () => {
        return {
            showMetadata: false
        } as ChatSettings
    },
    reducers: {
        setShowMetadata: (state, action: PayloadAction<boolean>) => {
            state.showMetadata = action.payload
        }
    }
})

export const {
    setShowMetadata,
} = chatSettingsSlice.actions

export const selectShowMetadata = (state: RootState) => state.chatSettings.showMetadata

export default chatSettingsSlice.reducer