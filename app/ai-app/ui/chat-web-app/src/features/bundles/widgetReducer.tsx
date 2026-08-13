import {useMemo} from "react";
import {useAppSelector} from "../../app/store.ts";
import {selectAuthConfig, selectProject, selectTenant} from "../chat/chatSettingsSlice.ts";
import {useGetBundlesListQuery} from "./bundlesAPI.ts";
import {selectCurrentBundle} from "./bundlesSlice.ts";
import {selectUserProfile} from "../profile/profile.ts";

const BUILT_IN_WIDGET_ALIASES = new Set([
    "economic_usage",
    "conversation_browser",
    "control_plane",
    "ai_bundles",
    "opex",
    "redis_browser",
    "svc_gateway",
]);

export const useGetBundleWidgets = () => {
    const tenant = useAppSelector(selectTenant)
    const project = useAppSelector(selectProject)

    const {data} = useGetBundlesListQuery({tenant, project})

    const userProfile = useAppSelector(selectUserProfile)
    const authConfig = useAppSelector(selectAuthConfig)

    const authDisabled = useMemo(() => authConfig.authType === "none", [authConfig.authType])

    const currentBundleId = useAppSelector(selectCurrentBundle)
    const currentBundleData = useMemo(() => {
        if (!data || !currentBundleId) return null;
        return data.bundles[currentBundleId] || null;
    }, [currentBundleId, data]);


    return useMemo(() => {
        const allWidgets = currentBundleData?.widgets || [];
        const userRoles = userProfile?.roles ?? []
        const widgets = allWidgets.filter((widget) => {
            // Role lists are ANY-OF, matching the proc-side visibility filter.
            return !BUILT_IN_WIDGET_ALIASES.has(widget.alias) && (!widget.roles || widget.roles.length === 0 || authDisabled || (!!userProfile && widget.roles.some(r=>userRoles.includes(r))))
        });

        // An app serves the default chat surface only when its descriptor
        // declares it (surfaces.as_provider.bundle.default_chat). While the
        // registry is still loading we assume chat to avoid a scene flash.
        const defaultChat = currentBundleData ? Boolean(currentBundleData.default_chat) : true;

        return {
            currentBundleId,
            currentBundleData,
            widgets,
            allWidgets,
            defaultChat
        }
    }, [authDisabled, currentBundleData, currentBundleId, userProfile])
}