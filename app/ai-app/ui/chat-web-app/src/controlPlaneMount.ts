const ENTRY_ROUTE_SEGMENTS = new Set(["chat", "callback", "dummy"]);

declare global {
    interface Window {
        __KDCUBE_CONTROL_PLANE_MOUNT__?: string;
    }
}

function normalizeMount(value: string): string {
    const trimmed = value.trim().replace(/\/+$/, "");
    if (!trimmed || trimmed === "/") {
        return "";
    }
    return trimmed.startsWith("/") ? trimmed : `/${trimmed}`;
}

export function detectControlPlaneMount(pathname: string = window.location.pathname): string {
    const parts = String(pathname || "/").split("/").filter(Boolean);
    const entryIndex = parts.findIndex((part) => ENTRY_ROUTE_SEGMENTS.has(part));
    if (entryIndex >= 0) {
        return entryIndex > 0 ? `/${parts.slice(0, entryIndex).join("/")}` : "";
    }
    return normalizeMount(String(pathname || "/"));
}

export function getControlPlaneMount(): string {
    return normalizeMount(window.__KDCUBE_CONTROL_PLANE_MOUNT__ || detectControlPlaneMount());
}

export function controlPlanePublicPath(path: string): string {
    const clean = String(path || "").replace(/^\/+/, "");
    const mount = getControlPlaneMount();
    return `${mount || ""}/${clean}`;
}
