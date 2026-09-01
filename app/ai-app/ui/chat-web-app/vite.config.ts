import {defineConfig, loadEnv} from 'vite'
import type {Plugin} from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from "@tailwindcss/vite";
import basicSsl from '@vitejs/plugin-basic-ssl'

function normalizeEmittedPublicPath(path: string): string {
    return path.trim().replace(/^\.?\//, "")
}

function controlPlaneRuntimeBootstrapPlugin(): Plugin {
    return {
        name: "kdcube-control-plane-runtime-bootstrap",
        apply: "build",
        transformIndexHtml: {
            order: "post",
            handler(html) {
                const moduleScripts: string[] = [];
                const stylesheets: string[] = [];
                const modulePreloads: string[] = [];
                const icons: string[] = [];

                html = html.replace(
                    /\s*<script\b(?=[^>]*\btype=["']module["'])([^>]*?)\bsrc=["']([^"']+)["']([^>]*)><\/script>/g,
                    (_unusedMatch, _unusedBefore, src) => {
                        moduleScripts.push(normalizeEmittedPublicPath(src));
                        return "";
                    },
                );
                html = html.replace(
                    /\s*<link\b(?=[^>]*\brel=["']stylesheet["'])([^>]*?)\bhref=["']([^"']+)["']([^>]*)>/g,
                    (_unusedMatch, _unusedBefore, href) => {
                        stylesheets.push(normalizeEmittedPublicPath(href));
                        return "";
                    },
                );
                html = html.replace(
                    /\s*<link\b(?=[^>]*\brel=["']modulepreload["'])([^>]*?)\bhref=["']([^"']+)["']([^>]*)>/g,
                    (_unusedMatch, _unusedBefore, href) => {
                        modulePreloads.push(normalizeEmittedPublicPath(href));
                        return "";
                    },
                );
                html = html.replace(
                    /\s*<link\b(?=[^>]*\brel=["']icon["'])([^>]*?)\bhref=["']([^"']+)["']([^>]*)>/g,
                    (_unusedMatch, _unusedBefore, href) => {
                        icons.push(normalizeEmittedPublicPath(href));
                        return "\n    <link rel=\"icon\" type=\"image/svg+xml\" data-kdcube-control-plane-icon>";
                    },
                );

                const bootstrap = `
    <script type="module" data-kdcube-control-plane-bootstrap>
      const entrySegments = new Set(["chat", "callback", "dummy"]);
      const moduleScripts = ${JSON.stringify(moduleScripts)};
      const stylesheets = ${JSON.stringify(stylesheets)};
      const modulePreloads = ${JSON.stringify(modulePreloads)};
      const icons = ${JSON.stringify(icons.length ? icons : ["img/favicon.svg"])};

      function controlPlaneMount(pathname) {
        const parts = String(pathname || "/").split("/").filter(Boolean);
        const entryIndex = parts.findIndex((part) => entrySegments.has(part));
        if (entryIndex >= 0) {
          return entryIndex > 0 ? "/" + parts.slice(0, entryIndex).join("/") : "";
        }
        const trimmed = String(pathname || "/").replace(/\\/+$/, "");
        if (!trimmed || trimmed === "/") {
          return "";
        }
        return trimmed.startsWith("/") ? trimmed : "/" + trimmed;
      }

      const mount = controlPlaneMount(window.location.pathname);
      window.__KDCUBE_CONTROL_PLANE_MOUNT__ = mount;

      function publicUrl(path) {
        const clean = String(path || "").replace(/^\\/+/, "");
        return (mount ? mount : "") + "/" + clean;
      }

      for (const href of modulePreloads) {
        const link = document.createElement("link");
        link.rel = "modulepreload";
        link.crossOrigin = "anonymous";
        link.href = publicUrl(href);
        document.head.appendChild(link);
      }

      for (const href of stylesheets) {
        const link = document.createElement("link");
        link.rel = "stylesheet";
        link.crossOrigin = "anonymous";
        link.href = publicUrl(href);
        document.head.appendChild(link);
      }

      const iconTarget = document.querySelector("link[data-kdcube-control-plane-icon]");
      if (iconTarget && icons.length > 0) {
        iconTarget.href = publicUrl(icons[0]);
      }

      for (const src of moduleScripts) {
        import(publicUrl(src));
      }
    </script>`;

                return html.replace("</head>", `${bootstrap}\n  </head>`);
            },
        },
    }
}

export default defineConfig(({mode}) => {

    const env = loadEnv(mode, process.cwd(), '')

    const apiBase = env.VITE_APP_API_BASE ?? 'http://localhost:8010/'
    const integrationsApiBase = env.VITE_APP_INTEGRATIONS_API_BASE ?? 'http://localhost:8020/'

    return {
        base: "./",
        plugins: [
            react(),
            tailwindcss(),
            basicSsl(),
            controlPlaneRuntimeBootstrapPlugin(),
        ],
        resolve: {
            dedupe: ["react", "react-dom"],
        },
        envPrefix: ["VITE_", "CHAT_WEB_APP_"],
        server: {
            https: env.VITE_HTTPS === 'true' ? {} : undefined,

            proxy: {
                '^/api/integrations/.*': {
                    target: integrationsApiBase,
                },
                '^/admin/integrations/.*': {
                    target: integrationsApiBase,
                },
                '^/api/.*': {
                    target: apiBase,
                },
                '^/profile': {
                    target: apiBase,
                },
                '^/admin/.*': {
                    target: apiBase,
                },
                '^/monitoring/.*': {
                    target: apiBase,
                },
                '^/socket.io': {
                    target: apiBase,
                },
                '^/sse/.*': {
                    target: apiBase,
                },
            }
        }
    }
})
