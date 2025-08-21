#!/usr/bin/env node
/**
 * Add license headers to source files in a React/TS project.
 * Usage: node scripts/add-license-headers.js [projectRoot]
 * Example: node scripts/add-license-headers.js .
 */

const fs = require("fs");
const fsp = fs.promises;
const path = require("path");

const PROJECT_ROOT = path.resolve(process.argv[2] || ".");
const HEADER_TEXT = `SPDX-License-Identifier: MIT
Copyright (c) 2025 Elena Viter`;

const ALLOWED_EXTS = new Set([
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".css",
    ".scss",
    ".sass",
    ".html",
    ".mjs",
    ".cjs",
    ".d.ts",
]);

const SKIP_DIRS = new Set([
    "node_modules",
    ".git",
    "build",
    "dist",
    ".next",
    "out",
    "coverage",
    ".turbo",
    ".cache",
    ".expo",
    "android",
    "ios",
]);

// Files that are almost always generated / not for headers
const SKIP_FILES_BY_NAME = new Set([
    "package-lock.json",
    "pnpm-lock.yaml",
    "yarn.lock",
]);

function makeBlockComment(body, style) {
    // style: "js" | "css" | "html"
    if (style === "html") {
        return `<!--
${body}
-->

`;
    }
    // js/css share /* ... */
    const lines = body.split("\n").map((l) => (l.length ? ` * ${l}` : " *"));
    return `/*\n${lines.join("\n")}\n */\n\n`;
}

function headerForExt(ext) {
    if (ext === ".html") return makeBlockComment(HEADER_TEXT, "html");
    // Default to JS/CSS style for code-like files
    return makeBlockComment(HEADER_TEXT, "js");
}

function hasHeaderAlready(content) {
    // Check near the top for both key strings to avoid duplicates.
    const head = content.slice(0, 800).toLowerCase();
    return head.includes("mit license") && head.includes("elena viter");
}

function isBinaryLikely(buffer) {
    // Very light heuristic: look for zero bytes
    for (let i = 0; i < Math.min(buffer.length, 512); i++) {
        if (buffer[i] === 0) return true;
    }
    return false;
}

async function shouldProcessFile(fullPath, stats) {
    if (!stats.isFile()) return false;
    const base = path.basename(fullPath);
    if (SKIP_FILES_BY_NAME.has(base)) return false;

    const ext = path.extname(fullPath).toLowerCase();
    if (!ALLOWED_EXTS.has(ext)) return false;

    // Skip minified bundles
    if (base.endsWith(".min.js") || base.endsWith(".min.css")) return false;

    // Quick binary check to be safe
    const fd = await fsp.open(fullPath, "r");
    const { buffer } = await fd.read(Buffer.alloc(512), 0, 512, 0);
    await fd.close();
    if (isBinaryLikely(buffer)) return false;

    return true;
}

async function addHeaderToFile(fullPath) {
    const ext = path.extname(fullPath).toLowerCase();
    const raw = await fsp.readFile(fullPath, "utf8");

    if (hasHeaderAlready(raw)) return { changed: false, reason: "already-present" };

    let content = raw;

    // Preserve BOM if present
    let bom = "";
    if (content.charCodeAt(0) === 0xfeff) {
        bom = "\uFEFF";
        content = content.slice(1);
    }

    // Preserve shebang if present
    let shebang = "";
    if (content.startsWith("#!")) {
        const nl = content.indexOf("\n");
        if (nl !== -1) {
            shebang = content.slice(0, nl + 1);
            content = content.slice(nl + 1);
        } else {
            // Entire file is a shebang line—rare, but just return unchanged
            return { changed: false, reason: "shebang-only" };
        }
    }

    const header = headerForExt(ext);
    const next = bom + shebang + header + content;
    await fsp.writeFile(fullPath, next, "utf8");
    return { changed: true };
}

async function walk(dir, out = []) {
    const entries = await fsp.readdir(dir, { withFileTypes: true });
    for (const ent of entries) {
        const name = ent.name;
        if (ent.isDirectory()) {
            if (SKIP_DIRS.has(name)) continue;
            // skip dot-directories except .vscode if you wish to allow it
            if (name.startsWith(".") && name !== ".vscode") continue;
            await walk(path.join(dir, name), out);
        } else {
            out.push(path.join(dir, name));
        }
    }
    return out;
}

async function ensureLicenseFile() {
    const licensePath = path.join(PROJECT_ROOT, "LICENSE");
    try {
        await fsp.access(licensePath, fs.constants.F_OK);
        return false; // exists
    } catch {
        const MIT_FULL = `MIT License

Copyright (c) 2025 Elena Viter

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
`;
        await fsp.writeFile(licensePath, MIT_FULL, "utf8");
        return true;
    }
}

(async function main() {
    console.log(`🔎 Scanning: ${PROJECT_ROOT}`);
    const files = await walk(PROJECT_ROOT);
    let touched = 0;
    let skipped = 0;
    for (const file of files) {
        try {
            const st = await fsp.lstat(file);
            if (!(await shouldProcessFile(file, st))) {
                skipped++;
                continue;
            }
            const res = await addHeaderToFile(file);
            if (res.changed) touched++;
            else skipped++;
        } catch (e) {
            console.warn(`⚠️  Skipped due to error: ${file}\n   ${e.message}`);
            skipped++;
        }
    }

    const createdLicense = await ensureLicenseFile();

    console.log(`\n✅ Done. Updated ${touched} file(s). Skipped ${skipped}.`);
    if (createdLicense) {
        console.log("📄 Created LICENSE file with MIT text.");
    } else {
        console.log("📄 LICENSE file already present (unchanged).");
    }
})();
