#!/usr/bin/env node
/**
 * Add license headers to source files (React/TS + Python library).
 * Usage:
 *   node scripts/add-license-headers.js [projectRoot]
 * Options via env:
 *   LICENSE_HEADER=spdx | full   (default: spdx)
 *   LICENSE_YEAR=2025            (default: 2025)
 *   LICENSE_NAME="Elena Viter"   (default: Elena Viter)
 */

const fs = require("fs");
const fsp = fs.promises;
const path = require("path");

const PROJECT_ROOT = path.resolve(process.argv[2] || ".");
const OWNER = process.env.LICENSE_NAME || "Elena Viter";
const YEAR = process.env.LICENSE_YEAR || "2025";
const VARIANT = (process.env.LICENSE_HEADER || "spdx").toLowerCase(); // "spdx" | "full"

const SPDX_TEXT = `SPDX-License-Identifier: MIT
Copyright (c) ${YEAR} ${OWNER}`;

const FULL_TEXT = `MIT License

Copyright (c) ${YEAR} ${OWNER}

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

const HEADER_BODY = VARIANT === "full" ? FULL_TEXT : SPDX_TEXT;

const EXT_STYLE = new Map([
    // JS / TS / CSS family -> block comment
    [".js", "block"], [".jsx", "block"], [".mjs", "block"], [".cjs", "block"],
    [".ts", "block"], [".tsx", "block"], [".d.ts", "block"],
    [".css", "block"], [".scss", "block"], [".sass", "block"],
    // HTML -> html comment
    [".html", "html"],
    // Python & config -> hash comments
    [".py", "hash"], [".pyi", "hash"],
    [".toml", "hash"], // pyproject.toml
    [".ini", "hash"], [".cfg", "hash"], // setup.cfg
    [".yml", "hash"], [".yaml", "hash"], // CI config
    [".sh", "hash"], // optional
]);

const SKIP_DIRS = new Set([
    "node_modules", ".git", "build", "dist", ".next", "out", "coverage",
    ".turbo", ".cache", ".expo", "android", "ios",
    ".venv", "venv", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    ".tox",
]);

const SKIP_FILES_BY_NAME = new Set([
    "package-lock.json", "pnpm-lock.yaml", "yarn.lock",
]);

function isBinaryLikely(buffer) {
    for (let i = 0; i < Math.min(buffer.length, 512); i++) {
        if (buffer[i] === 0) return true;
    }
    return false;
}

function makeHeader(style, body) {
    if (style === "html") {
        return `<!--
${body}
-->

`;
    }
    if (style === "block") {
        const lines = body.split("\n").map((l) => (l ? ` * ${l}` : " *"));
        return `/*\n${lines.join("\n")}\n */\n\n`;
    }
    if (style === "hash") {
        const lines = body.split("\n").map((l) => (l ? `# ${l}` : "#"));
        return `${lines.join("\n")}\n\n`;
    }
    // Fallback to hash
    const lines = body.split("\n").map((l) => (l ? `# ${l}` : "#"));
    return `${lines.join("\n")}\n\n`;
}

function hasHeaderAlready(content) {
    const head = content.slice(0, 1500).toLowerCase();
    // Detect SPDX MIT anywhere near top
    if (head.includes("spdx-license-identifier:") && head.includes("mit")) return true;
    // Detect full MIT license or typical copyright
    if (head.includes("mit license") && head.includes("permission is hereby granted")) return true;
    // Detect your name + copyright near top
    if (head.includes("copyright (c)") && head.includes(OWNER.toLowerCase())) return true;
    return false;
}

async function shouldProcessFile(fullPath, stats) {
    if (!stats.isFile()) return false;
    const base = path.basename(fullPath);
    if (SKIP_FILES_BY_NAME.has(base)) return false;

    const ext = path.extname(fullPath).toLowerCase();
    if (!EXT_STYLE.has(ext)) return false;

    // Skip minified bundles
    if (base.endsWith(".min.js") || base.endsWith(".min.css")) return false;

    // Quick binary check
    const fd = await fsp.open(fullPath, "r");
    const { buffer } = await fd.read(Buffer.alloc(512), 0, 512, 0);
    await fd.close();
    if (isBinaryLikely(buffer)) return false;

    return true;
}

// Special handling for Python: preserve shebang and keep encoding on line 1/2 (PEP 263).
function splitPythonPrologue(content) {
    let rest = content;
    let shebang = "";
    let encoding = "";

    // Shebang on line 1
    if (rest.startsWith("#!")) {
        const nl = rest.indexOf("\n");
        if (nl !== -1) {
            shebang = rest.slice(0, nl + 1);
            rest = rest.slice(nl + 1);
        } else return { shebang: rest, encoding: "", body: "" };
    }

    // Encoding on first or second line (now the first line of 'rest')
    // Pattern per PEP 263
    const encRe = /^[ \t\f]*#.*coding[:=][ \t]*([-\w.]+)/i;
    const nl2 = rest.indexOf("\n");
    const firstLine = nl2 === -1 ? rest : rest.slice(0, nl2 + 1);
    if (encRe.test(firstLine)) {
        encoding = firstLine;
        rest = rest.slice(firstLine.length);
    }

    return { shebang, encoding, body: rest };
}

async function addHeaderToFile(fullPath) {
    const ext = path.extname(fullPath).toLowerCase();
    const style = EXT_STYLE.get(ext);
    let content = await fsp.readFile(fullPath, "utf8");

    if (hasHeaderAlready(content)) return { changed: false, reason: "already-present" };

    // Preserve BOM
    let bom = "";
    if (content.charCodeAt(0) === 0xfeff) {
        bom = "\uFEFF";
        content = content.slice(1);
    }

    // Python prologue handling
    let prologue = "";
    if (ext === ".py" || ext === ".pyi") {
        const { shebang, encoding, body } = splitPythonPrologue(content);
        prologue = shebang + encoding;
        content = body;
    } else {
        // Shebang for other scripts (e.g., .sh)
        if (content.startsWith("#!")) {
            const nl = content.indexOf("\n");
            if (nl !== -1) {
                prologue = content.slice(0, nl + 1);
                content = content.slice(nl + 1);
            } else {
                return { changed: false, reason: "shebang-only" };
            }
        }
    }

    const header = makeHeader(style, HEADER_BODY);
    await fsp.writeFile(fullPath, bom + prologue + header + content, "utf8");
    return { changed: true };
}

async function walk(dir, out = []) {
    const entries = await fsp.readdir(dir, { withFileTypes: true });
    for (const ent of entries) {
        const name = ent.name;
        const full = path.join(dir, name);
        if (ent.isDirectory()) {
            if (SKIP_DIRS.has(name)) continue;
            if (name.endsWith(".egg-info")) continue;
            if (name.startsWith(".") && name !== ".vscode") continue;
            await walk(full, out);
        } else {
            out.push(full);
        }
    }
    return out;
}

async function ensureLicenseFile() {
    const licensePath = path.join(PROJECT_ROOT, "LICENSE");
    try {
        await fsp.access(licensePath, fs.constants.F_OK);
        return false;
    } catch {
        await fsp.writeFile(licensePath, FULL_TEXT, "utf8");
        return true;
    }
}

(async function main() {
    console.log(`🔎 Scanning: ${PROJECT_ROOT}`);
    console.log(`• Variant: ${VARIANT.toUpperCase()}  • Year: ${YEAR}  • Owner: ${OWNER}`);
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
    console.log(createdLicense ? "📄 Created LICENSE file." : "📄 LICENSE file already present.");
})();
