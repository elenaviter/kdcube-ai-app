/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2025 Elena Viter
 */

import {useEffect, useState} from "react";

export function useWordStreamEffect(text: string, delay = 100, maxDelta = 3) {
    const [displayedText, setDisplayedText] = useState('');
    const [wordIndex, setWordIndex] = useState(0);

    useEffect(() => {
        const words = text.split(' ');
        if (wordIndex < words.length) {
            const delta = Math.min(words.length - wordIndex - 1, maxDelta);
            const timeout = setTimeout(() => {
                setDisplayedText(words.slice(0, wordIndex + 1).join(' '));
                setWordIndex(prev => prev + delta);
            }, delay);

            return () => clearTimeout(timeout);
        }
    }, [text, wordIndex, delay, maxDelta]);

    return displayedText;
}

function closeUpCodeBlocks(text: string, onlyFull = ['mermaid']) {
    const blockRegex = /^[\t ]*(`{3}|~{3})[^\n\r~`]*$/g

    const lines = text.split(/\r?\n|\r|\n/g);
    const result = []
    let currentBlockType: string = ""
    let currentLang: string = ""
    let codeLines: string[] = []
    for (const line of lines) {
        if (blockRegex.test(line)) {
            const trimmedLine = line.trim();
            const lineBlockType = trimmedLine.substring(0, 3)
            const lineLang = trimmedLine.substring(3)
            if (lineBlockType.length > 0) {
                if (currentBlockType.length > 0 && currentBlockType !== lineBlockType) {
                    // we assume that new code block opening is actually part of code
                    codeLines.push(line)
                    continue
                } else if (currentBlockType === lineBlockType && lineLang.length === 0) {
                    result.push(...codeLines, line)
                    codeLines = []
                    currentLang = ""
                    continue
                }
                if (currentBlockType.length === 0) {
                    currentBlockType = lineBlockType
                    currentLang = lineLang
                }

                codeLines.push(line)
            }
        } else {
            if (currentBlockType.length > 0) {
                codeLines.push(line)
            } else {
            result.push(line);
            }
        }
    }
    if (codeLines.length > 1) {
        if (currentLang.length === 0 || !onlyFull.includes(currentLang)) {
            result.push(...codeLines, currentBlockType)
        }
    }
    return result.join('\n');
}

export function closeUpMarkdown(text: string) {
    return closeUpCodeBlocks(text)
}