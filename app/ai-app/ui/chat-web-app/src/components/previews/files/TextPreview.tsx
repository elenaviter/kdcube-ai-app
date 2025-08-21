/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2025 Elena Viter
 */

import {useState} from "react";
import {FileLoading, FileLoadingError, FilesPreviewProps} from "./Shared.tsx";

const TextPreview = ({content, loading, error}: FilesPreviewProps) => {
    const [fontSize, setFontSize] = useState(14);

    if (loading) {
        return <FileLoading/>
    }

    if (error) {
        return <FileLoadingError error={error}/>;
    }

    return (
        <div className="h-full flex flex-col">
            {/* Text Toolbar */}
            <div className="flex items-center justify-between p-3 bg-gray-100 border-b">
                <div className="flex items-center space-x-2">
                    <span className="text-sm text-gray-600">Font size:</span>
                    <button
                        onClick={() => setFontSize(Math.max(10, fontSize - 1))}
                        className="px-2 py-1 text-sm bg-gray-200 text-gray-700 rounded hover:bg-gray-300"
                    >
                        A-
                    </button>
                    <span className="text-sm">{fontSize}px</span>
                    <button
                        onClick={() => setFontSize(Math.min(24, fontSize + 1))}
                        className="px-2 py-1 text-sm bg-gray-200 text-gray-700 rounded hover:bg-gray-300"
                    >
                        A+
                    </button>
                </div>
            </div>

            {/* Text Content */}
            <div className="flex-1 overflow-auto p-6 bg-white">
                <div
                    className="text-gray-800 whitespace-pre-wrap leading-relaxed"
                    style={{ fontSize: `${fontSize}px` }}
                >
                    {content}
                </div>
            </div>
        </div>
    );
};

export default TextPreview