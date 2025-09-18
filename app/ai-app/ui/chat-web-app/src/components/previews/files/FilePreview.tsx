/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2025 Elena Viter
 */

import {X, Download} from "lucide-react";
import FullScreenOverlay from "../../FullScreenOverlay.tsx";
import {FileRef, handleContentDownload, useFileContent} from "../shared.ts";
import CSVPreview from "./CSVPreview.tsx";
import JSONPreview from "./JSONPreview.tsx";
import MarkdownPreview from "./MarkdownPreview.tsx";
import YAMLPreview from "./YAMLPreview.tsx";
import PDFPreview from "./PDFPreview.tsx";
import TextPreview from "./TextPreview.tsx";
import {getFileIcon, getMimeTypeDisplayName} from "./Shared.tsx";

interface FilePreviewProps {
    isOpen: boolean;
    onClose: () => void;
    file: FileRef | null;
}

const FilePreview = ({isOpen, onClose, file}: FilePreviewProps) => {
    const {content, loading, error} = useFileContent(file);

    if (!isOpen || !file) return null;

    const getPreviewComponent = () => {
        switch (file.mimeType) {
            case 'application/pdf':
                return <PDFPreview content={content} loading={loading} error={error}/>;
            case 'text/csv':
                return <CSVPreview content={content} loading={loading} error={error}/>
            case 'application/json':
                return <JSONPreview content={content} loading={loading} error={error}/>
            case 'text/yaml':
            case 'text/x-yaml':
            case 'application/yaml':
            case 'application/x-yaml':
                return <YAMLPreview content={content} loading={loading} error={error}/>
            case 'text/markdown':
                return <MarkdownPreview content={content} loading={loading} error={error}/>
            default:
                return <TextPreview content={content} loading={loading} error={error}/>
        }
    }

    const handleDownload = () => {
        if (content)
            handleContentDownload(file.name, content, file.mimeType);
    };

    return (
        <>
            <FullScreenOverlay className="backdrop-blur-xs" onClick={onClose}/>
            <div className="fixed inset-0 flex items-center justify-center z-50 p-4">
                <div
                    className="flex flex-col bg-white rounded-lg shadow-xl w-full max-w-6xl h-5/6 max-h-5/6 overflow-hidden border border-gray-400">
                    {/* Header */}
                    <div className="flex items-center justify-between p-4 border-b border-gray-400">
                        <div className="flex items-center space-x-3">
                            {getFileIcon(file.mimeType)}
                            <div>
                                <h3 className="text-lg font-semibold text-gray-800">{file.name}</h3>
                                <p className="text-sm text-gray-500">{file.size} • {getMimeTypeDisplayName(file.mimeType)}</p>
                            </div>
                        </div>
                        <div className="flex items-center space-x-2">
                            <button
                                className="flex items-center px-3 py-1 text-sm bg-green-500 text-white rounded hover:bg-green-600"
                                onClick={handleDownload}
                                title="Download file"
                            >
                                <Download size={14} className="mr-1"/>
                                Download
                            </button>
                            <button
                                onClick={onClose}
                                className="p-2 text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded-lg transition-colors"
                            >
                                <X size={20}/>
                            </button>
                        </div>
                    </div>

                    {getPreviewComponent()}
                </div>
            </div>
        </>
    );
};

export default FilePreview;