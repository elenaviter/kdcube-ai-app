/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2025 Elena Viter
 */

import {FileText, Loader} from "lucide-react";

export interface FilesPreviewProps {
    content: string | null
    loading: boolean
    error: string | null
}

export const FileLoading = () => {
    return (<div className="flex-1 min-h-1 flex items-center justify-center">
        <Loader className="animate-spin h-8 w-8 text-blue-500"/>
        <span className="ml-2 text-gray-600">Loading data...</span>
    </div>)
}

export const FileLoadingError = ({error}: { error: string }) => {
    return (
        <div className="flex-1 min-h-1 flex items-center justify-center">
            <div className="text-center text-red-500">
                <p>Error loading data: {error}</p>
            </div>
        </div>
    );
}

export const getMimeTypeDisplayName = (mimeType:string) => {
    switch (mimeType) {
        case 'application/pdf':
            return 'PDF';
        case 'text/csv':
            return 'CSV';
        case 'application/json':
            return 'JSON';
        case 'text/yaml':
        case 'text/x-yaml':
        case 'application/yaml':
        case 'application/x-yaml':
            return 'YAML';
        case 'text/markdown':
            return 'Markdown';
        case 'text/plain':
            return 'Text';
        default:
            return mimeType.split('/')[1]?.toUpperCase() || mimeType.toUpperCase();
    }
};

export const getFileIcon = (mimeType:string) => {
    switch (mimeType) {
        case 'application/pdf':
            return <FileText className="text-red-500" size={20}/>;
        case 'text/csv':
            return <FileText className="text-green-500" size={20}/>;
        case 'application/json':
            return <FileText className="text-blue-500" size={20}/>;
        case 'text/yaml':
        case 'text/x-yaml':
        case 'application/yaml':
        case 'application/x-yaml':
            return <FileText className="text-orange-500" size={20}/>;
        case 'text/markdown':
            return <FileText className="text-purple-500" size={20}/>;
        case 'text/plain':
            return <FileText className="text-gray-500" size={20}/>;
        default:
            return <FileText className="text-gray-400" size={20}/>;
    }
};