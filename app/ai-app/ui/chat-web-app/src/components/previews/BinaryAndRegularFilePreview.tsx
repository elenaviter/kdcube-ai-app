/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2025 Elena Viter
 */

import {SearchPreviewContent} from "../search/SearchInterfaces";
import React, {useState} from "react";
import {AlertCircle, ArrowLeft, ArrowRight, Download, Eye, FileText, Highlighter, MapPin, X, File} from "lucide-react";

export const RegularPreviewModal: React.FC<{
    isOpen: boolean;
    onClose: () => void;
    content: SearchPreviewContent;
    searchQuery: string;
}> = ({ isOpen, onClose, content, searchQuery }) => {
    const [currentSegmentIndex, setCurrentSegmentIndex] = useState(0);
    const [showHighlights, setShowHighlights] = useState(true);

    if (!isOpen) return null;

    const currentNavigation = content.navigation?.[currentSegmentIndex];
    const totalSegments = content.navigation?.length || 0;

    const highlightText = (text: string) => {
        if (!showHighlights) return text;
        return text.replace(
            new RegExp(`(${searchQuery})`, 'gi'),
            '<mark class="bg-yellow-200 px-1 rounded">$1</mark>'
        );
    };

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50">
            <div className="bg-white rounded-lg max-w-6xl max-h-[90vh] overflow-hidden m-4 w-full flex flex-col">
                <div className="flex items-center justify-between p-4 border-b border-gray-200 bg-gray-50">
                    <div className="flex items-center space-x-4">
                        <div>
                            <h3 className="text-lg font-semibold flex items-center">
                                {content.type === 'original' ? (
                                    <><FileText size={20} className="mr-2 text-green-600" />Original Document</>
                                ) : (
                                    <><Eye size={20} className="mr-2 text-blue-600" />Processed Content</>
                                )}
                            </h3>
                            <p className="text-sm text-gray-600">{content.filename}</p>
                        </div>

                        {currentNavigation && (
                            <div className="flex items-center space-x-2 border-l pl-4">
                                <MapPin size={16} className="text-blue-500" />
                                <div className="text-sm">
                                    <span className="font-medium">
                                        {currentNavigation.heading || `Segment ${currentSegmentIndex + 1}`}
                                    </span>
                                    <span className="text-gray-500 ml-2">
                                        Lines {currentNavigation.start_line}-{currentNavigation.end_line}
                                    </span>
                                </div>
                            </div>
                        )}

                        {totalSegments > 1 && (
                            <div className="flex items-center space-x-2 border-l pl-4">
                                <button
                                    onClick={() => setCurrentSegmentIndex(Math.max(0, currentSegmentIndex - 1))}
                                    disabled={currentSegmentIndex === 0}
                                    className="p-1 rounded hover:bg-gray-200 disabled:opacity-50"
                                >
                                    <ArrowLeft size={16} />
                                </button>
                                <span className="text-sm text-gray-600">
                                    {currentSegmentIndex + 1} of {totalSegments}
                                </span>
                                <button
                                    onClick={() => setCurrentSegmentIndex(Math.min(totalSegments - 1, currentSegmentIndex + 1))}
                                    disabled={currentSegmentIndex === totalSegments - 1}
                                    className="p-1 rounded hover:bg-gray-200 disabled:opacity-50"
                                >
                                    <ArrowRight size={16} />
                                </button>
                            </div>
                        )}
                    </div>

                    <div className="flex items-center space-x-2">
                        <button
                            onClick={() => setShowHighlights(!showHighlights)}
                            className={`flex items-center px-3 py-1 text-sm rounded ${
                                showHighlights ? 'bg-yellow-100 text-yellow-800' : 'bg-gray-100 text-gray-600'
                            }`}
                        >
                            <Highlighter size={14} className="mr-1" />
                            Highlights
                        </button>
                        <button onClick={onClose} className="p-2 hover:bg-gray-200 rounded-lg transition-colors">
                            <X size={20} />
                        </button>
                    </div>
                </div>

                <div className="flex-1 overflow-y-auto p-6">
                    {content.mimeType === 'text/markdown' ? (
                        <div
                            className="prose prose-sm max-w-none"
                            dangerouslySetInnerHTML={{ __html: highlightText(content.content) }}
                        />
                    ) : (
                        <pre
                            className="whitespace-pre-wrap font-mono text-sm"
                            dangerouslySetInnerHTML={{ __html: highlightText(content.content) }}
                        />
                    )}
                </div>

                <div className="border-t border-gray-200 px-6 py-3 bg-gray-50 text-sm text-gray-500">
                    <div className="flex items-center justify-between">
                        <span>RN: {content.rn}</span>
                        <span>Resource: {content.resource_id} (v{content.version})</span>
                    </div>
                </div>
            </div>
        </div>
    );
};

export const BinaryFilePreviewModal: React.FC<{
    isOpen: boolean;
    onClose: () => void;
    content: SearchPreviewContent;
}> = ({ isOpen, onClose, content }) => {
    if (!isOpen) return null;

    const handleDownload = () => {
        if (content.downloadUrl) {
            window.open(content.downloadUrl, '_blank');
        }
    };

    const handlePreview = () => {
        if (content.previewUrl) {
            window.open(content.previewUrl, '_blank');
        }
    };

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50">
            <div className="bg-white rounded-lg max-w-md overflow-hidden m-4 w-full">
                <div className="flex items-center justify-between p-4 border-b border-gray-200 bg-gray-50">
                    <div>
                        <h3 className="text-lg font-semibold flex items-center">
                            <File size={20} className="mr-2 text-orange-600" />
                            Binary File Preview
                        </h3>
                        <p className="text-sm text-gray-600">{content.filename}</p>
                    </div>
                    <button onClick={onClose} className="p-2 hover:bg-gray-200 rounded-lg transition-colors">
                        <X size={20} />
                    </button>
                </div>

                <div className="p-6">
                    <div className="text-center">
                        <AlertCircle size={48} className="mx-auto text-orange-500 mb-4" />
                        <h4 className="text-lg font-medium text-gray-900 mb-2">
                            Cannot Display Binary Content
                        </h4>
                        <p className="text-gray-600 mb-6">
                            This is a binary file ({content.mimeType}) that cannot be displayed as text.
                            Use the options below to access the content.
                        </p>

                        <div className="space-y-3">
                            <button
                                onClick={handlePreview}
                                className="w-full flex items-center justify-center px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
                            >
                                <Eye size={16} className="mr-2" />
                                Open in Browser
                            </button>
                            <button
                                onClick={handleDownload}
                                className="w-full flex items-center justify-center px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600"
                            >
                                <Download size={16} className="mr-2" />
                                Download File
                            </button>
                        </div>
                    </div>
                </div>

                <div className="border-t border-gray-200 px-6 py-3 bg-gray-50 text-sm text-gray-500">
                    <div className="text-center">
                        <span>MIME Type: {content.mimeType}</span>
                    </div>
                </div>
            </div>
        </div>
    );
};