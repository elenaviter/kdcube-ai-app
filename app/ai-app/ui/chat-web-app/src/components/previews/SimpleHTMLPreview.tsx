/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2025 Elena Viter
 */

import React from "react";
import { X, Download, ExternalLink, Globe } from "lucide-react";
import FullScreenOverlay from "../FullScreenOverlay";
import {getKBAPIBaseAddress} from "../../AppConfig.ts";

interface SimpleHTMLPreviewProps {
    isOpen: boolean;
    onClose: () => void;
    file: {
        name: string;
        size: string;
        mimeType: string;
        url?: string;
        resourceId?: string; // KB resource ID
        version: string;
    } | null;
    originalUrl?: string;
}

const SimpleHTMLPreview = ({ isOpen, onClose, file, originalUrl }: SimpleHTMLPreviewProps) => {
    if (!isOpen || !file) return null;

    // FIXED: Better URL construction with proper parameter ordering
    const getPreviewUrl = () => {
        if (file.resourceId) {
            // Ensure version is provided and properly encoded
            const version = file.version || '1';
            return `${getKBAPIBaseAddress()}/api/kb/resource/${encodeURIComponent(file.resourceId)}/preview?version=${encodeURIComponent(version)}&attached=false`;
        }
        // Fallback - should rarely be used for KB resources
        return file.url;
    };

    const getDownloadUrl = () => {
        if (file.resourceId) {
            const version = file.version || '1';
            return `${getKBAPIBaseAddress()}/api/kb/resource/${encodeURIComponent(file.resourceId)}/preview?version=${encodeURIComponent(version)}&attached=true`;
        }
        return file.url;
    };

    const handleDownload = () => {
        const downloadUrl = getDownloadUrl();
        const link = document.createElement('a');
        link.href = downloadUrl;
        link.download = file.name;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    };

    const previewUrl = getPreviewUrl();

    // Debug logging
    console.log('SimpleHTMLPreview Debug:', {
        resourceId: file.resourceId,
        version: file.version,
        previewUrl,
        downloadUrl: getDownloadUrl(),
        originalUrl
    });

    return (
        <>
            <FullScreenOverlay onClick={onClose} />
            <div className="fixed inset-0 flex items-center justify-center z-50 p-4">
                <div className="bg-white rounded-lg shadow-xl w-full max-w-6xl h-5/6 overflow-hidden">
                    {/* Header */}
                    <div className="flex items-center justify-between p-4 border-b border-gray-400">
                        <div className="flex items-center space-x-3">
                            <Globe className="text-blue-600" size={20} />
                            <div>
                                <h3 className="text-lg font-semibold text-gray-800">{file.name}</h3>
                                <div className="text-sm text-gray-500">
                                    {file.size} • Website Content
                                    {file.resourceId && (
                                        <span className="text-xs ml-2 bg-gray-100 px-2 py-1 rounded">
                                            v{file.version}
                                        </span>
                                    )}
                                </div>
                            </div>
                        </div>
                        <div className="flex items-center space-x-2">
                            {originalUrl && (
                                <a
                                    href={originalUrl}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="flex items-center px-3 py-1 text-sm bg-blue-500 text-white rounded hover:bg-blue-600"
                                >
                                    <ExternalLink size={14} className="mr-1" />
                                    Visit Original
                                </a>
                            )}
                            <button
                                className="flex items-center px-3 py-1 text-sm bg-green-500 text-white rounded hover:bg-green-600"
                                onClick={handleDownload}
                                title="Download HTML"
                            >
                                <Download size={14} className="mr-1" />
                                Download
                            </button>
                            <button
                                onClick={onClose}
                                className="p-2 text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded-lg transition-colors"
                            >
                                <X size={20} />
                            </button>
                        </div>
                    </div>

                    {/* Preview Content with Error Handling */}
                    <div className="h-[calc(100%-4rem)] bg-gray-50">
                        {previewUrl ? (
                            <iframe
                                src={previewUrl}
                                title="Website Preview"
                                className="w-full h-full border-0"
                                style={{ border: 'none' }}
                                onError={(e) => {
                                    console.error('Link preview iframe error:', e);
                                    console.error('Failed URL:', previewUrl);
                                }}
                                onLoad={(e) => {
                                    console.log('Link preview loaded successfully for:', previewUrl);
                                    // Check if iframe loaded an error page
                                    try {
                                        const iframe = e.target as HTMLIFrameElement;
                                        const iframeDoc = iframe.contentDocument || iframe.contentWindow?.document;
                                        if (iframeDoc?.title?.includes('404') || iframeDoc?.title?.includes('Error')) {
                                            console.warn('Iframe loaded an error page');
                                        }
                                    } catch (err) {
                                        // CORS prevents access to iframe content, which is normal
                                        console.log('Cannot access iframe content (normal for cross-origin)');
                                    }
                                }}
                            />
                        ) : (
                            <div className="flex items-center justify-center h-full">
                                <div className="text-center">
                                    <Globe size={48} className="mx-auto text-gray-400 mb-4" />
                                    <p className="text-gray-600">No preview URL available</p>
                                    <p className="text-sm text-gray-500 mt-2">
                                        Resource ID: {file.resourceId || 'None'}<br/>
                                        Version: {file.version || 'None'}
                                    </p>
                                </div>
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </>
    );
};

export default SimpleHTMLPreview;