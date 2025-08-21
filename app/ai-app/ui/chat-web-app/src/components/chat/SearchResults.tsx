/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2025 Elena Viter
 */

import React, { useState, useCallback, useRef, useEffect } from 'react';
import {
    X,
    Eye,
    Copy,
    ChevronDown,
    ChevronUp,
    FileText,
    Globe,
    Database,
    Search,
    Zap,
    Check,
    AlertCircle,
    BookOpen,
    File,
    ExternalLink
} from 'lucide-react';
import {useAuthManagerContext} from "../auth/AuthManager.tsx";
import {getKBAPIBaseAddress} from "../../AppConfig.ts";

// FilePreview component (same as your KB panel)
const FilePreview = ({ isOpen, onClose, file }) => {
    if (!isOpen || !file) return null;

    const getFileIcon = (mimeType) => {
        switch (mimeType) {
            case 'application/pdf':
                return <FileText className="text-red-500" size={20} />;
            case 'text/csv':
                return <FileText className="text-green-500" size={20} />;
            case 'application/json':
                return <FileText className="text-blue-500" size={20} />;
            case 'text/markdown':
                return <FileText className="text-purple-500" size={20} />;
            case 'text/plain':
                return <FileText className="text-gray-500" size={20} />;
            default:
                return <FileText className="text-gray-400" size={20} />;
        }
    };

    const getMimeTypeDisplayName = (mimeType) => {
        switch (mimeType) {
            case 'application/pdf':
                return 'PDF';
            case 'text/csv':
                return 'CSV';
            case 'application/json':
                return 'JSON';
            case 'text/markdown':
                return 'Markdown';
            case 'text/plain':
                return 'Text';
            default:
                return mimeType.split('/')[1]?.toUpperCase() || mimeType.toUpperCase();
        }
    };

    const getPreviewUrl = () => {
        if (file.resourceId) {
            return `${getKBAPIBaseAddress()}/api/kb/resource/${file.resourceId}/preview?attached=false&version=${file.version}`;
        }
        return file.url?.includes('?') ? `${file.url}&attached=false` : `${file.url}?attached=false`;
    };

    const getDownloadUrl = () => {
        if (file.resourceId) {
            return `${getKBAPIBaseAddress()}/api/kb/resource/${file.resourceId}/preview?attached=true&version=${file.version}`;
        }
        return file.url?.includes('?') ? `${file.url}&attached=true&version=${file.version}` : `${file.url}?attached=true&version=${file.version}`;
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

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50">
            <div className="bg-white rounded-lg shadow-xl w-full max-w-6xl h-5/6 overflow-hidden m-4">
                <div className="flex items-center justify-between p-4 border-b border-gray-200">
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
                            <ExternalLink size={14} className="mr-1" />
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

                <div className="h-[calc(100%-4rem)]">
                    <iframe
                        src={previewUrl}
                        title="File Preview"
                        className="w-full h-full border-0"
                        style={{ border: 'none' }}
                        onError={(e) => {
                            console.error('Preview iframe error:', e);
                        }}
                        onLoad={() => {
                            console.log('Preview loaded successfully for:', previewUrl);
                        }}
                    />
                </div>
            </div>
        </div>
    );
};

// Markdown renderer component with fixed scrolling
const MarkdownRenderer = ({ content, className = "", highlightTerms = [] }) => {
    const containerRef = useRef(null);

    const renderMarkdown = (markdown) => {
        if (!markdown) return '';

        const html = markdown
            .replace(/^### (.*$)/gim, '<h3 class="text-lg font-semibold mt-4 mb-2 text-gray-800">$1</h3>')
            .replace(/^## (.*$)/gim, '<h2 class="text-xl font-semibold mt-6 mb-3 text-gray-800">$1</h2>')
            .replace(/^# (.*$)/gim, '<h1 class="text-2xl font-bold mt-8 mb-4 text-gray-900">$1</h1>')
            .replace(/\*\*(.*?)\*\*/g, '<strong class="font-semibold bg-yellow-200 px-1 rounded highlight-term" data-highlight="true">$1</strong>')
            .replace(/\*(.*?)\*/g, '<em class="italic">$1</em>')
            .replace(/```([\s\S]*?)```/g, '<pre class="bg-gray-100 p-3 rounded text-sm overflow-x-auto my-2"><code>$1</code></pre>')
            .replace(/`(.*?)`/g, '<code class="bg-gray-100 px-1 rounded text-sm">$1</code>')
            .replace(/^- (.*$)/gim, '<li class="ml-4 mb-1">• $1</li>')
            .replace(/<mark class=['"](.*?)['"]>(.*?)<\/mark>/g, '<mark class="$1 highlight-term" data-highlight="true">$2</mark>')
            .replace(/\n\n/g, '</p><p class="mb-3">')
            .replace(/\n/g, '<br />');

        return `<div class="prose prose-sm max-w-none"><p class="mb-3">${html}</p></div>`;
    };

    return (
        <div
            ref={containerRef}
            className={`${className}`}
            dangerouslySetInnerHTML={{ __html: renderMarkdown(content) }}
        />
    );
};

// Copy to clipboard hook
const useCopyToClipboard = () => {
    const [copied, setCopied] = useState(false);

    const copy = useCallback(async (text) => {
        try {
            await navigator.clipboard.writeText(text);
            setCopied(true);
            setTimeout(() => setCopied(false), 2000);
            return true;
        } catch (err) {
            console.error('Failed to copy:', err);
            return false;
        }
    }, []);

    return { copy, copied };
};

// Enhanced search result item component
const SearchResultItem = ({ result, index, onPreview, onPreviewFile, isExpanded, onToggleExpand }) => {
    const { copy, copied } = useCopyToClipboard();

    const formatRelevanceScore = (score) => {
        const percentage = (score * 50);
        return `${percentage.toFixed(1)}%`;
    };

    const getResourceNameFromRN = (rn) => {
        const parts = rn.split('|');
        if (parts.length > 1) {
            return parts[1].split(':')[0];
        }
        return 'Unknown';
    };

    const getResourceTypeFromRN = (rn) => {
        if (rn.includes(':file|')) return 'file';
        if (rn.includes(':url|')) return 'website';
        return 'document';
    };

    // Extract resource ID from RN for file preview
    const getResourceIdFromRN = (rn) => {
        // Extract resource ID from RN format like "ef:<tenant>:<project_id>:knowledge_base:raw:file|ML_Best_Practices.pdf:1"
        if (rn && rn.includes(':')) {
            const parts = rn.split(':');
            return parts.length >= 5
                ? parts[4]                // returns "file|ML_Best_Practices.pdf"
                : null;
        }
        return null;
    };

    const resourceName = getResourceNameFromRN(result.backtrack?.raw?.rn || '');
    const resourceType = getResourceTypeFromRN(result.backtrack?.raw?.rn || '');
    const resourceId = getResourceIdFromRN(result.backtrack?.raw?.rn || '');

    // Truncated preview of the first navigation segment
    const previewText = result.backtrack?.segmentation?.navigation?.[0]?.text || result.context_text || '';
    const truncatedPreview = previewText.length > 150 ? previewText.slice(0, 150) + '...' : previewText;

    return (
        <div className="mb-3 border border-gray-200 rounded-lg bg-white overflow-hidden">
            <div className="p-4">
                {/* Header */}
                <div className="flex items-start justify-between mb-3">
                    <div className="flex-1 min-w-0">
                        <h4 className="font-medium text-gray-900 mb-1 flex items-center">
                            {resourceType === 'website' ? (
                                <Globe size={16} className="text-blue-500 mr-2 flex-shrink-0" />
                            ) : (
                                <FileText size={16} className="text-gray-500 mr-2 flex-shrink-0" />
                            )}
                            <span className="truncate">{result.heading || resourceName}</span>
                        </h4>
                        {result.subheading && result.subheading !== result.heading && (
                            <p className="text-sm text-gray-600 mb-2 line-clamp-2">{result.subheading}</p>
                        )}
                        <div className="text-xs text-gray-500 mb-2">
                            From: {resourceName}
                        </div>

                        {/* Compact preview */}
                        <div className="text-sm text-gray-700 bg-gray-50 p-2 rounded mb-2">
                            {truncatedPreview}
                        </div>
                    </div>

                    <div className="ml-4 flex items-center gap-2 flex-shrink-0">
                        <span className="text-xs bg-blue-100 text-blue-800 px-2 py-1 rounded font-medium">
                            {formatRelevanceScore(result.relevance_score)}
                        </span>
                    </div>
                </div>

                {/* Action buttons */}
                <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2 flex-wrap">
                        {/* View File button - same as KB panel */}
                        {resourceId && resourceType === 'file' && (
                            <button
                                onClick={() => onPreviewFile(result, resourceId)}
                                className="flex items-center px-2 py-1 text-xs bg-purple-100 text-purple-700 hover:bg-purple-200 rounded transition-colors"
                                title="View original file (PDF, etc.)"
                            >
                                <Eye size={12} className="mr-1" />
                                View File
                            </button>
                        )}
                        <button
                            onClick={() => onPreview(result, 'extraction')}
                            className="flex items-center px-2 py-1 text-xs bg-green-100 text-green-700 hover:bg-green-200 rounded transition-colors"
                            title="View extracted content"
                        >
                            <FileText size={12} className="mr-1" />
                            Extraction
                        </button>
                        <button
                            onClick={() => onPreview(result, 'original')}
                            className="flex items-center px-2 py-1 text-xs bg-blue-100 text-blue-700 hover:bg-blue-200 rounded transition-colors"
                            title="View raw content"
                        >
                            <File size={12} className="mr-1" />
                            Original
                        </button>
                        <button
                            onClick={() => copy(previewText)}
                            className="flex items-center px-2 py-1 text-xs bg-gray-100 text-gray-700 hover:bg-gray-200 rounded transition-colors"
                            title="Copy content"
                        >
                            {copied ? <Check size={12} className="mr-1" /> : <Copy size={12} className="mr-1" />}
                            {copied ? 'Copied!' : 'Copy'}
                        </button>
                    </div>

                    <button
                        onClick={() => onToggleExpand(index)}
                        className="flex items-center text-xs text-gray-600 hover:text-gray-900 transition-colors"
                    >
                        {isExpanded ? (
                            <>
                                <ChevronUp size={14} className="mr-1" />
                                Hide Details
                            </>
                        ) : (
                            <>
                                <ChevronDown size={14} className="mr-1" />
                                Show Details
                            </>
                        )}
                    </button>
                </div>

                {/* Expanded details */}
                {isExpanded && result.backtrack?.segmentation?.navigation && (
                    <div className="mt-4 pt-4 border-t border-gray-100">
                        <div className="space-y-3">
                            <h5 className="text-sm font-medium text-gray-700 mb-2">
                                All Matches ({result.backtrack.segmentation.navigation.length})
                            </h5>
                            {result.backtrack.segmentation.navigation.map((nav, navIndex) => (
                                <div key={navIndex} className="bg-gray-50 border rounded p-3">
                                    <div className="text-sm text-gray-800 leading-relaxed mb-2">
                                        <MarkdownRenderer content={nav.text} />
                                    </div>
                                    <div className="flex items-center justify-between text-xs text-gray-500">
                                        <span>Lines {nav.start_line}-{nav.end_line}</span>
                                        <div className="flex items-center gap-2">
                                            {nav.citations?.length > 0 && (
                                                <span className="bg-yellow-100 text-yellow-800 px-1 rounded">
                                                    {nav.citations.length} matches
                                                </span>
                                            )}
                                            <button
                                                onClick={() => copy(nav.text)}
                                                className="flex items-center px-2 py-1 bg-white hover:bg-gray-100 rounded transition-colors"
                                            >
                                                <Copy size={10} className="mr-1" />
                                                Copy
                                            </button>
                                        </div>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
};

// Enhanced preview modal with FIXED scrolling and real API integration
const PreviewModal = ({ isOpen, onClose, previewContent, apiService }) => {
    const [contentType, setContentType] = useState('extraction');
    const [loading, setLoading] = useState(false);
    const [currentContent, setCurrentContent] = useState('');
    const [highlightedContent, setHighlightedContent] = useState('');
    const { copy, copied } = useCopyToClipboard();
    const contentRef = useRef(null);
    const authContext = useAuthManagerContext()

    useEffect(() => {
        if (isOpen && previewContent) {
            setContentType(previewContent.contentType || 'extraction');
            loadContent(previewContent.contentType || 'extraction');
        }
    }, [isOpen, previewContent]);

    useEffect(() => {
        if (isOpen && previewContent) {
            loadContent(contentType);
        }
    }, [contentType]);

    const loadContent = async (type) => {
        if (!previewContent || !apiService) return;

        setLoading(true);
        try {
            // Use your existing API service method
            const enhancedPreview = await apiService.getEnhancedPreview(
                previewContent.result,
                type,
                authContext.getUserAuthToken()
            );

            setCurrentContent(enhancedPreview.content);
            setHighlightedContent(enhancedPreview.highlightedContent || enhancedPreview.content);

            // FIXED SCROLLING - Multiple approaches to ensure it works
            setTimeout(() => {
                scrollToFirstHighlight();
            }, 300); // Increased delay

        } catch (error) {
            console.error('Failed to load content:', error);
            setCurrentContent('Failed to load content. Please try again.');
            setHighlightedContent('Failed to load content. Please try again.');
        } finally {
            setLoading(false);
        }
    };

    // IMPROVED scroll to highlight function
    const scrollToFirstHighlight = () => {
        if (!contentRef.current) return;

        const container = contentRef.current;

        // Reset scroll first
        container.scrollTop = 0;

        // Multiple selectors to find highlights
        const highlightSelectors = [
            '[data-highlight="true"]',
            '.highlight-term',
            '.bg-yellow-200',
            'mark',
            'strong.bg-yellow-200'
        ];

        let firstHighlight = null;

        for (const selector of highlightSelectors) {
            firstHighlight = container.querySelector(selector);
            if (firstHighlight) break;
        }

        if (firstHighlight) {
            // Use multiple timing approaches
            requestAnimationFrame(() => {
                requestAnimationFrame(() => {
                    try {
                        // Method 1: scrollIntoView
                        firstHighlight.scrollIntoView({
                            behavior: 'smooth',
                            block: 'center',
                            inline: 'nearest'
                        });

                        // Method 2: Manual scroll calculation as backup
                        setTimeout(() => {
                            const elementRect = firstHighlight.getBoundingClientRect();
                            const containerRect = container.getBoundingClientRect();

                            if (elementRect.top < containerRect.top || elementRect.bottom > containerRect.bottom) {
                                const scrollTop = container.scrollTop + elementRect.top - containerRect.top - 100;
                                container.scrollTo({
                                    top: Math.max(0, scrollTop),
                                    behavior: 'smooth'
                                });
                            }
                        }, 100);

                        // Visual emphasis
                        firstHighlight.style.transition = 'all 0.3s ease';
                        firstHighlight.style.boxShadow = '0 0 15px rgba(59, 130, 246, 0.7)';
                        firstHighlight.style.transform = 'scale(1.02)';

                        setTimeout(() => {
                            firstHighlight.style.boxShadow = '';
                            firstHighlight.style.transform = '';
                        }, 2000);

                    } catch (error) {
                        console.error('Scroll error:', error);
                    }
                });
            });
        }
    };

    if (!isOpen || !previewContent) return null;

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50">
            <div className="bg-white rounded-lg max-w-5xl max-h-[85vh] overflow-hidden m-4 w-full">
                <div className="flex flex-col h-full max-h-[85vh]">
                    {/* Header */}
                    <div className="flex items-center justify-between p-4 border-b border-gray-200 bg-gray-50 flex-shrink-0">
                        <div className="flex-1">
                            <h3 className="text-lg font-semibold">{previewContent.result.heading}</h3>
                            <p className="text-sm text-gray-600">{previewContent.result.subheading}</p>
                        </div>

                        {/* Content type toggle */}
                        <div className="flex items-center gap-2 mx-4">
                            <button
                                onClick={() => setContentType('extraction')}
                                className={`px-3 py-1 text-sm rounded transition-colors ${
                                    contentType === 'extraction'
                                        ? 'bg-green-500 text-white'
                                        : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                                }`}
                            >
                                <FileText size={14} className="inline mr-1" />
                                Extraction
                            </button>
                            <button
                                onClick={() => setContentType('original')}
                                className={`px-3 py-1 text-sm rounded transition-colors ${
                                    contentType === 'original'
                                        ? 'bg-blue-500 text-white'
                                        : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                                }`}
                            >
                                <File size={14} className="inline mr-1" />
                                Original
                            </button>
                        </div>

                        {/* Action buttons */}
                        <div className="flex items-center gap-2">
                            <button
                                onClick={() => copy(currentContent)}
                                className="flex items-center px-3 py-1 text-sm bg-gray-100 hover:bg-gray-200 rounded transition-colors"
                                title="Copy all content"
                            >
                                {copied ? <Check size={14} className="mr-1" /> : <Copy size={14} className="mr-1" />}
                                {copied ? 'Copied!' : 'Copy All'}
                            </button>
                            <button
                                onClick={scrollToFirstHighlight}
                                className="flex items-center px-3 py-1 text-sm bg-blue-100 hover:bg-blue-200 rounded transition-colors"
                                title="Scroll to first highlight"
                            >
                                <Search size={14} className="mr-1" />
                                Find
                            </button>
                            <button
                                onClick={onClose}
                                className="p-2 hover:bg-gray-200 rounded-lg transition-colors"
                            >
                                <X size={20} />
                            </button>
                        </div>
                    </div>

                    {/* Content with FIXED scrolling container */}
                    <div
                        ref={contentRef}
                        className="flex-1 overflow-y-auto p-6 min-h-0"
                        style={{
                            scrollBehavior: 'smooth',
                            height: 'calc(85vh - 120px)' // Fixed height calculation
                        }}
                    >
                        {loading ? (
                            <div className="flex items-center justify-center h-32">
                                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
                                <span className="ml-2 text-gray-600">Loading content...</span>
                            </div>
                        ) : (
                            <MarkdownRenderer
                                content={highlightedContent || currentContent}
                                className="max-w-none"
                            />
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
};

// Main enhanced KB search results component
export const EnhancedKBSearchResults = ({ searchResults, onClose, kbEndpoint, apiService }) => {
    const [selectedResultIndex, setSelectedResultIndex] = useState(0);
    const [expandedResults, setExpandedResults] = useState(new Set());
    const [previewContent, setPreviewContent] = useState(null);
    const [filePreviewData, setFilePreviewData] = useState(null);

    const toggleResultExpansion = useCallback((index) => {
        setExpandedResults(prev => {
            const newSet = new Set(prev);
            if (newSet.has(index)) {
                newSet.delete(index);
            } else {
                newSet.add(index);
            }
            return newSet;
        });
    }, []);

    const handlePreviewResult = async (result, contentType = 'extraction') => {
        setPreviewContent({
            result,
            contentType
        });
    };

    const handlePreviewFile = async (result, resourceId) => {
        try {
            // Get resource details from your API
            const resources = await apiService.listKBResources('file');
            const resource = resources.resources.find(r => r.id === resourceId);

            if (resource) {
                setFilePreviewData({
                    name: resource.filename,
                    size: resource.size_bytes ? `${(resource.size_bytes / 1024 / 1024).toFixed(1)} MB` : 'Unknown size',
                    mimeType: resource.mime || 'application/octet-stream',
                    url: apiService.getKBResourceDownloadUrl(resource.id),
                    resourceId: resource.id,
                    version: resource.version
                });
            } else {
                // Fallback to extracting from RN if resource not found
                const getResourceNameFromRN = (rn) => {
                    const parts = rn.split('|');
                    if (parts.length > 1) {
                        return parts[1].split(':')[0];
                    }
                    return 'Unknown';
                };

                const filename = getResourceNameFromRN(result.backtrack?.raw?.rn || '');

                setFilePreviewData({
                    name: filename,
                    size: 'Unknown size',
                    mimeType: 'application/pdf', // Default assumption
                    resourceId: resourceId,
                    version: '1'
                });
            }
        } catch (error) {
            console.error('Error getting resource details:', error);
            // Fallback behavior
            const getResourceNameFromRN = (rn) => {
                const parts = rn.split('|');
                if (parts.length > 1) {
                    return parts[1].split(':')[0];
                }
                return 'Unknown';
            };

            const filename = getResourceNameFromRN(result.backtrack?.raw?.rn || '');

            setFilePreviewData({
                name: filename,
                size: 'Unknown size',
                mimeType: 'application/pdf',
                resourceId: resourceId,
                version: '1'
            });
        }
    };

    const closePreview = () => {
        setPreviewContent(null);
    };

    const closeFilePreview = () => {
        setFilePreviewData(null);
    };

    if (!searchResults || searchResults.length === 0) {
        return (
            <div className="h-full flex flex-col">
                <div className="px-4 py-3 border-b border-gray-200 bg-gray-50 flex items-center justify-between">
                    <h3 className="font-semibold text-gray-900 text-sm">KB Search Results</h3>
                    <button onClick={onClose} className="p-1 hover:bg-gray-200 rounded text-gray-500 hover:text-gray-700">
                        <X size={14} />
                    </button>
                </div>
                <div className="flex-1 flex items-center justify-center text-gray-500">
                    <div className="text-center">
                        <Database size={24} className="mx-auto mb-2 opacity-50" />
                        <p>No search results available</p>
                    </div>
                </div>
            </div>
        );
    }

    const latestResult = searchResults[0];
    const isAutomatic = latestResult.searchType === 'automatic';

    return (
        <div className="h-full flex flex-col bg-white">
            {/* File Preview Modal - same as KB panel */}
            <FilePreview
                isOpen={!!filePreviewData}
                onClose={closeFilePreview}
                file={filePreviewData}
            />

            {/* Text Content Preview Modal */}
            <PreviewModal
                isOpen={!!previewContent}
                onClose={closePreview}
                previewContent={previewContent}
                apiService={apiService}
            />

            {/* Header */}
            <div className="px-4 py-3 border-b border-gray-200 bg-gray-50">
                <div className="flex items-center justify-between mb-2">
                    <h3 className="font-semibold text-gray-900 text-sm">KB Search Results</h3>
                    <button onClick={onClose} className="p-1 hover:bg-gray-200 rounded text-gray-500 hover:text-gray-700">
                        <X size={14} />
                    </button>
                </div>
                <div className="flex items-center justify-between">
                    <div className="flex items-center text-xs text-gray-600">
                        <span className={`inline-flex items-center px-2 py-1 rounded text-xs font-medium mr-2 ${
                            isAutomatic ? 'bg-blue-100 text-blue-800' : 'bg-green-100 text-green-800'
                        }`}>
                            {isAutomatic ? (
                                <>
                                    <Zap size={10} className="mr-1" />
                                    Auto Search
                                </>
                            ) : (
                                <>
                                    <Search size={10} className="mr-1" />
                                    Manual Search
                                </>
                            )}
                        </span>
                        <span>Query: "{latestResult.query}"</span>
                    </div>
                    <span className="text-xs text-gray-500">{latestResult.results?.length || 0} results</span>
                </div>
            </div>

            {/* Search History Tabs */}
            {searchResults.length > 1 && (
                <div className="border-b border-gray-200 bg-gray-50 p-2">
                    <div className="text-xs text-gray-600 mb-2">Recent Searches:</div>
                    <div className="flex gap-1 overflow-x-auto">
                        {searchResults.slice(0, 5).map((result, index) => (
                            <button
                                key={index}
                                onClick={() => setSelectedResultIndex(index)}
                                className={`flex-shrink-0 px-2 py-1 text-xs rounded transition-colors ${
                                    selectedResultIndex === index
                                        ? 'bg-blue-500 text-white'
                                        : 'bg-white text-gray-700 hover:bg-gray-100'
                                }`}
                            >
                                "{result.query.length > 20 ? result.query.slice(0, 20) + '...' : result.query}"
                            </button>
                        ))}
                    </div>
                </div>
            )}

            {/* Results List */}
            <div className="flex-1 overflow-y-auto p-4">
                {searchResults[selectedResultIndex]?.results?.map((result, index) => (
                    <SearchResultItem
                        key={index}
                        result={result}
                        index={index}
                        onPreview={handlePreviewResult}
                        onPreviewFile={handlePreviewFile}
                        isExpanded={expandedResults.has(index)}
                        onToggleExpand={toggleResultExpansion}
                    />
                ))}

                {(!searchResults[selectedResultIndex]?.results || searchResults[selectedResultIndex].results.length === 0) && (
                    <div className="text-center py-8 text-gray-500">
                        <Search size={24} className="mx-auto mb-2 opacity-50" />
                        <p>No results found for this search</p>
                    </div>
                )}
            </div>
        </div>
    );
};

export default EnhancedKBSearchResults;