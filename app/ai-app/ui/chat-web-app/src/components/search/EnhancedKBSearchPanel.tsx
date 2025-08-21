/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2025 Elena Viter
 */

import React, { useState, useEffect } from 'react';
import {
    Search,
    MessageSquare,
    ChevronDown,
    ChevronUp,
    ExternalLink,
    Eye,
    FileText,
    Navigation,
    Highlighter,
    ArrowLeft,
    ArrowRight,
    X,
    Download, Hash, Target, MapPin, Zap, File, AlertCircle
} from 'lucide-react';
import { apiService } from '../kb/ApiService';
import {
    EnhancedSearchResult,
    SearchPreviewContent,
} from './SearchInterfaces';
import {BinaryFilePreviewModal, RegularPreviewModal} from "../previews/BinaryAndRegularFilePreview";
import {useAuthManagerContext} from "../auth/AuthManager.tsx";
import {getWorkingScope} from "../../AppConfig.ts";

// Enhanced Search Result Component
const SearchResultCard: React.FC<{
    result: EnhancedSearchResult;
    index: number;
    expandedResult: string | null;
    onToggleExpand: (resultKey: string) => void;
    onPreviewOriginal: (result: EnhancedSearchResult) => void;
    onPreviewExtraction: (result: EnhancedSearchResult) => void;
}> = ({ result, index, expandedResult, onToggleExpand, onPreviewOriginal, onPreviewExtraction }) => {
    const resultKey = `${index}-${result.relevance_score}`;
    const isExpanded = expandedResult === resultKey;

    const resourceId = result.backtrack.raw.rn.split(':')[4] || 'Unknown';
    const isPdf = resourceId.includes('.pdf');
    const combinedText = result.backtrack.segmentation.navigation
        .map(nav => nav.text)
        .filter(text => text)
        .join(' ')
        .substring(0, 200);

    const getRelevanceColor = (score: number) => {
        if (score >= 0.7) return 'text-green-600 bg-green-50';
        if (score >= 0.5) return 'text-yellow-600 bg-yellow-50';
        return 'text-gray-600 bg-gray-50';
    };

    return (
        <div className="bg-white border border-gray-200 rounded-lg overflow-hidden shadow-sm hover:shadow-md transition-shadow">
            <div
                onClick={() => onToggleExpand(resultKey)}
                className="p-4 cursor-pointer"
            >
                <div className="flex items-start justify-between">
                    <div className="flex-1">
                        <div className="flex items-center justify-between mb-2">
                            <div className="font-medium text-gray-900 flex items-center">
                                {isPdf && <File size={16} className="mr-2 text-orange-500" />}
                                <span className="text-blue-700">
                                    {result.heading.replace(/\*\*/g, '').trim()}
                                </span>
                                {isExpanded ?
                                    <ChevronUp size={16} className="ml-2 text-gray-400" /> :
                                    <ChevronDown size={16} className="ml-2 text-gray-400" />
                                }
                            </div>
                            <div className={`px-2 py-1 rounded-full text-xs font-medium ${getRelevanceColor(result.relevance_score)}`}>
                                {Math.round(result.relevance_score * 100)}%
                            </div>
                        </div>

                        {result.subheading && (
                            <div className="text-sm text-gray-600 mb-2">
                                {result.subheading.replace(/\*\*/g, '').trim()}
                            </div>
                        )}

                        <div className="text-sm text-gray-700 mb-3 leading-relaxed">
                            <div dangerouslySetInnerHTML={{
                                __html: combinedText.replace(
                                    new RegExp(`(${result.query})`, 'gi'),
                                    '<mark class="bg-yellow-200 px-1 rounded">$1</mark>'
                                )
                            }} />
                            {combinedText.length >= 200 && '...'}
                        </div>

                        <div className="flex items-center justify-between">
                            <div className="flex items-center space-x-4 text-xs text-gray-500">
                                <span className="flex items-center">
                                    <Hash size={12} className="mr-1" />
                                    {result.backtrack.segmentation.navigation.length} segments
                                </span>
                                <span className="flex items-center">
                                    <Target size={12} className="mr-1" />
                                    {result.backtrack.raw.citations.length} citations
                                </span>
                                <span>{resourceId.replace(/[|_]/g, ' ')}</span>
                                {isPdf && <span className="text-orange-600 font-medium">PDF</span>}
                            </div>
                            <div className="flex items-center space-x-1">
                                <button
                                    onClick={(e) => {
                                        e.stopPropagation();
                                        onPreviewOriginal(result);
                                    }}
                                    className={`flex items-center px-2 py-1 text-xs rounded ${
                                        isPdf 
                                            ? 'bg-orange-500 text-white hover:bg-orange-600' 
                                            : 'bg-green-500 text-white hover:bg-green-600'
                                    }`}
                                    title={isPdf ? "Open PDF file" : "View in original document"}
                                >
                                    {isPdf ? <File size={12} className="mr-1" /> : <FileText size={12} className="mr-1" />}
                                    {isPdf ? 'Open PDF' : 'Original'}
                                </button>
                                <button
                                    onClick={(e) => {
                                        e.stopPropagation();
                                        onPreviewExtraction(result);
                                    }}
                                    className="flex items-center px-2 py-1 text-xs bg-blue-500 text-white rounded hover:bg-blue-600"
                                    title="View processed content"
                                >
                                    <Eye size={12} className="mr-1" />
                                    Processed
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            {isExpanded && (
                <div className="border-t border-gray-100 p-4 bg-gray-50">
                    <div className="space-y-4">
                        {isPdf && (
                            <div className="bg-orange-50 border border-orange-200 rounded p-3">
                                <div className="flex items-center text-orange-700">
                                    <AlertCircle size={16} className="mr-2" />
                                    <span className="text-sm font-medium">PDF Document</span>
                                </div>
                                <p className="text-orange-600 text-sm mt-1">
                                    Original content is a PDF file. Use "Open PDF" to view the full document or "Processed" to see the extracted text.
                                </p>
                            </div>
                        )}

                        <div>
                            <label className="block text-sm font-medium text-gray-700 mb-2">
                                Segment Breakdown ({result.backtrack.segmentation.navigation.length} parts)
                            </label>
                            <div className="space-y-2 max-h-60 overflow-y-auto">
                                {result.backtrack.segmentation.navigation.map((nav, idx) => (
                                    <div key={idx} className="bg-white p-3 rounded border">
                                        <div className="flex items-center justify-between mb-2">
                                            <div className="flex items-center space-x-2">
                                                <MapPin size={14} className="text-blue-500" />
                                                <span className="font-medium text-sm text-gray-700">
                                                    {nav.heading || `Segment ${idx + 1}`}
                                                </span>
                                            </div>
                                            <span className="text-xs text-gray-500">
                                                Lines {nav.start_line}-{nav.end_line}
                                            </span>
                                        </div>

                                        {nav.text && (
                                            <p className="text-sm text-gray-700 mb-2" dangerouslySetInnerHTML={{
                                                __html: nav.text.substring(0, 150).replace(
                                                    new RegExp(`(${result.query})`, 'gi'),
                                                    '<mark class="bg-yellow-200 px-1 rounded">$1</mark>'
                                                )
                                            }} />
                                        )}

                                        {nav.citations && nav.citations.length > 0 && (
                                            <div className="flex flex-wrap gap-1">
                                                {nav.citations.map((citation, citIdx) => (
                                                    <span key={citIdx} className="bg-yellow-100 text-yellow-800 text-xs px-2 py-1 rounded">
                                                        {citation}
                                                    </span>
                                                ))}
                                            </div>
                                        )}
                                    </div>
                                ))}
                            </div>
                        </div>

                        <div>
                            <label className="block text-sm font-medium text-gray-700 mb-2">
                                Resource References
                            </label>
                            <div className="bg-gray-100 p-3 rounded">
                                <div className="text-xs text-gray-600 space-y-1">
                                    <div><strong>Raw:</strong> <code className="bg-white px-1 rounded">{result.backtrack.raw.rn}</code></div>
                                    <div><strong>Processed:</strong> <code className="bg-white px-1 rounded">{result.backtrack.extraction.rn}</code></div>
                                    <div><strong>Segmentation:</strong> <code className="bg-white px-1 rounded">{result.backtrack.segmentation.rn}</code></div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

// Enhanced Preview Modal with Navigation
const EnhancedPreviewModal: React.FC<{
    isOpen: boolean;
    onClose: () => void;
    content: SearchPreviewContent | null;
    searchQuery: string;
}> = ({ isOpen, onClose, content, searchQuery }) => {
    const [currentSegmentIndex, setCurrentSegmentIndex] = useState(0);
    const [showHighlights, setShowHighlights] = useState(true);
    const [scale, setScale] = useState(1);

    useEffect(() => {
        if (content?.navigation) {
            setCurrentSegmentIndex(0);
        }
    }, [content]);

    if (!isOpen || !content) return null;

    const currentNavigation = content.navigation?.[currentSegmentIndex];
    const totalSegments = content.navigation?.length || 0;

    const highlightText = (text: string) => {
        if (!showHighlights) return text;
        return text.replace(
            new RegExp(`(${searchQuery})`, 'gi'),
            '<mark class="bg-yellow-200 px-1 rounded">$1</mark>'
        );
    };

    const navigateToSegment = (direction: 'prev' | 'next') => {
        if (direction === 'next' && currentSegmentIndex < totalSegments - 1) {
            setCurrentSegmentIndex(currentSegmentIndex + 1);
        } else if (direction === 'prev' && currentSegmentIndex > 0) {
            setCurrentSegmentIndex(currentSegmentIndex - 1);
        }
    };

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50">
            <div className="bg-white rounded-lg max-w-6xl max-h-[90vh] overflow-hidden m-4 w-full flex flex-col">
                {/* Header */}
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

                        {/* Current Segment Info */}
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

                        {/* Navigation Controls */}
                        {totalSegments > 1 && (
                            <div className="flex items-center space-x-2 border-l pl-4">
                                <button
                                    onClick={() => navigateToSegment('prev')}
                                    disabled={currentSegmentIndex === 0}
                                    className="p-1 rounded hover:bg-gray-200 disabled:opacity-50 disabled:cursor-not-allowed"
                                >
                                    <ArrowLeft size={16} />
                                </button>
                                <span className="text-sm text-gray-600">
                                    {currentSegmentIndex + 1} of {totalSegments}
                                </span>
                                <button
                                    onClick={() => navigateToSegment('next')}
                                    disabled={currentSegmentIndex === totalSegments - 1}
                                    className="p-1 rounded hover:bg-gray-200 disabled:opacity-50 disabled:cursor-not-allowed"
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
                                showHighlights
                                    ? 'bg-yellow-100 text-yellow-800'
                                    : 'bg-gray-100 text-gray-600'
                            }`}
                        >
                            <Highlighter size={14} className="mr-1" />
                            Highlights
                        </button>
                        <button
                            onClick={onClose}
                            className="p-2 hover:bg-gray-200 rounded-lg transition-colors"
                        >
                            <X size={20} />
                        </button>
                    </div>
                </div>

                {/* Content */}
                <div className="flex-1 overflow-y-auto p-6" style={{ fontSize: `${scale * 14}px` }}>
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

                {/* Footer */}
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

// Main Enhanced Search Panel
const IntegratedEnhancedSearchPanel: React.FC = () => {
    const [searchQuery, setSearchQuery] = useState('');
    const [searchResults, setSearchResults] = useState<EnhancedSearchResult[]>([]);
    const [isSearching, setIsSearching] = useState(false);
    const [hasSearched, setHasSearched] = useState(false);
    const [expandedResult, setExpandedResult] = useState<string | null>(null);
    const [previewContent, setPreviewContent] = useState<SearchPreviewContent | null>(null);
    const [searchStats, setSearchStats] = useState<{
        totalSegments: number;
        avgRelevance: number;
        processingTime: number
    } | null>(null);
    const authContext = useAuthManagerContext()

    const handleSearch = async () => {
        if (!searchQuery.trim()) return;

        setIsSearching(true);
        setHasSearched(true);
        const startTime = Date.now();
        const workingScope = getWorkingScope()
        try {
            const response = await apiService.searchKBEnhanced({
                query: searchQuery,
                top_k: 10,
                project: workingScope.project,
                tenant: workingScope.tenant,
            }, authContext);

            setSearchResults(response.results);

            // Calculate search stats
            const totalSegments = response.results.reduce((sum, result) =>
                sum + result.backtrack.segmentation.navigation.length, 0
            );
            const avgRelevance = response.results.reduce((sum, result) =>
                sum + result.relevance_score, 0) / response.results.length;

            setSearchStats({
                totalSegments,
                avgRelevance,
                processingTime: Date.now() - startTime
            });

        } catch (error) {
            console.error('Search failed:', error);
            setSearchResults([]);
            setSearchStats(null);
        } finally {
            setIsSearching(false);
        }
    };

    const handlePreviewOriginal = async (result: EnhancedSearchResult) => {
        try {
            const response = await apiService.getContentByRN({
                rn: result.backtrack.raw.rn,
                content_type: "raw"
            }, authContext);
            const isBinary = response.metadata?.is_binary || false;

            setPreviewContent({
                type: 'original',
                resource_id: result.backtrack.raw.rn.split(':')[4] || '',
                version: '1',
                rn: result.backtrack.raw.rn,
                content: response.content,
                mimeType: response.metadata.mime || 'text/plain',
                filename: response.metadata.filename || 'Unknown',
                navigation: result.backtrack.segmentation.navigation,
                citations: result.backtrack.raw.citations,
                isBinary,
                previewUrl: response.metadata?.preview_url,
                downloadUrl: response.metadata?.download_url
            });
        } catch (error) {
            console.error('Failed to load original content:', error);
        }
    };

    const handlePreviewExtraction = async (result: EnhancedSearchResult) => {
        try {
            const response = await apiService.getContentByRN({
                rn: result.backtrack.extraction.rn,
                content_type: "extraction"
            }, authContext);

            setPreviewContent({
                type: 'extraction',
                resource_id: result.backtrack.raw.rn.split(':')[4] || '',
                version: '1',
                rn: result.backtrack.extraction.rn,
                content: response.content,
                mimeType: response.metadata.mime || 'text/markdown',
                filename: response.metadata.filename || 'extraction.md',
                navigation: result.backtrack.segmentation.navigation,
                citations: result.backtrack.raw.citations,
                isBinary: false
            });
        } catch (error) {
            console.error('Failed to load extraction content:', error);
        }
    };

    return (
        <div className="h-full w-full">
            <div className="mb-6">
                <h2 className="text-xl font-semibold text-gray-800 mb-3 flex items-center">
                    <Zap size={24} className="mr-2 text-blue-500" />
                    Knowledge Base Search
                </h2>

                <div className="flex space-x-2">
                    <div className="flex-1 relative">
                        <input
                            type="text"
                            placeholder="Search PDFs, documents, and text files..."
                            value={searchQuery}
                            onChange={(e) => setSearchQuery(e.target.value)}
                            onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
                            className="w-full px-4 py-3 pr-10 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                        />
                        <Search size={18} className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
                    </div>
                    <button
                        onClick={handleSearch}
                        disabled={isSearching || !searchQuery.trim()}
                        className={`px-6 py-3 rounded-lg font-medium ${
                            isSearching || !searchQuery.trim()
                                ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                                : 'bg-blue-500 text-white hover:bg-blue-600'
                        }`}
                    >
                        {isSearching ? 'Searching...' : 'Search'}
                    </button>
                </div>

                {/* Search Stats */}
                {searchStats && (
                    <div className="mt-3 flex items-center space-x-4 text-sm text-gray-600">
                        <span>{searchResults.length} results</span>
                        <span>{searchStats.totalSegments} segments analyzed</span>
                        <span>Avg relevance: {Math.round(searchStats.avgRelevance * 100)}%</span>
                        <span>{searchStats.processingTime}ms</span>
                    </div>
                )}
            </div>

            {hasSearched && (
                <div className="space-y-4">
                    {searchResults.length === 0 ? (
                        <div className="text-center py-12 bg-gray-50 rounded-lg">
                            <MessageSquare size={32} className="mx-auto text-gray-400 mb-4" />
                            <p className="text-gray-600 text-lg mb-2">
                                {isSearching ? 'Searching with enhanced navigation...' : 'No results found'}
                            </p>
                            {!isSearching && (
                                <p className="text-gray-500 text-sm">
                                    Try different keywords or check your spelling
                                </p>
                            )}
                        </div>
                    ) : (
                        <>
                            <div className="text-sm text-gray-600 mb-4 p-3 bg-blue-50 rounded-lg border border-blue-200">
                                <strong>Enhanced Search Results:</strong> Found {searchResults.length} documents with detailed navigation and highlighting support
                            </div>
                            {searchResults.map((result, index) => (
                                <SearchResultCard
                                    key={index}
                                    result={result}
                                    index={index}
                                    expandedResult={expandedResult}
                                    onToggleExpand={setExpandedResult}
                                    onPreviewOriginal={handlePreviewOriginal}
                                    onPreviewExtraction={handlePreviewExtraction}
                                />
                            ))}
                        </>
                    )}
                </div>
            )}

            {/*<EnhancedPreviewModal*/}
            {/*    isOpen={!!previewContent}*/}
            {/*    onClose={() => setPreviewContent(null)}*/}
            {/*    content={previewContent}*/}
            {/*    searchQuery={searchQuery}*/}
            {/*/>*/}
            {/* Choose the right modal based on content type */}
            {previewContent && previewContent.isBinary ? (
                <BinaryFilePreviewModal
                    isOpen={!!previewContent}
                    onClose={() => setPreviewContent(null)}
                    content={previewContent}
                />
            ) : (
                <RegularPreviewModal
                    isOpen={!!previewContent}
                    onClose={() => setPreviewContent(null)}
                    content={previewContent}
                    searchQuery={searchQuery}
                />
            )}
        </div>
    );
};

export default IntegratedEnhancedSearchPanel;