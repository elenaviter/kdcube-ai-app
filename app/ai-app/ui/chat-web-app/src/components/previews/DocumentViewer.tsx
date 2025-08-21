/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2025 Elena Viter
 */

import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
    X,
    Download,
    Navigation,
    Highlighter,
    ArrowLeft,
    ArrowRight,
    Eye,
    FileText,
    Hash,
    Search,
    ZoomIn,
    ZoomOut,
    RotateCw,
    MapPin,
    List,
    Bookmark
} from 'lucide-react';
import { SearchPreviewContent, BacktrackNavigation } from '../search/SearchInterfaces';

interface EnhancedDocumentViewerProps {
    isOpen: boolean;
    onClose: () => void;
    content: SearchPreviewContent | null;
    searchQuery: string;
    onNavigateToSegment?: (segmentIndex: number) => void;
}

// Navigation Sidebar Component
const NavigationSidebar: React.FC<{
    navigation: BacktrackNavigation[];
    currentIndex: number;
    onNavigate: (index: number) => void;
    searchQuery: string;
    isOpen: boolean;
    onToggle: () => void;
}> = ({ navigation, currentIndex, onNavigate, searchQuery, isOpen, onToggle }) => {
    return (
        <>
            {/* Toggle Button */}
            <button
                onClick={onToggle}
                className={`fixed left-4 top-1/2 transform -translate-y-1/2 z-50 p-2 bg-blue-500 text-white rounded-r shadow-lg transition-all ${
                    isOpen ? 'translate-x-80' : 'translate-x-0'
                }`}
                title="Toggle Navigation"
            >
                <List size={16} />
            </button>

            {/* Sidebar */}
            <div className={`fixed left-0 top-0 h-full w-80 bg-white border-r border-gray-200 transform transition-transform z-40 ${
                isOpen ? 'translate-x-0' : '-translate-x-full'
            }`}>
                <div className="p-4 border-b border-gray-200">
                    <h3 className="font-semibold text-gray-800">Document Segments</h3>
                    <p className="text-sm text-gray-600">{navigation.length} segments found</p>
                </div>

                <div className="flex-1 overflow-y-auto">
                    {navigation.map((nav, index) => (
                        <div
                            key={index}
                            onClick={() => onNavigate(index)}
                            className={`p-3 border-b border-gray-100 cursor-pointer transition-colors ${
                                currentIndex === index
                                    ? 'bg-blue-50 border-l-4 border-l-blue-500'
                                    : 'hover:bg-gray-50'
                            }`}
                        >
                            <div className="flex items-center justify-between mb-1">
                                <span className="font-medium text-sm text-gray-800">
                                    {nav.heading || `Segment ${index + 1}`}
                                </span>
                                <span className="text-xs text-gray-500">
                                    L{nav.start_line}-{nav.end_line}
                                </span>
                            </div>

                            {nav.subheading && (
                                <p className="text-xs text-gray-600 mb-2">{nav.subheading}</p>
                            )}

                            {nav.text && (
                                <p className="text-xs text-gray-700 line-clamp-3">
                                    {nav.text.substring(0, 120)}...
                                </p>
                            )}

                            {nav.citations && nav.citations.length > 0 && (
                                <div className="flex flex-wrap gap-1 mt-2">
                                    {nav.citations.map((citation, citIndex) => (
                                        <span
                                            key={citIndex}
                                            className="bg-yellow-100 text-yellow-800 text-xs px-1 rounded"
                                        >
                                            {citation}
                                        </span>
                                    ))}
                                </div>
                            )}
                        </div>
                    ))}
                </div>
            </div>

            {/* Overlay */}
            {isOpen && (
                <div
                    className="fixed inset-0 bg-black bg-opacity-20 z-30"
                    onClick={onToggle}
                />
            )}
        </>
    );
};

// Enhanced Content Renderer
const ContentRenderer: React.FC<{
    content: string;
    mimeType: string;
    showHighlights: boolean;
    currentNavigation?: BacktrackNavigation;
    scale: number;
    onLineClick?: (lineNumber: number) => void;
}> = ({ content, mimeType, showHighlights, currentNavigation, scale, onLineClick }) => {
    const contentRef = useRef<HTMLDivElement>(null);

    // Scroll to current segment
    useEffect(() => {
        if (currentNavigation && contentRef.current) {
            const lines = contentRef.current.querySelectorAll('[data-line]');
            const targetLine = lines[currentNavigation.start_line - 1];
            if (targetLine) {
                targetLine.scrollIntoView({ behavior: 'smooth', block: 'center' });
            }
        }
    }, [currentNavigation]);

    const renderLineNumbers = (text: string) => {
        const lines = text.split('\n');
        return lines.map((line, index) => (
            <div
                key={index}
                data-line={index + 1}
                className={`flex ${
                    currentNavigation &&
                    index + 1 >= currentNavigation.start_line &&
                    index + 1 <= currentNavigation.end_line
                        ? 'bg-blue-50 border-l-4 border-l-blue-400'
                        : ''
                }`}
                onClick={() => onLineClick?.(index + 1)}
            >
                <span className="flex-shrink-0 w-12 text-xs text-gray-400 text-right pr-3 py-1 bg-gray-50 border-r border-gray-200 cursor-pointer hover:bg-gray-100">
                    {index + 1}
                </span>
                <span className="flex-1 px-3 py-1 font-mono text-sm whitespace-pre-wrap">
                    {line}
                </span>
            </div>
        ));
    };

    const renderMarkdown = (markdown: string) => {
        // Simple markdown renderer with line tracking
        const lines = markdown.split('\n');
        return lines.map((line, index) => {
            const lineNumber = index + 1;
            const isInCurrentSegment = currentNavigation &&
                lineNumber >= currentNavigation.start_line &&
                lineNumber <= currentNavigation.end_line;

            let processedLine = line;

            // Basic markdown parsing
            processedLine = processedLine
                .replace(/^### (.*$)/g, '<h3 class="text-lg font-semibold mt-4 mb-2">$1</h3>')
                .replace(/^## (.*$)/g, '<h2 class="text-xl font-semibold mt-6 mb-3">$1</h2>')
                .replace(/^# (.*$)/g, '<h1 class="text-2xl font-bold mt-8 mb-4">$1</h1>')
                .replace(/\*\*(.*?)\*\*/g, '<strong class="font-semibold">$1</strong>')
                .replace(/\*(.*?)\*/g, '<em class="italic">$1</em>')
                .replace(/`(.*?)`/g, '<code class="bg-gray-100 px-1 rounded text-sm">$1</code>');

            return (
                <div
                    key={index}
                    data-line={lineNumber}
                    className={`flex ${isInCurrentSegment ? 'bg-blue-50 border-l-4 border-l-blue-400' : ''}`}
                    onClick={() => onLineClick?.(lineNumber)}
                >
                    <span className="flex-shrink-0 w-12 text-xs text-gray-400 text-right pr-3 py-1 bg-gray-50 border-r border-gray-200 cursor-pointer hover:bg-gray-100">
                        {lineNumber}
                    </span>
                    <div
                        className="flex-1 px-3 py-1"
                        dangerouslySetInnerHTML={{ __html: processedLine }}
                    />
                </div>
            );
        });
    };

    return (
        <div
            ref={contentRef}
            className="h-full overflow-auto bg-white"
            style={{ fontSize: `${scale * 14}px` }}
        >
            {mimeType === 'text/markdown' || mimeType.includes('markdown') ? (
                <div className="border border-gray-200 rounded">
                    {renderMarkdown(content)}
                </div>
            ) : (
                <div className="border border-gray-200 rounded">
                    {renderLineNumbers(content)}
                </div>
            )}
        </div>
    );
};

// Mini-map Component
const DocumentMinimap: React.FC<{
    navigation: BacktrackNavigation[];
    currentIndex: number;
    totalLines: number;
    onNavigate: (index: number) => void;
}> = ({ navigation, currentIndex, totalLines, onNavigate }) => {
    const minimapHeight = 200;

    return (
        <div className="w-16 bg-gray-100 border-l border-gray-200 flex flex-col">
            <div className="p-2 text-xs font-medium text-gray-600 border-b border-gray-200">
                Map
            </div>
            <div className="flex-1 relative p-1">
                <div
                    className="w-full bg-gray-200 rounded"
                    style={{ height: minimapHeight }}
                >
                    {navigation.map((nav, index) => {
                        const top = (nav.start_line / totalLines) * minimapHeight;
                        const height = Math.max(2, ((nav.end_line - nav.start_line) / totalLines) * minimapHeight);

                        return (
                            <div
                                key={index}
                                onClick={() => onNavigate(index)}
                                className={`absolute w-full cursor-pointer rounded-sm ${
                                    currentIndex === index
                                        ? 'bg-blue-500'
                                        : nav.citations.length > 0
                                            ? 'bg-yellow-400 hover:bg-yellow-500'
                                            : 'bg-gray-400 hover:bg-gray-500'
                                }`}
                                style={{
                                    top: `${top}px`,
                                    height: `${height}px`
                                }}
                                title={`${nav.heading || 'Segment'} (Lines ${nav.start_line}-${nav.end_line})`}
                            />
                        );
                    })}
                </div>
            </div>
        </div>
    );
};

// Main Enhanced Document Viewer
const EnhancedDocumentViewer: React.FC<EnhancedDocumentViewerProps> = ({
                                                                           isOpen,
                                                                           onClose,
                                                                           content,
                                                                           searchQuery,
                                                                           onNavigateToSegment
                                                                       }) => {
    const [navigationState, setNavigationState] = useState<PreviewNavigationState>({
        currentSegmentIndex: 0,
        totalSegments: 0,
        showHighlights: true,
        viewMode: 'original'
    });

    const [scale, setScale] = useState(1);
    const [showNavSidebar, setShowNavSidebar] = useState(true);
    const [showMinimap, setShowMinimap] = useState(true);
    const [bookmarks, setBookmarks] = useState<number[]>([]);

    useEffect(() => {
        if (content?.navigation) {
            setNavigationState(prev => ({
                ...prev,
                totalSegments: content.navigation!.length,
                currentSegmentIndex: 0
            }));
        }
    }, [content]);

    const navigateToSegment = useCallback((index: number) => {
        setNavigationState(prev => ({
            ...prev,
            currentSegmentIndex: index
        }));
        onNavigateToSegment?.(index);
    }, [onNavigateToSegment]);

    const handleKeyboardNavigation = useCallback((e: KeyboardEvent) => {
        if (!content?.navigation) return;

        switch (e.key) {
            case 'ArrowUp':
                if (e.ctrlKey) {
                    e.preventDefault();
                    navigateToSegment(Math.max(0, navigationState.currentSegmentIndex - 1));
                }
                break;
            case 'ArrowDown':
                if (e.ctrlKey) {
                    e.preventDefault();
                    navigateToSegment(Math.min(navigationState.totalSegments - 1, navigationState.currentSegmentIndex + 1));
                }
                break;
            case 'h':
                if (e.ctrlKey) {
                    e.preventDefault();
                    setNavigationState(prev => ({ ...prev, showHighlights: !prev.showHighlights }));
                }
                break;
            case 'Escape':
                onClose();
                break;
        }
    }, [content, navigationState, navigateToSegment, onClose]);

    useEffect(() => {
        if (isOpen) {
            document.addEventListener('keydown', handleKeyboardNavigation);
            return () => document.removeEventListener('keydown', handleKeyboardNavigation);
        }
    }, [isOpen, handleKeyboardNavigation]);

    const toggleBookmark = () => {
        const currentIndex = navigationState.currentSegmentIndex;
        setBookmarks(prev =>
            prev.includes(currentIndex)
                ? prev.filter(i => i !== currentIndex)
                : [...prev, currentIndex]
        );
    };

    const currentNavigation = content?.navigation?.[navigationState.currentSegmentIndex];
    const totalLines = content?.content.split('\n').length || 0;

    if (!isOpen || !content) return null;

    return (
        <div className="fixed inset-0 z-50 bg-black bg-opacity-50">
            <div className="h-full w-full bg-white flex flex-col">
                {/* Header */}
                <div className="flex items-center justify-between p-4 border-b border-gray-200 bg-gray-50">
                    <div className="flex items-center space-x-4">
                        <div>
                            <h3 className="text-lg font-semibold">
                                {content.type === 'original' ? 'Original Document' : 'Processed Content'}
                            </h3>
                            <p className="text-sm text-gray-600">{content.filename}</p>
                        </div>

                        {/* Current Segment Info */}
                        {currentNavigation && (
                            <div className="flex items-center space-x-2 border-l pl-4">
                                <MapPin size={16} className="text-blue-500" />
                                <div className="text-sm">
                                    <span className="font-medium">
                                        {currentNavigation.heading || `Segment ${navigationState.currentSegmentIndex + 1}`}
                                    </span>
                                    <span className="text-gray-500 ml-2">
                                        Lines {currentNavigation.start_line}-{currentNavigation.end_line}
                                    </span>
                                </div>
                            </div>
                        )}
                    </div>

                    {/* Controls */}
                    <div className="flex items-center space-x-2">
                        {/* Navigation Controls */}
                        {content.navigation && content.navigation.length > 1 && (
                            <div className="flex items-center space-x-1 border-r pr-3">
                                <button
                                    onClick={() => navigateToSegment(Math.max(0, navigationState.currentSegmentIndex - 1))}
                                    disabled={navigationState.currentSegmentIndex === 0}
                                    className="p-1 rounded hover:bg-gray-200 disabled:opacity-50 disabled:cursor-not-allowed"
                                    title="Previous segment (Ctrl+↑)"
                                >
                                    <ArrowLeft size={16} />
                                </button>
                                <span className="text-sm text-gray-600 px-2">
                                    {navigationState.currentSegmentIndex + 1} / {navigationState.totalSegments}
                                </span>
                                <button
                                    onClick={() => navigateToSegment(Math.min(navigationState.totalSegments - 1, navigationState.currentSegmentIndex + 1))}
                                    disabled={navigationState.currentSegmentIndex === navigationState.totalSegments - 1}
                                    className="p-1 rounded hover:bg-gray-200 disabled:opacity-50 disabled:cursor-not-allowed"
                                    title="Next segment (Ctrl+↓)"
                                >
                                    <ArrowRight size={16} />
                                </button>
                            </div>
                        )}

                        {/* View Controls */}
                        <div className="flex items-center space-x-1 border-r pr-3">
                            <button
                                onClick={() => setScale(Math.max(0.5, scale - 0.1))}
                                className="p-1 rounded hover:bg-gray-200"
                                title="Zoom out"
                            >
                                <ZoomOut size={16} />
                            </button>
                            <span className="text-sm text-gray-600 px-2">
                                {Math.round(scale * 100)}%
                            </span>
                            <button
                                onClick={() => setScale(Math.min(2, scale + 0.1))}
                                className="p-1 rounded hover:bg-gray-200"
                                title="Zoom in"
                            >
                                <ZoomIn size={16} />
                            </button>
                        </div>

                        {/* Feature Toggles */}
                        <button
                            onClick={() => setNavigationState(prev => ({ ...prev, showHighlights: !prev.showHighlights }))}
                            className={`flex items-center px-2 py-1 text-sm rounded ${
                                navigationState.showHighlights
                                    ? 'bg-yellow-100 text-yellow-800'
                                    : 'bg-gray-100 text-gray-600'
                            }`}
                            title="Toggle highlights (Ctrl+H)"
                        >
                            <Highlighter size={14} className="mr-1" />
                            Highlights
                        </button>

                        <button
                            onClick={toggleBookmark}
                            className={`p-1 rounded ${
                                bookmarks.includes(navigationState.currentSegmentIndex)
                                    ? 'bg-blue-100 text-blue-600'
                                    : 'hover:bg-gray-200'
                            }`}
                            title="Bookmark current segment"
                        >
                            <Bookmark size={16} />
                        </button>

                        <button
                            onClick={() => setShowMinimap(!showMinimap)}
                            className={`p-1 rounded ${showMinimap ? 'bg-blue-100 text-blue-600' : 'hover:bg-gray-200'}`}
                            title="Toggle minimap"
                        >
                            <Hash size={16} />
                        </button>

                        <button
                            onClick={onClose}
                            className="p-2 hover:bg-gray-200 rounded-lg transition-colors"
                        >
                            <X size={20} />
                        </button>
                    </div>
                </div>

                {/* Content Area */}
                <div className="flex-1 flex overflow-hidden">
                    {/* Navigation Sidebar */}
                    {content.navigation && (
                        <NavigationSidebar
                            navigation={content.navigation}
                            currentIndex={navigationState.currentSegmentIndex}
                            onNavigate={navigateToSegment}
                            searchQuery={searchQuery}
                            isOpen={showNavSidebar}
                            onToggle={() => setShowNavSidebar(!showNavSidebar)}
                        />
                    )}

                    {/* Main Content */}
                    <div className={`flex-1 transition-all ${showNavSidebar ? 'ml-80' : 'ml-0'}`}>
                        <div className="flex h-full">
                            <div className="flex-1">
                                <ContentRenderer
                                    content={navigationState.showHighlights && content.highlightedContent ? content.highlightedContent : content.content}
                                    mimeType={content.mimeType}
                                    showHighlights={navigationState.showHighlights}
                                    currentNavigation={currentNavigation}
                                    scale={scale}
                                    onLineClick={(lineNumber) => {
                                        // Find segment containing this line
                                        const segmentIndex = content.navigation?.findIndex(nav =>
                                            lineNumber >= nav.start_line && lineNumber <= nav.end_line
                                        );
                                        if (segmentIndex !== undefined && segmentIndex >= 0) {
                                            navigateToSegment(segmentIndex);
                                        }
                                    }}
                                />
                            </div>

                            {/* Minimap */}
                            {showMinimap && content.navigation && (
                                <DocumentMinimap
                                    navigation={content.navigation}
                                    currentIndex={navigationState.currentSegmentIndex}
                                    totalLines={totalLines}
                                    onNavigate={navigateToSegment}
                                />
                            )}
                        </div>
                    </div>
                </div>

                {/* Footer */}
                <div className="border-t border-gray-200 px-6 py-3 bg-gray-50 text-sm text-gray-500">
                    <div className="flex items-center justify-between">
                        <div className="flex items-center space-x-4">
                            <span>Resource: {content.resource_id} (v{content.version})</span>
                            <span>RN: {content.rn}</span>
                            {bookmarks.length > 0 && (
                                <span>{bookmarks.length} bookmarks</span>
                            )}
                        </div>
                        <div className="text-xs text-gray-400">
                            Use Ctrl+↑/↓ to navigate segments • Ctrl+H to toggle highlights • ESC to close
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default EnhancedDocumentViewer;