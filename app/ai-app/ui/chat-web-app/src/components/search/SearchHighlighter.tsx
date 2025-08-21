/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2025 Elena Viter
 */

// Search utilities and highlighting functions

import { BacktrackNavigation, EnhancedSearchResult } from './SearchInterfaces';
import {useEffect, useState} from "react";
import {apiService} from "../ApiService";
import {useAuthManagerContext} from "../auth/AuthManager.tsx";

export class SearchHighlighter {
    private static highlightClass = 'search-highlight';

    /**
     * Apply smart highlighting to text content
     */
    static highlightText(
        text: string,
        citations: string[],
        highlightClass: string = 'bg-yellow-200 px-1 rounded'
    ): string {
        if (!citations.length) return text;

        let highlightedText = text;

        // Sort citations by length (longer first) to avoid partial matches
        const sortedCitations = citations
            .filter(citation => citation.trim().length > 0)
            .sort((a, b) => b.length - a.length);

        sortedCitations.forEach((citation, index) => {
            // Escape special regex characters
            const escapedCitation = citation.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

            // Create regex with word boundaries for better matching
            const regex = new RegExp(`\\b(${escapedCitation})\\b`, 'gi');

            // Use unique markers to avoid conflicts
            const marker = `__HIGHLIGHT_${index}__`;
            highlightedText = highlightedText.replace(regex, marker);
        });

        // Replace markers with actual highlighting
        sortedCitations.forEach((citation, index) => {
            const marker = `__HIGHLIGHT_${index}__`;
            const replacement = `<mark class="${highlightClass}">${citation}</mark>`;
            highlightedText = highlightedText.replace(new RegExp(marker, 'g'), replacement);
        });

        return highlightedText;
    }

    /**
     * Apply navigation-aware highlighting
     */
    static highlightWithNavigation(
        text: string,
        navigation: BacktrackNavigation[],
        currentSegmentIndex?: number
    ): string {
        const lines = text.split('\n');
        const highlightedLines = [...lines];

        navigation.forEach((nav, index) => {
            const isCurrentSegment = index === currentSegmentIndex;
            const segmentClass = isCurrentSegment
                ? 'bg-blue-50 border-l-4 border-l-blue-400'
                : 'hover:bg-gray-50';

            // Apply segment-level highlighting
            for (let i = nav.start_line - 1; i < nav.end_line; i++) {
                if (i >= 0 && i < highlightedLines.length) {
                    // Apply citations highlighting within this segment
                    let lineContent = highlightedLines[i];
                    if (nav.citations.length > 0) {
                        lineContent = this.highlightText(lineContent, nav.citations);
                    }

                    // Wrap with segment styling
                    highlightedLines[i] = `<div class="${segmentClass}" data-segment="${index}" data-line="${i + 1}">${lineContent}</div>`;
                }
            }
        });

        return highlightedLines.join('\n');
    }

    /**
     * Extract text preview with highlighted context
     */
    static extractPreview(
        text: string,
        citations: string[],
        maxLength: number = 200,
        contextWords: number = 10
    ): string {
        if (!citations.length) {
            return text.substring(0, maxLength) + (text.length > maxLength ? '...' : '');
        }

        const firstCitation = citations[0];
        const lowerText = text.toLowerCase();
        const lowerCitation = firstCitation.toLowerCase();

        const index = lowerText.indexOf(lowerCitation);
        if (index === -1) {
            return text.substring(0, maxLength) + (text.length > maxLength ? '...' : '');
        }

        // Find word boundaries for context
        const words = text.split(/\s+/);
        let wordIndex = 0;
        let charCount = 0;

        // Find word containing the citation
        while (wordIndex < words.length && charCount + words[wordIndex].length < index) {
            charCount += words[wordIndex].length + 1; // +1 for space
            wordIndex++;
        }

        // Extract context around the citation
        const startWord = Math.max(0, wordIndex - contextWords);
        const endWord = Math.min(words.length, wordIndex + contextWords);

        const contextText = words.slice(startWord, endWord).join(' ');
        const highlightedPreview = this.highlightText(contextText, citations);

        const prefix = startWord > 0 ? '...' : '';
        const suffix = endWord < words.length ? '...' : '';

        return prefix + highlightedPreview + suffix;
    }
}

export class SearchResultProcessor {
    /**
     * Process search results for better UI display
     */
    static processResults(results: EnhancedSearchResult[]): EnhancedSearchResult[] {
        return results.map(result => ({
            ...result,
            // Generate smart previews for each text block
            text_blocks: result.text_blocks?.map(block =>
                SearchHighlighter.extractPreview(block, result.backtrack.raw.citations, 150)
            ),
            // Enhance combined text with better highlighting
            combined_text: result.combined_text ?
                SearchHighlighter.highlightText(result.combined_text, result.backtrack.raw.citations) :
                undefined,
            // Add relevance indicators
            relevance_indicator: this.getRelevanceIndicator(result.relevance_score),
            // Add segment count for UI
            segment_count: result.backtrack.segmentation.navigation.length
        }));
    }

    private static getRelevanceIndicator(score: number): 'high' | 'medium' | 'low' {
        if (score >= 0.8) return 'high';
        if (score >= 0.5) return 'medium';
        return 'low';
    }

    /**
     * Group results by resource for better organization
     */
    static groupByResource(results: EnhancedSearchResult[]): Record<string, EnhancedSearchResult[]> {
        return results.reduce((groups, result) => {
            const resourceId = result.resource_id || 'unknown';
            if (!groups[resourceId]) {
                groups[resourceId] = [];
            }
            groups[resourceId].push(result);
            return groups;
        }, {} as Record<string, EnhancedSearchResult[]>);
    }

    /**
     * Filter results by criteria
     */
    static filterResults(
        results: EnhancedSearchResult[],
        filters: {
            minRelevance?: number;
            hasNavigation?: boolean;
            resourceIds?: string[];
            contentTypes?: string[];
        }
    ): EnhancedSearchResult[] {
        return results.filter(result => {
            // Relevance filter
            if (filters.minRelevance && result.relevance_score < filters.minRelevance) {
                return false;
            }

            // Navigation filter
            if (filters.hasNavigation && (!result.backtrack.segmentation.navigation || result.backtrack.segmentation.navigation.length === 0)) {
                return false;
            }

            // Resource ID filter
            if (filters.resourceIds && filters.resourceIds.length > 0 && !filters.resourceIds.includes(result.backtrack.raw.rn || '')) {
                return false;
            }

            return true;
        });
    }
}

export class NavigationHelper {
    /**
     * Find the best navigation item for a given search query
     */
    static findBestNavigationMatch(navigation: BacktrackNavigation[], query: string): number {
        let bestMatch = 0;
        let bestScore = 0;

        navigation.forEach((nav, index) => {
            let score = 0;

            // Score based on citations
            if (nav.citations.includes(query)) {
                score += 10;
            }

            // Score based on text content
            if (nav.text && nav.text.toLowerCase().includes(query.toLowerCase())) {
                score += 5;
            }

            // Score based on heading relevance
            if (nav.heading && nav.heading.toLowerCase().includes(query.toLowerCase())) {
                score += 8;
            }

            // Score based on subheading relevance
            if (nav.subheading && nav.subheading.toLowerCase().includes(query.toLowerCase())) {
                score += 6;
            }

            if (score > bestScore) {
                bestScore = score;
                bestMatch = index;
            }
        });

        return bestMatch;
    }

    /**
     * Generate breadcrumb navigation
     */
    static generateBreadcrumb(navigation: BacktrackNavigation[], currentIndex: number): string[] {
        const current = navigation[currentIndex];
        if (!current) return [];

        const breadcrumb = [];

        if (current.heading) {
            breadcrumb.push(current.heading);
        }

        if (current.subheading && current.subheading !== current.heading) {
            breadcrumb.push(current.subheading);
        }

        return breadcrumb;
    }

    /**
     * Calculate reading time for a navigation segment
     */
    static calculateReadingTime(navigation: BacktrackNavigation): number {
        if (!navigation.text) return 0;

        const wordsPerMinute = 200; // Average reading speed
        const wordCount = navigation.text.split(/\s+/).length;
        return Math.ceil(wordCount / wordsPerMinute);
    }
}

export class ContentAnalyzer {
    /**
     * Analyze content structure and suggest improvements
     */
    static analyzeContent(content: string, navigation: BacktrackNavigation[]): {
        quality_score: number;
        suggestions: string[];
        statistics: {
            total_words: number;
            total_lines: number;
            avg_segment_length: number;
            segments_with_citations: number;
        };
    } {
        const words = content.split(/\s+/).length;
        const lines = content.split('\n').length;
        const segmentsWithCitations = navigation.filter(nav => nav.citations.length > 0).length;
        const avgSegmentLength = navigation.length > 0 ?
            navigation.reduce((sum, nav) => sum + (nav.text?.split(/\s+/).length || 0), 0) / navigation.length : 0;

        const suggestions = [];
        let qualityScore = 70; // Base score

        // Quality analysis
        if (segmentsWithCitations / navigation.length > 0.8) {
            qualityScore += 10;
        } else if (segmentsWithCitations / navigation.length < 0.3) {
            qualityScore -= 15;
            suggestions.push('Many segments lack search term matches - consider refining search terms');
        }

        if (avgSegmentLength < 20) {
            qualityScore -= 10;
            suggestions.push('Segments are quite short - consider adjusting segmentation parameters');
        } else if (avgSegmentLength > 200) {
            qualityScore -= 5;
            suggestions.push('Segments are quite long - consider more granular segmentation');
        }

        if (navigation.length < 3) {
            qualityScore -= 10;
            suggestions.push('Very few segments found - document might be too short or poorly structured');
        }

        return {
            quality_score: Math.max(0, Math.min(100, qualityScore)),
            suggestions,
            statistics: {
                total_words: words,
                total_lines: lines,
                avg_segment_length: Math.round(avgSegmentLength),
                segments_with_citations: segmentsWithCitations
            }
        };
    }
}

export class SearchCache {
    private static cache = new Map<string, { data: any; timestamp: number; ttl: number }>();

    static set(key: string, data: any, ttlMinutes: number = 30): void {
        const ttl = ttlMinutes * 60 * 1000; // Convert to milliseconds
        this.cache.set(key, {
            data,
            timestamp: Date.now(),
            ttl
        });
    }

    static get(key: string): any | null {
        const item = this.cache.get(key);
        if (!item) return null;

        if (Date.now() - item.timestamp > item.ttl) {
            this.cache.delete(key);
            return null;
        }

        return item.data;
    }

    static generateSearchKey(query: string, resourceId?: string, filters?: any): string {
        return `search:${query}:${resourceId || 'all'}:${JSON.stringify(filters || {})}`;
    }

    static generateContentKey(rn: string, version: string): string {
        return `content:${rn}:${version}`;
    }

    static clear(): void {
        this.cache.clear();
    }

    static getStats(): { size: number; items: string[] } {
        return {
            size: this.cache.size,
            items: Array.from(this.cache.keys())
        };
    }
}

// Custom hooks for React components
export const useSearchResults = (query: string, options?: {
    resourceId?: string;
    filters?: any;
    cacheMinutes?: number;
}) => {
    const [results, setResults] = useState<EnhancedSearchResult[]>([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const authContext = useAuthManagerContext();

    useEffect(() => {
        if (!query.trim()) {
            setResults([]);
            return;
        }

        const performSearch = async () => {
            setLoading(true);
            setError(null);

            try {
                // Check cache first
                const cacheKey = SearchCache.generateSearchKey(query, options?.resourceId, options?.filters);
                const cachedResults = SearchCache.get(cacheKey);

                if (cachedResults) {
                    setResults(SearchResultProcessor.processResults(cachedResults));
                    setLoading(false);
                    return;
                }

                // Perform actual search
                const searchResponse = await apiService.searchKBEnhanced({
                    query,
                    resource_id: options?.resourceId,
                    top_k: 10,
                    include_backtrack: true,
                    include_navigation: true
                },
                    authContext);

                // Cache results
                SearchCache.set(cacheKey, searchResponse.results, options?.cacheMinutes || 30);

                // Process and set results
                const processedResults = SearchResultProcessor.processResults(searchResponse.results);
                setResults(processedResults);

            } catch (err) {
                setError(err instanceof Error ? err.message : 'Search failed');
                setResults([]);
            } finally {
                setLoading(false);
            }
        };

        // Debounce search
        const timeoutId = setTimeout(performSearch, 300);
        return () => clearTimeout(timeoutId);

    }, [query, options?.resourceId, JSON.stringify(options?.filters)]);

    return { results, loading, error };
};

export const useContentHighlighting = (rn: string, citations: string[]) => {
    const [highlightedContent, setHighlightedContent] = useState<string>('');
    const [loading, setLoading] = useState(false);

    const authContext = useAuthManagerContext();
    useEffect(() => {
        if (!rn || !citations.length) return;

        const loadHighlightedContent = async () => {
            setLoading(true);

            try {
                const result = await apiService.getContentWithHighlighting(rn, citations, authContext);
                setHighlightedContent(result.highlighted_content);
            } catch (error) {
                console.error('Failed to load highlighted content:', error);
            } finally {
                setLoading(false);
            }
        };

        loadHighlightedContent();
    }, [rn, citations.join(',')]);

    return { highlightedContent, loading };
};

// Export utility classes
export {
    // SearchHighlighter,
    // SearchResultProcessor,
    // NavigationHelper,
    // ContentAnalyzer,
    // SearchCache
};