export interface BacktrackNavigation {
    start_line: number;
    end_line: number;
    start_pos: number;
    end_pos: number;
    citations: string[];
    text?: string; // The actual text content of this base segment
    heading?: string;
    subheading?: string;
}

export interface BacktrackInfo {
    raw: {
        citations: string[];
        rn: string; // Resource name for raw file
    };
    extraction: {
        related_rns: string[]; // Array of extraction resource RNs
        rn: string; // Primary extraction markdown RN
    };
    segmentation: {
        rn: string; // Segmentation file RN
        navigation: BacktrackNavigation[]; // Navigation info for each base segment
    };
}

export interface EnhancedSearchResult {
    query: string;
    relevance_score: number;
    heading: string;
    subheading: string;
    backtrack: BacktrackInfo;
}

export interface EnhancedKBSearchRequest {
    query: string;
    resource_id?: string;
    top_k?: number;
    include_backtrack?: boolean;
    include_navigation?: boolean;
}

export interface EnhancedKBSearchResponse {
    query: string;
    results: EnhancedSearchResult[];
    total_results: number;
    search_metadata: {
        enhanced_search: boolean;
        backtrack_enabled: boolean;
        navigation_enabled: boolean;
        [key: string]: any;
    };
}

export interface SearchPreviewContent {
    type: 'original' | 'extraction';
    resource_id: string;
    version: string;
    rn: string;
    content: string;
    mimeType: string;
    filename: string;
    navigation?: BacktrackNavigation[];
    highlightedContent?: string;
    citations?: string[];
    isBinary?: boolean;
    previewUrl?: string;
    downloadUrl?: string;
}