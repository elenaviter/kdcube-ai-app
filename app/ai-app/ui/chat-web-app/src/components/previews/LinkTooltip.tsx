/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2025 Elena Viter
 */

import { Globe, ExternalLink } from "lucide-react";

/**
 * Interface for the tooltip props
 */
interface LinkTooltipProps {
    resource: {
        id: string;
        name: string;
        uri: string;
        extraction_info?: Array<{
            metadata?: {
                title?: string;
                description?: string;
                author?: string;
                [key: string]: any;
            };
        }>;
        fully_processed: boolean;
    };
    isVisible: boolean;
    onClose: () => void;
    position?: { x: number; y: number };
}

/**
 * A tooltip component that displays KB metadata for links
 * instead of trying to fetch external Open Graph data
 */
const LinkTooltip = ({ resource, isVisible, onClose, position }: LinkTooltipProps) => {
    if (!isVisible) return null;

    // Extract metadata from KB
    const metadata = resource.extraction_info?.[0]?.metadata;
    const title = metadata?.title || resource.name;
    const description = metadata?.description;
    const author = metadata?.author;

    // Get hostname for display
    let hostname = '';
    try {
        hostname = new URL(resource.uri).hostname;
    } catch {
        hostname = resource.uri;
    }

    return (
        <div
            className="fixed z-50 bg-white shadow-xl rounded-lg border border-gray-200 w-80 overflow-hidden animate-fade-in"
            style={{
                left: position?.x || '50%',
                top: position?.y || '50%',
                transform: position ? 'translate(-10px, 10px)' : 'translate(-50%, -50%)',
                animation: 'fadeIn 0.2s ease-in-out'
            }}
            onMouseLeave={onClose}
        >
            {/* Header */}
            <div className="p-3 border-b border-gray-100">
                <div className="flex items-center space-x-2 mb-1">
                    <Globe size={16} className="text-blue-500" />
                    <span className="text-xs text-gray-500">{hostname}</span>
                    {!resource.fully_processed && (
                        <span className="text-xs bg-yellow-100 text-yellow-700 px-2 py-0.5 rounded">
                            Processing...
                        </span>
                    )}
                </div>
                <h3 className="font-medium text-gray-900 text-sm leading-tight">
                    {title}
                </h3>
            </div>

            {/* Content */}
            <div className="p-3">
                {description && (
                    <p className="text-gray-600 text-xs mb-3 leading-relaxed line-clamp-3">
                        {description.length > 120 ? `${description.substring(0, 120)}...` : description}
                    </p>
                )}

                {author && (
                    <div className="text-xs text-gray-500 mb-2">
                        <strong>Author:</strong> {author}
                    </div>
                )}

                {/* Status */}
                <div className="flex items-center justify-between text-xs">
                    <span className="text-gray-500">
                        {resource.fully_processed ? 'Fully processed' : 'Processing content...'}
                    </span>
                    <a
                        href={resource.uri}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="flex items-center text-blue-600 hover:text-blue-800"
                        onClick={(e) => e.stopPropagation()}
                    >
                        <ExternalLink size={12} className="mr-1" />
                        Visit
                    </a>
                </div>
            </div>

            {/* No metadata fallback */}
            {!metadata && (
                <div className="p-3 text-center">
                    <p className="text-xs text-gray-500">
                        {resource.fully_processed
                            ? 'No additional metadata extracted'
                            : 'Metadata will be available after processing completes'
                        }
                    </p>
                </div>
            )}
        </div>
    );
};

export default LinkTooltip;