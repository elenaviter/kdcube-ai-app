/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2025 Elena Viter
 */

import React, { useState, useEffect } from "react";
import { Loader } from "lucide-react";
import "./LinkPreview.css"

/**
 * Interface for Open Graph data fetched from a URL
 */
interface OpenGraphData {
    title?: string;
    description?: string;
    image?: string;
    url?: string;
    siteName?: string;
}

/**
 * Component props for the LinkPreview component
 */
interface LinkPreviewProps {
    url: string;
    isVisible: boolean;
    onClose: () => void;
}

/**
 * A component that displays a preview of a website using Open Graph data
 * when hovering over a link.
 */
const LinkPreview = ({ url, isVisible, onClose }: LinkPreviewProps) => {
    const [data, setData] = useState<OpenGraphData | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        if (!isVisible || !url) return;

        const fetchOpenGraphData = async () => {
            setLoading(true);
            setError(null);

            try {
                // In a real implementation, this would be a server endpoint that fetches and parses OG data
                // For demo purposes, we'll use a proxy service (this should be replaced with your backend endpoint)
                const proxyUrl = `https://api.allorigins.win/get?url=${encodeURIComponent(url)}`;

                const response = await fetch(proxyUrl);
                if (!response.ok) throw new Error("Failed to fetch website data");

                const result = await response.json();

                // Simple parsing of Open Graph tags from HTML (in production, use a proper parser on the backend)
                const html = result.contents;
                const parser = new DOMParser();
                const doc = parser.parseFromString(html, "text/html");

                const ogData: OpenGraphData = {
                    title: getMetaContent(doc, 'og:title') || doc.title,
                    description: getMetaContent(doc, 'og:description') || getMetaContent(doc, 'description'),
                    image: getMetaContent(doc, 'og:image'),
                    url: getMetaContent(doc, 'og:url') || url,
                    siteName: getMetaContent(doc, 'og:site_name')
                };

                setData(ogData);
            } catch (err) {
                console.error("Error fetching Open Graph data:", err);
                setError("Failed to load preview");
            } finally {
                setLoading(false);
            }
        };

        // Only fetch data if the preview is visible
        if (isVisible) {
            fetchOpenGraphData();
        }

    }, [url, isVisible]);

    // Helper function to get meta tag content
    const getMetaContent = (doc: Document, property: string): string | undefined => {
        const meta = doc.querySelector(`meta[property="${property}"], meta[name="${property}"]`);
        return meta ? meta.getAttribute('content') || undefined : undefined;
    };

    if (!isVisible) return null;

    return (
        <div
            className="absolute z-50 bg-white shadow-lg rounded-lg border border-gray-200 w-80 overflow-hidden animate-fade-in"
            style={{
                marginTop: '10px',
                animation: 'fadeIn 0.2s ease-in-out'
            }}
            onMouseLeave={onClose}
        >
            {loading && (
                <div className="flex items-center justify-center p-4">
                    <Loader className="animate-spin h-5 w-5 text-blue-500 mr-2" />
                    <span className="text-sm text-gray-600">Loading preview...</span>
                </div>
            )}

            {error && (
                <div className="p-4 text-sm text-red-500">
                    {error}
                </div>
            )}

            {!loading && !error && data && (
                <>
                    {data.image && (
                        <div className="w-full h-40 bg-gray-100 overflow-hidden">
                            <img
                                src={data.image}
                                alt={data.title || "Website preview"}
                                className="w-full h-full object-cover"
                                onError={(e) => {
                                    // Hide image on error
                                    (e.target as HTMLImageElement).style.display = 'none';
                                }}
                            />
                        </div>
                    )}

                    <div className="p-4">
                        <h3 className="font-medium text-gray-900 text-sm mb-1 line-clamp-2">
                            {data.title || "No title available"}
                        </h3>

                        {data.description && (
                            <p className="text-gray-600 text-xs mb-2 line-clamp-3">
                                {data.description}
                            </p>
                        )}

                        <div className="flex items-center text-xs text-gray-500">
                            {data.siteName ? (
                                <span>{data.siteName}</span>
                            ) : (
                                <span>{new URL(url).hostname}</span>
                            )}
                        </div>
                    </div>
                </>
            )}
        </div>
    );
};

export default LinkPreview;