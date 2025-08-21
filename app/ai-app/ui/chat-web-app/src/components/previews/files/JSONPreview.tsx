/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2025 Elena Viter
 */

import {useEffect, useState} from "react";
import {FileLoading, FileLoadingError, FilesPreviewProps} from "./Shared.tsx";

const JSONPreview = ({content, loading, error}: FilesPreviewProps) => {
    const [collapsed, setCollapsed] = useState({});
    const [parsedJson, setParsedJson] = useState(null);

    useEffect(() => {
        if (content) {
            try {
                const jsonData = JSON.parse(content);
                setParsedJson(jsonData);
            } catch (err) {
                console.error('Error parsing JSON:', err);
            }
        }
    }, [content]);

    const toggleCollapse = (path) => {
        setCollapsed(prev => ({...prev, [path]: !prev[path]}));
    };

    const renderJSON = (obj, path = '', level = 0) => {
        if (!obj) return null;
        const indent = level * 20;

        return Object.entries(obj).map(([key, value]) => {
            const currentPath = path ? `${path}.${key}` : key;
            const isObject = typeof value === 'object' && value !== null && !Array.isArray(value);
            const isArray = Array.isArray(value);
            const isCollapsed = collapsed[currentPath];

            return (
                <div key={currentPath} style={{marginLeft: `${indent}px`}}>
                    <div className="flex items-center py-1">
                        {(isObject || isArray) && (
                            <button
                                onClick={() => toggleCollapse(currentPath)}
                                className="mr-2 text-gray-500 hover:text-gray-700"
                            >
                                {isCollapsed ? '▶' : '▼'}
                            </button>
                        )}
                        <span className="text-blue-600 font-medium">"{key}"</span>
                        <span className="mx-2 text-gray-500">:</span>
                        {!isObject && !isArray && (
                            <span className={`${
                                typeof value === 'string' ? 'text-green-600' :
                                    typeof value === 'number' ? 'text-purple-600' :
                                        typeof value === 'boolean' ? 'text-orange-600' :
                                            'text-gray-600'
                            }`}>
                                {typeof value === 'string' ? `"${value}"` : String(value)}
                            </span>
                        )}
                        {(isObject || isArray) && !isCollapsed && (
                            <span className="text-gray-500">{isArray ? '[' : '{'}</span>
                        )}
                    </div>
                    {(isObject || isArray) && !isCollapsed && (
                        <div>
                            {isArray ?
                                (value).map((item, index) => (
                                    <div key={index} style={{marginLeft: `${indent + 20}px`}} className="py-1">
                                        <span className="text-gray-500">[{index}]:</span>
                                        {typeof item === 'object' && item !== null ? (
                                            <div>
                                                {renderJSON(item, `${currentPath}[${index}]`, level + 2)}
                                            </div>
                                        ) : (
                                            <span className="ml-2 text-green-600">
                                                {typeof item === 'string' ? `"${item}"` : String(item)}
                                            </span>
                                        )}
                                    </div>
                                )) :
                                renderJSON(value, currentPath, level + 1)
                            }
                        </div>
                    )}
                    {(isObject || isArray) && !isCollapsed && (
                        <div style={{marginLeft: `${indent}px`}} className="text-gray-500">
                            {isArray ? ']' : '}'}
                        </div>
                    )}
                </div>
            );
        });
    };

    if (loading) {
        return <FileLoading />
    }

    if (error) {
        return <FileLoadingError error={error} />;
    }

    return (
        <div className="flex-1 min-h-1 flex flex-col">
            {/* JSON Toolbar */}
            <div className="flex items-center justify-between p-3 bg-gray-100 border-b border-gray-200">
                <div className="flex items-center space-x-2">
                    <button
                        onClick={() => setCollapsed({})}
                        className="px-3 py-1 text-sm bg-gray-200 text-gray-700 rounded hover:bg-gray-300"
                    >
                        Expand All
                    </button>
                    <button
                        onClick={() => {
                            const allPaths = {};
                            const collectPaths = (obj, path = '') => {
                                if (obj && typeof obj === 'object') {
                                    Object.entries(obj).forEach(([key, value]) => {
                                        const currentPath = path ? `${path}.${key}` : key;
                                        if (typeof value === 'object' && value !== null) {
                                            allPaths[currentPath] = true;
                                            collectPaths(value, currentPath);
                                        }
                                    });
                                }
                            };
                            collectPaths(parsedJson);
                            setCollapsed(allPaths);
                        }}
                        className="px-3 py-1 text-sm bg-gray-200 text-gray-700 rounded hover:bg-gray-300"
                    >
                        Collapse All
                    </button>
                </div>
            </div>

            {/* JSON Content */}
            <div className="flex-1 overflow-auto p-2 bg-gray-50">
                <div className="bg-white p-4 rounded font-mono text-sm">
                    {parsedJson ? renderJSON(parsedJson) : (
                        <div className="text-gray-500">No valid JSON data found</div>
                    )}
                </div>
            </div>
        </div>
    );
};

export default JSONPreview