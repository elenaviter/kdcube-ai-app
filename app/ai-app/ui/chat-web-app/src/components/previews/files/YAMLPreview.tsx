/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2025 Elena Viter
 */

import {useEffect, useState} from "react";
import {FileLoading, FileLoadingError, FilesPreviewProps} from "./Shared.tsx";
import yaml from 'js-yaml';

const YAMLPreview = ({content, loading, error}: FilesPreviewProps) => {
    const [collapsed, setCollapsed] = useState({});
    const [parsedYaml, setParsedYaml] = useState(null);
    const [parseError, setParseError] = useState(null);

    useEffect(() => {
        if (content) {
            try {
                const yamlData = yaml.load(content);
                setParsedYaml(yamlData);
                setParseError(null);
            } catch (err) {
                console.error('Error parsing YAML:', err);
                setParseError(err.message);
            }
        }
    }, [content]);

    const toggleCollapse = (path) => {
        setCollapsed(prev => ({...prev, [path]: !prev[path]}));
    };

    const renderYAML = (obj, path = '', level = 0) => {
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
                        <span className="text-orange-600 font-medium">{key}</span>
                        <span className="mx-2 text-gray-500">:</span>
                        {!isObject && !isArray && (
                            <span className={`${
                                typeof value === 'string' ? 'text-green-600' :
                                    typeof value === 'number' ? 'text-purple-600' :
                                        typeof value === 'boolean' ? 'text-blue-600' :
                                            value === null ? 'text-gray-400' :
                                                'text-gray-600'
                            }`}>
                                {value === null ? 'null' :
                                    typeof value === 'string' ? value :
                                        String(value)}
                            </span>
                        )}
                        {(isObject || isArray) && !isCollapsed && (
                            <span className="text-gray-500">{isArray ? '' : ''}</span>
                        )}
                    </div>
                    {(isObject || isArray) && !isCollapsed && (
                        <div>
                            {isArray ?
                                (value).map((item, index) => (
                                    <div key={index} style={{marginLeft: `${indent + 20}px`}} className="py-1">
                                        <span className="text-gray-500">- </span>
                                        {typeof item === 'object' && item !== null ? (
                                            <div className="inline-block">
                                                {renderYAML(item, `${currentPath}[${index}]`, level + 1)}
                                            </div>
                                        ) : (
                                            <span className={`${
                                                typeof item === 'string' ? 'text-green-600' :
                                                    typeof item === 'number' ? 'text-purple-600' :
                                                        typeof item === 'boolean' ? 'text-blue-600' :
                                                            item === null ? 'text-gray-400' :
                                                                'text-gray-600'
                                            }`}>
                                                {item === null ? 'null' :
                                                    typeof item === 'string' ? item :
                                                        String(item)}
                                            </span>
                                        )}
                                    </div>
                                )) :
                                renderYAML(value, currentPath, level + 1)
                            }
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
            {/* YAML Toolbar */}
            <div className="flex items-center justify-between p-3 bg-orange-50 border-b border-orange-200">
                <div className="flex items-center space-x-2">
                    <button
                        onClick={() => setCollapsed({})}
                        className="px-3 py-1 text-sm bg-orange-100 text-orange-700 rounded hover:bg-orange-200"
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
                            collectPaths(parsedYaml);
                            setCollapsed(allPaths);
                        }}
                        className="px-3 py-1 text-sm bg-orange-100 text-orange-700 rounded hover:bg-orange-200"
                    >
                        Collapse All
                    </button>
                    {parseError && (
                        <span className="text-red-600 text-sm">Parse Error: {parseError}</span>
                    )}
                </div>
                <div className="text-sm text-orange-600 font-medium">
                    YAML
                </div>
            </div>

            {/* YAML Content */}
            <div className="flex-1 overflow-auto p-2 bg-orange-25">
                <div className="bg-white p-4 rounded font-mono text-sm border border-orange-100">
                    {parsedYaml ? renderYAML(parsedYaml) : (
                        <div className="text-gray-500">
                            {parseError ? 'Invalid YAML format' : 'No valid YAML data found'}
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

export default YAMLPreview;