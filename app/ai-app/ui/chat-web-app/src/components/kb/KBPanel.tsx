/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2025 Elena Viter
 */

import {
    Archive,
    Check,
    ChevronDown,
    ChevronUp,
    CloudDownload,
    Code,
    Download,
    Eye,
    File as FileIcon,
    FileAudio,
    FileImage,
    FileSpreadsheet,
    FileText,
    FileVideo,
    Filter,
    Globe,
    Link,
    Plus,
    Search,
    Trash2,
    Upload,
    X
} from "lucide-react";
import React, {Fragment, useEffect, useRef, useState} from "react";
import {useApiDataContext as useDemoDataContext} from "./ApiDataProvider";
import FilePreview from "../previews/files/FilePreview.tsx";
import SimpleHTMLPreview from "../previews/SimpleHTMLPreview";

import {apiService, KBResource} from "./ApiService";
import IntegratedEnhancedSearchPanel from "../search/EnhancedKBSearchPanel";
import {getWorkingScope} from "../../AppConfig.ts";
import {useAppSelector} from "../../app/store.ts";
import {selectAuthToken, selectIdToken} from "../../features/auth/authSlice.ts";


// ================================================================================
//                            FILE ICON HELPER
// ================================================================================

const getFileIcon = (mimeType: string, sourceType: string = 'file', size: number = 20) => {
    const iconProps = {size, className: "text-gray-500"};

    // Handle URL resources differently
    if (sourceType === 'url') {
        return <Globe {...iconProps} className="text-blue-500"/>;
    }

    if (!mimeType) return <FileIcon {...iconProps} />;

    if (mimeType.startsWith('image/')) {
        return <FileImage {...iconProps} className="text-green-500"/>;
    }

    if (mimeType.startsWith('video/')) {
        return <FileVideo {...iconProps} className="text-red-500"/>;
    }

    if (mimeType.startsWith('audio/')) {
        return <FileAudio {...iconProps} className="text-purple-500"/>;
    }

    switch (mimeType) {
        case 'application/pdf':
            return <FileText {...iconProps} className="text-red-600"/>;
        case 'text/csv':
        case 'application/vnd.ms-excel':
        case 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
            return <FileSpreadsheet {...iconProps} className="text-green-600"/>;
        case 'application/json':
        case 'application/xml':
        case 'text/xml':
            return <Code {...iconProps} className="text-blue-600"/>;
        case 'text/markdown':
            return <FileText {...iconProps} className="text-purple-600"/>;
        case 'text/plain':
            return <FileText {...iconProps} className="text-gray-600"/>;
        case 'application/zip':
        case 'application/x-rar-compressed':
        case 'application/x-7z-compressed':
            return <Archive {...iconProps} className="text-yellow-600"/>;
        case 'application/msword':
        case 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
            return <FileText {...iconProps} className="text-blue-600"/>;
        default:
            return <FileIcon {...iconProps} />;
    }
};

// ================================================================================
//                            MARKDOWN RENDERER COMPONENT
// ================================================================================

const MarkdownRenderer = ({content, className = ""}) => {
    const renderMarkdown = (markdown: string) => {
        if (!markdown) return '';

        const html = markdown
            .replace(/^### (.*$)/gim, '<h3 class="text-lg font-semibold mt-4 mb-2">$1</h3>')
            .replace(/^## (.*$)/gim, '<h2 class="text-xl font-semibold mt-6 mb-3">$1</h2>')
            .replace(/^# (.*$)/gim, '<h1 class="text-2xl font-bold mt-8 mb-4">$1</h1>')
            .replace(/\*\*(.*?)\*\*/g, '<strong class="font-semibold bg-yellow-200 px-1">$1</strong>')
            .replace(/\*(.*?)\*/g, '<em class="italic">$1</em>')
            .replace(/```([\s\S]*?)```/g, '<pre class="bg-gray-100 p-3 rounded text-sm overflow-x-auto"><code>$1</code></pre>')
            .replace(/`(.*?)`/g, '<code class="bg-gray-100 px-1 rounded text-sm">$1</code>')
            .replace(/\n\n/g, '</p><p class="mb-3">')
            .replace(/\n/g, '<br />');

        return `<div class="prose prose-sm max-w-none"><p class="mb-3">${html}</p></div>`;
    };

    return (
        <div
            className={`${className}`}
            dangerouslySetInnerHTML={{__html: renderMarkdown(content)}}
        />
    );
};

// ================================================================================
//                            FILES PANEL (Files Only)
// ================================================================================

const FilesPanel = () => {
    const [uploading, setUploading] = useState(false);
    const [uploadProgress, setUploadProgress] = useState<{ [key: string]: number }>({});
    const [uploadErrors, setUploadErrors] = useState<{ [key: string]: string }>({});
    const [uploadStages, setUploadStages] = useState<{ [key: string]: string }>({});
    const [previewFile, setPreviewFile] = useState<any>(null);

    // Cleanup blob URLs to prevent memory leaks
    useEffect(() => {
        return () => {
            if (previewFile?.url && previewFile.url.startsWith('blob:')) {
                URL.revokeObjectURL(previewFile.url);
            }
        };
    }, [previewFile]);

    const [fileResources, setFileResources] = useState<KBResource[]>([]);
    const [expandedResource, setExpandedResource] = useState<string | null>(null);

    const [filterQuery, setFilterQuery] = useState('');
    const [filterType, setFilterType] = useState<string>('all');
    const [filterProcessed, setFilterProcessed] = useState<string>('all');
    const [isFilterOpen, setIsFilterOpen] = useState(false);
    const [availableTypes, setAvailableTypes] = useState<string[]>([]);

    const activeResourcesRef = useRef<Set<string>>(new Set());
    const unsubRefs = useRef<Record<string, () => void>>({});

    const workingScope = getWorkingScope();
    const project = workingScope.project;
    const tenant = workingScope.tenant;

    useEffect(() => {
        loadFileResources();
        return () => {
            // unsubscribe all resource listeners
            Object.values(unsubRefs.current).forEach(fn => fn());
            unsubRefs.current = {};
            activeResourcesRef.current.clear();
            // close the shared socket
            apiService.disconnectKBSocket();
        };
    }, []);

    const loadFileResources = async () => {
        try {
            // const filesResources = await apiService.listKBResources('file', auth.user?.access_token);
            const filesResources = await apiService.listKBResources(project, tenant, 'file');
            const files = filesResources.resources;
            // Filter to only show file resources, not URL resources
            // const files = response.resources.filter(resource => resource.source_type === 'file');
            setFileResources(files);

            // Extract file types for filter
            const types = files.map(file => {
                const ext = file.filename.split('.').pop() || 'unknown';
                return ext.toLowerCase();
            });
            const uniqueTypes = Array.from(new Set(types));
            setAvailableTypes(uniqueTypes);
        } catch (error) {
            console.error('Error loading file resources:', error);
        }
    };

    const filteredFileResources = fileResources.filter(resource => {
        const matchesQuery = resource.name.toLowerCase().includes(filterQuery.toLowerCase()) ||
            resource.filename.toLowerCase().includes(filterQuery.toLowerCase());

        const fileExt = resource.filename.split('.').pop()?.toLowerCase() || '';
        const matchesType = filterType === 'all' || fileExt === filterType.toLowerCase();

        const matchesProcessed = filterProcessed === 'all' ||
            (filterProcessed === 'processed' && resource.fully_processed) ||
            (filterProcessed === 'unprocessed' && !resource.fully_processed);

        return matchesQuery && matchesType && matchesProcessed;
    });

    const authToken = useAppSelector(selectAuthToken)
    const idToken = useAppSelector(selectIdToken)

    const handleFileUpload = () => {
        const input = document.createElement("input");
        input.type = "file";
        input.multiple = true;
        input.accept = ".pdf,.doc,.docx,.txt,.csv,.xlsx,.xls,.pptx,.ppt,.md,.json";

        input.onchange = async (e) => {
            const files = (e.target as HTMLInputElement).files;
            if (!files) return;

            setUploading(true);

            for (const file of Array.from(files)) {
                const fileId = `file_${file.name}_${Date.now()}`;

                // --- initialize UI state ---
                setUploadProgress(p => ({...p, [fileId]: 0}));
                setUploadStages(s => ({...s, [fileId]: "Uploading file..."}));
                setUploadErrors(errs => {
                    const nxt = {...errs};
                    delete nxt[fileId];
                    return nxt;
                });

                let resourceId: string | null = null;

                // helper to clear per-file listeners and maybe close socket
                const cleanup = () => {
                    if (resourceId && unsubRefs.current[resourceId]) {
                        unsubRefs.current[resourceId]();
                        delete unsubRefs.current[resourceId];
                    }
                    if (resourceId) {
                        activeResourcesRef.current.delete(resourceId);
                        if (activeResourcesRef.current.size === 0) {
                            apiService.disconnectKBSocket();
                        }
                    }
                };

                try {
                    // 1) upload (0–20%)
                    const uploadResp = await apiService.uploadFileToKB(
                        project,
                        tenant,
                        file,
                        pct => setUploadProgress(p => ({...p, [fileId]: Math.round(pct * 0.2)})),
                    );
                    if (!uploadResp.success) {
                        throw new Error(uploadResp.message || "Upload failed");
                    }

                    const userSessionId = uploadResp.user_session_id;
                    const meta = uploadResp.resource_metadata as KBResource;
                    resourceId = meta.id;

                    setUploadProgress(p => ({...p, [fileId]: 20}));
                    setUploadStages(s => ({...s, [fileId]: "File uploaded. Preparing processing…"}));

                    // 2) ensure ONE shared socket and wait for it
                    apiService.ensureKBSocket(authToken, idToken, project, tenant, userSessionId);
                    await apiService.waitForKBConnected();

                    // 3) subscribe to this resource’s channel
                    activeResourcesRef.current.add(resourceId);
                    unsubRefs.current[resourceId] = apiService.subscribeResourceProgress(resourceId, (msg: {
                        event: string;
                        resource_id: string;
                        progress?: number;
                        message?: string;
                        error?: string;
                        [k: string]: any;
                    }) => {
                        switch (msg.event) {
                            case "processing_started":
                                setUploadStages(s => ({...s, [fileId]: msg.message || "Processing started…"}));
                                setUploadProgress(p => ({...p, [fileId]: 20}));
                                break;

                            case "processing_extraction":
                            case "processing_segmentation":
                            case "processing_metadata":
                            case "processing_embedding":
                            case "processing_search_indexing":
                            case "processing_search_indexing_complete":
                            default: {
                                if (typeof msg.progress === "number") {
                                    const percent = 20 + Math.round((msg.progress ?? 0) * 80);
                                    setUploadProgress(p => ({...p, [fileId]: percent}));
                                }
                                if (msg.message) {
                                    setUploadStages(s => ({...s, [fileId]: msg.message || "Processing..."}));
                                }
                                if (msg.error) {
                                    setUploadErrors(e => ({...e, [fileId]: msg.error}));
                                }
                                break;
                            }

                            case "processing_completed":
                                setUploadStages(s => ({...s, [fileId]: msg.message || "Completed"}));
                                setUploadProgress(p => ({...p, [fileId]: 100}));
                                loadFileResources();

                                cleanup();

                                // optional: remove progress UI a bit later
                                setTimeout(() => {
                                    setUploadProgress(prev => {
                                        const next = {...prev};
                                        delete next[fileId];
                                        return next;
                                    });
                                    setUploadStages(prev => {
                                        const next = {...prev};
                                        delete next[fileId];
                                        return next;
                                    });
                                }, 2000);
                                return;

                            case "processing_failed":
                                setUploadErrors(e => ({...e, [fileId]: msg.error || "Processing failed"}));
                                cleanup();
                                return;
                        }
                    });

                    // 4) start processing using the shared socket id
                    const sharedSocketId = (apiService as any)["kbSocket"]?.id;
                    setUploadStages(s => ({...s, [fileId]: "Starting processing…"}));
                    await apiService.processKBFileWithSocket(project, tenant, meta, sharedSocketId);

                } catch (err: any) {
                    console.error("Upload/processing error:", err);
                    setUploadErrors(e => ({...e, [fileId]: err?.message || "Failed"}));

                    cleanup();

                    setTimeout(() => {
                        setUploadProgress(prev => {
                            const next = {...prev};
                            delete next[fileId];
                            return next;
                        });
                        setUploadStages(prev => {
                            const next = {...prev};
                            delete next[fileId];
                            return next;
                        });
                    }, 5000);
                }
            }

            setUploading(false);
        };

        input.click();
    };

    const handleDeleteResource = async (resourceId: string) => {
        try {
            await apiService.deleteKBResource(project, tenant, resourceId);
            await loadFileResources();
        } catch (err) {
            console.error("Error deleting resource:", err);
        }
    };

    const handleDownloadResource = async (resourceId: string, filename: string) => {
        try {
            const url = apiService.getKBResourceDownloadUrl(project, tenant, resourceId);
            const link = document.createElement('a');
            link.href = url;
            link.download = filename;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        } catch (err) {
            console.error("Error downloading resource:", err);
        }
    };

    const handlePreview = async (resource: KBResource) => {
        try {
            // Don't fetch content and create blob URLs - use preview endpoint directly
            setPreviewFile({
                name: resource.filename,
                size: resource.size_bytes ? `${(resource.size_bytes / 1024 / 1024).toFixed(1)} MB` : 'Unknown size',
                mimeType: resource.mime || 'application/octet-stream',
                url: apiService.getKBResourceDownloadUrl(project, tenant, resource.id), // For download button
                resourceId: resource.id, // For preview iframe - THIS IS THE KEY FIX
                version: resource.version
            });
        } catch (err) {
            console.error("Error preparing preview:", err);
            setPreviewFile({
                name: resource.filename,
                size: resource.size_bytes ? `${(resource.size_bytes / 1024 / 1024).toFixed(1)} MB` : 'Unknown size',
                mimeType: resource.mime || 'application/octet-stream',
                url: apiService.getKBResourceDownloadUrl(project, tenant, resource.id),
                resourceId: resource.id,
                version: resource.version
            });
        }
    };

    const handleResourceClick = (resourceId: string) => {
        setExpandedResource(expandedResource === resourceId ? null : resourceId);
    };

    const clearFilters = () => {
        setFilterQuery('');
        setFilterType('all');
        setFilterProcessed('all');
    };

    const closePreview = () => {
        // No need to revoke blob URLs since we're not using them anymore
        setPreviewFile(null);
    };

    return (
        <div className="h-full w-full">
            <FilePreview
                isOpen={!!previewFile}
                onClose={closePreview}
                file={previewFile}
            />

            <div className="flex items-center justify-between mb-4">
                <h2 className="text-xl font-semibold text-gray-800">Files & Documents</h2>
                <div className="flex space-x-2">
                    <button
                        onClick={() => setIsFilterOpen((o) => !o)}
                        className="flex items-center px-3 py-1 text-sm rounded border border-gray-300 bg-white hover:bg-gray-100"
                    >
                        <Filter size={14} className="mr-1"/>
                        Filter
                    </button>
                    <button
                        onClick={handleFileUpload}
                        disabled={uploading}
                        className={`flex items-center px-3 py-1 text-sm rounded ${
                            uploading
                                ? 'bg-gray-400 text-gray-200 cursor-not-allowed'
                                : 'bg-blue-500 text-white hover:bg-blue-600'
                        }`}
                    >
                        <Upload size={14} className="mr-1"/>
                        Upload File
                    </button>
                </div>
            </div>

            {/* Filter Panel */}
            {isFilterOpen && (
                <div className="mb-4 p-3 bg-gray-50 border border-gray-400 rounded-lg">
                    <div className="flex flex-col space-y-3">
                        <div>
                            <label className="block text-sm font-medium text-gray-700 mb-1">
                                Search files
                            </label>
                            <div className="relative">
                                <input
                                    type="text"
                                    placeholder="Search by filename..."
                                    value={filterQuery}
                                    onChange={(e) => setFilterQuery(e.target.value)}
                                    className="w-full px-3 py-2 pl-9 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                                />
                                <Search size={16}
                                        className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400"/>
                            </div>
                        </div>

                        <div className="grid grid-cols-2 gap-3">
                            <div>
                                <label className="block text-sm font-medium text-gray-700 mb-1">
                                    File type
                                </label>
                                <select
                                    value={filterType}
                                    onChange={(e) => setFilterType(e.target.value)}
                                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                                >
                                    <option value="all">All Types</option>
                                    {availableTypes.map(type => (
                                        <option key={type} value={type}>
                                            {type.toUpperCase()}
                                        </option>
                                    ))}
                                </select>
                            </div>

                            <div>
                                <label className="block text-sm font-medium text-gray-700 mb-1">
                                    Processing status
                                </label>
                                <select
                                    value={filterProcessed}
                                    onChange={(e) => setFilterProcessed(e.target.value)}
                                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                                >
                                    <option value="all">All Files</option>
                                    <option value="processed">Fully Processed</option>
                                    <option value="unprocessed">Processing/Pending</option>
                                </select>
                            </div>
                        </div>

                        <div className="flex justify-end pt-2">
                            <button
                                onClick={clearFilters}
                                className="px-3 py-1 text-sm text-gray-600 hover:text-gray-800 border border-gray-300 rounded mr-2 hover:bg-gray-100"
                            >
                                Clear filters
                            </button>
                            <button
                                onClick={() => setIsFilterOpen(false)}
                                className="px-3 py-1 text-sm bg-blue-500 text-white rounded hover:bg-blue-600"
                            >
                                Apply
                            </button>
                        </div>
                    </div>
                </div>
            )}

            {/* Upload Progress Indicators */}
            {Object.keys(uploadProgress).length > 0 && (
                <div className="mb-4 grid grid-cols-3 gap-4">
                    {Object.entries(uploadProgress).map(([fileId, progress]) => (
                        <div key={fileId} className="bg-white rounded border p-4 flex flex-col items-center">
                            <FileIcon size={32} className="text-gray-500 mb-2"/>
                            <div className="text-sm text-gray-700 mb-2">
                                {fileId.includes('file_') ? fileId.split("file_")[1].split("_")[0] : fileId.split("_")[0]}
                            </div>
                            <div className="w-full bg-gray-200 rounded-full h-2 mb-1">
                                <div
                                    className="bg-blue-500 h-2 rounded-full transition-all duration-300"
                                    style={{width: `${progress}%`}}
                                />
                            </div>
                            <div className="text-xs text-gray-600">{progress}%</div>
                            {uploadStages[fileId] && (
                                <div className="text-xs text-gray-500 mt-1 text-center">
                                    {uploadStages[fileId]}
                                </div>
                            )}
                        </div>
                    ))}
                </div>
            )}

            {/* Upload Errors */}
            {Object.entries(uploadErrors).length > 0 && (
                <div className="mb-4 space-y-2">
                    {Object.entries(uploadErrors).map(([fileId, error]) => (
                        <div key={fileId} className="bg-red-50 border border-red-200 rounded p-2 flex items-center">
                            <X size={16} className="text-red-500 mr-2"/>
                            <span className="text-sm text-red-700">{error}</span>
                        </div>
                    ))}
                </div>
            )}

            {/* File Resources List */}
            <div className="space-y-2">
                {filteredFileResources.length === 0 && fileResources.length === 0 ? (
                    <div className="p-6 text-center bg-gray-50 border border-gray-400 rounded-lg">
                        <FileIcon size={24} className="mx-auto text-gray-400 mb-2"/>
                        <p className="text-gray-600">No files uploaded yet</p>
                        <button
                            onClick={handleFileUpload}
                            className="mt-2 px-3 py-1 text-sm bg-blue-500 text-white rounded hover:bg-blue-600"
                        >
                            Upload your first file
                        </button>
                    </div>
                ) : filteredFileResources.length === 0 ? (
                    <div className="p-6 text-center bg-gray-50 border border-gray-400 rounded-lg">
                        <Search size={24} className="mx-auto text-gray-400 mb-2"/>
                        <p className="text-gray-600">No files match your filters</p>
                        <button
                            onClick={clearFilters}
                            className="mt-2 text-blue-600 hover:text-blue-800 hover:underline text-sm"
                        >
                            Clear filters
                        </button>
                    </div>
                ) : (
                    filteredFileResources.map((resource) => (
                        <div key={resource.id} className="bg-white border border-gray-400 rounded-lg overflow-hidden">
                            <div
                                className="p-3 cursor-pointer hover:shadow-sm"
                                onClick={() => handleResourceClick(resource.id)}
                            >
                                <div className="flex items-center justify-between">
                                    <div className="flex items-center space-x-3">
                                        {getFileIcon(resource.mime || 'application/octet-stream', resource.source_type, 20)}
                                        <div className="flex-1">
                                            <div className="font-medium text-sm text-gray-900 flex items-center">
                                                {resource.name}
                                                {expandedResource === resource.id ?
                                                    <ChevronUp size={16} className="ml-2 text-gray-400"/> :
                                                    <ChevronDown size={16} className="ml-2 text-gray-400"/>
                                                }
                                            </div>
                                            <div className="text-xs text-gray-500 mb-1">
                                                {resource.filename} • {(resource.size_bytes ? (resource.size_bytes / 1024 / 1024).toFixed(1) + ' MB' : 'Unknown size')}
                                            </div>
                                            <div className="flex items-center space-x-2">
                                                <span className={`inline-block w-2 h-2 rounded-full ${
                                                    resource.fully_processed ? 'bg-green-500' : 'bg-yellow-500'
                                                }`}></span>
                                                <span className="text-xs text-gray-600">
                                                    {resource.fully_processed ? 'Fully Processed' : 'Processing/Pending'}
                                                </span>
                                                <span className="text-xs text-gray-500">
                                                    v{resource.version}
                                                </span>
                                            </div>
                                        </div>
                                    </div>
                                    <div className="flex items-center space-x-2">
                                        <button
                                            onClick={(e) => {
                                                e.stopPropagation();
                                                handlePreview(resource);
                                            }}
                                            className="p-1 text-gray-500 hover:text-blue-600 hover:bg-blue-50 rounded"
                                            title="Preview file"
                                        >
                                            <Eye size={14}/>
                                        </button>
                                        <button
                                            onClick={(e) => {
                                                e.stopPropagation();
                                                handleDownloadResource(resource.id, resource.filename);
                                            }}
                                            className="p-1 text-gray-500 hover:text-green-600 hover:bg-green-50 rounded"
                                            title="Download file"
                                        >
                                            <Download size={14}/>
                                        </button>
                                        <button
                                            onClick={(e) => {
                                                e.stopPropagation();
                                                handleDeleteResource(resource.id);
                                            }}
                                            className="p-1 text-gray-500 hover:text-red-600 hover:bg-red-50 rounded"
                                            title="Delete file"
                                        >
                                            <Trash2 size={14}/>
                                        </button>
                                    </div>
                                </div>
                            </div>

                            {expandedResource === resource.id && (
                                <div className="border-t border-gray-100 p-4 bg-gray-50">
                                    <div className="space-y-3">
                                        <div>
                                            <label className="block text-sm font-medium text-gray-700 mb-1">Processing
                                                Status</label>
                                            <div className="space-y-1">
                                                {Object.entries(resource.processing_status).map(([stage, completed]) => (
                                                    <div key={stage} className="flex items-center space-x-2 text-sm">
                                                        {completed ? (
                                                            <Check size={14} className="text-green-500"/>
                                                        ) : (
                                                            <X size={14} className="text-gray-400"/>
                                                        )}
                                                        <span
                                                            className={completed ? 'text-green-700' : 'text-gray-500'}>
                                                            {stage.charAt(0).toUpperCase() + stage.slice(1)}
                                                        </span>
                                                    </div>
                                                ))}
                                            </div>
                                        </div>

                                        <div className="grid grid-cols-2 gap-4 text-sm">
                                            <div>
                                                <label className="block font-medium text-gray-700 mb-1">Source
                                                    Type</label>
                                                <p className="capitalize">{resource.source_type}</p>
                                            </div>
                                            <div>
                                                <label className="block font-medium text-gray-700 mb-1">Version</label>
                                                <p>{resource.version}</p>
                                            </div>
                                        </div>

                                        <div>
                                            <label className="block text-sm font-medium text-gray-700 mb-1">Resource
                                                ID</label>
                                            <p className="text-xs text-gray-600 bg-gray-100 p-2 rounded">{resource.id}</p>
                                        </div>
                                    </div>
                                </div>
                            )}
                        </div>
                    ))
                )}
            </div>
        </div>
    );
};

// ================================================================================
//                            LINKS PANEL (URLs Only)
// ================================================================================

const LinksPanel = () => {
    const [newUrl, setNewUrl] = useState('');
    const [isValidUrl, setIsValidUrl] = useState(true);
    const [isAdding, setIsAdding] = useState(false);
    const [previewFile, setPreviewFile] = useState<any>(null);

    // Link resources state
    const [linkResources, setLinkResources] = useState<KBResource[]>([]);
    const [expandedResource, setExpandedResource] = useState<string | null>(null);

    // Tooltip state
    const [tooltipResource, setTooltipResource] = useState<KBResource | null>(null);
    const [tooltipPosition, setTooltipPosition] = useState<{ x: number, y: number } | null>(null);

    // Upload tracking for links
    const [uploadProgress, setUploadProgress] = useState<{ [key: string]: number }>({});
    const [uploadErrors, setUploadErrors] = useState<{ [key: string]: string }>({});
    const [uploadStages, setUploadStages] = useState<{ [key: string]: string }>({});

    const activeResourcesRef = useRef<Set<string>>(new Set());
    const unsubRefs = useRef<Record<string, () => void>>({});

    const workingScope = getWorkingScope();
    const project = workingScope.project;
    const tenant = workingScope.tenant;

    useEffect(() => {
        loadLinkResources();
        return () => {
            // unsubscribe all resource listeners
            Object.values(unsubRefs.current).forEach(fn => fn());
            unsubRefs.current = {};
            activeResourcesRef.current.clear();
            // close shared socket when panel unmounts
            apiService.disconnectKBSocket();
        };
    }, []);

    const authToken = useAppSelector(selectAuthToken)
    const idToken = useAppSelector(selectIdToken)

    const loadLinkResources = async () => {
        try {
            // const linksResources = await apiService.listKBResources('url', auth.user?.access_token);
            const linksResources = await apiService.listKBResources(project, tenant, 'url');
            const links = linksResources.resources;
            setLinkResources(links);
        } catch (error) {
            console.error('Error loading link resources:', error);
        }
    };

    // Get extracted title from metadata
    const getDisplayTitle = (resource: KBResource): string => {
        const extractionInfo = resource.extraction_info;
        if (extractionInfo && extractionInfo.length > 0) {
            const firstExtraction = extractionInfo[0];
            const title = firstExtraction?.metadata?.title;
            if (title && title.trim()) {
                return title.trim();
            }
        }

        if (resource.name && resource.name !== new URL(resource.uri).hostname) {
            return resource.name;
        }

        try {
            return new URL(resource.uri).hostname;
        } catch {
            return resource.uri;
        }
    };

    // Enhanced tooltip handlers
    const handleMouseEnter = (resource: KBResource, event: React.MouseEvent) => {
        const rect = (event.target as HTMLElement).getBoundingClientRect();
        setTooltipPosition({
            x: rect.right + 10,
            y: rect.top
        });
        setTooltipResource(resource);
    };

    const handleMouseLeave = () => {
        // Small delay to allow moving to tooltip
        setTimeout(() => {
            setTooltipResource(null);
            setTooltipPosition(null);
        }, 150);
    };

    const autocompleteURL = (url: string) => {
        return url.startsWith("https://") || url.startsWith("http://") ? url : `https://${url}`
    }

    const validateUrl = (url: string) => {
        if (!url.trim()) {
            setIsValidUrl(true);
            return true;
        }

        url = autocompleteURL(url)

        try {
            new URL(url);
            setIsValidUrl(true);
            return true;
        } catch {
            setIsValidUrl(false);
            return false;
        }
    };

    const handleUrlChange = (value: string) => {
        setNewUrl(value);
        validateUrl(value);
    };

    const handleAddLink = async () => {
        if (!(newUrl.trim() && validateUrl(newUrl))) return;

        setIsAdding(true);

        const url = autocompleteURL(newUrl);
        const linkId = `link_${Date.now()}`;

        // --- initialize UI state ---
        setUploadProgress(prev => ({...prev, [linkId]: 0}));
        setUploadStages(prev => ({...prev, [linkId]: "Adding URL to knowledge base..."}));
        setUploadErrors(prev => {
            const nxt = {...prev};
            delete nxt[linkId];
            return nxt;
        });

        let resourceId: string | null = null;

        // helper to clean up subscription + maybe close socket
        const cleanup = () => {
            // remove this resource listener
            if (resourceId && unsubRefs.current[resourceId]) {
                unsubRefs.current[resourceId]();
                delete unsubRefs.current[resourceId];
            }
            if (resourceId) {
                activeResourcesRef.current.delete(resourceId);
            }
            // if no resources left, close the shared socket
            if (activeResourcesRef.current.size === 0) {
                apiService.disconnectKBSocket();
            }
        };

        try {
            // STEP 1: add URL (0–20%)
            const addResp = await apiService.addURLToKB(
                project,
                tenant,
                {url, name: new URL(url).hostname}
            );
            if (!addResp?.success) {
                throw new Error(addResp?.message || "Failed to add URL");
            }

            const resourceMeta = addResp.resource_metadata as KBResource;
            resourceId = resourceMeta.id;
            const userSessionId = addResp.user_session_id;

            setUploadProgress(prev => ({...prev, [linkId]: 20}));
            setUploadStages(prev => ({...prev, [linkId]: "URL added. Preparing processing…"}));

            // STEP 2: ensure ONE shared socket and wait for connection
            apiService.ensureKBSocket(authToken, idToken, project, tenant, userSessionId);
            await apiService.waitForKBConnected();

            // Mark this resource as active
            activeResourcesRef.current.add(resourceId);

            // STEP 3: subscribe to the resource-specific channel
            unsubRefs.current[resourceId] = apiService.subscribeResourceProgress(resourceId, (msg: {
                event: string;
                resource_id: string;
                progress?: number;
                message?: string;
                error?: string;
                [k: string]: any;
            }) => {
                switch (msg.event) {
                    case "processing_started":
                        setUploadStages(s => ({...s, [linkId]: msg.message || "Processing started…"}));
                        setUploadProgress(p => ({...p, [linkId]: 20}));
                        break;

                    case "processing_extraction":
                    case "processing_segmentation":
                    case "processing_metadata":
                    case "processing_embedding":
                    case "processing_search_indexing":
                    case "processing_search_indexing_complete":
                    default: {
                        if (typeof msg.progress === "number") {
                            const processProgress = 20 + Math.round((msg.progress ?? 0) * 80);
                            setUploadProgress(p => ({...p, [linkId]: processProgress}));
                        }
                        if (msg.message) {
                            setUploadStages(s => ({...s, [linkId]: msg.message || "Processing..."}));
                        }
                        if (msg.error) {
                            setUploadErrors(e => ({...e, [linkId]: msg.error}));
                        }
                        break;
                    }

                    case "processing_completed":
                        setUploadStages(s => ({...s, [linkId]: msg.message || "Completed"}));
                        setUploadProgress(p => ({...p, [linkId]: 100}));
                        loadLinkResources();

                        cleanup();

                        // clear progress UI a bit later (optional)
                        setTimeout(() => {
                            setUploadProgress(prev => {
                                const next = {...prev};
                                delete next[linkId];
                                return next;
                            });
                            setUploadStages(prev => {
                                const next = {...prev};
                                delete next[linkId];
                                return next;
                            });
                        }, 2000);
                        return;

                    case "processing_failed":
                        setUploadErrors(e => ({...e, [linkId]: msg.error || "Processing failed"}));
                        cleanup();
                        return;
                }
            });

            setUploadStages(prev => ({...prev, [linkId]: "Starting URL processing…"}));

            // STEP 4: start processing (shared socket id is fine)
            const sharedSocketId = (apiService as any)["kbSocket"]?.id;
            await apiService.processKBURLWithSocket(
                project,
                tenant,
                resourceMeta,
                sharedSocketId,
                "retrieval_only"
            );

            // clean input state
            setNewUrl("");
            setIsValidUrl(true);

        } catch (error: any) {
            console.error("Error adding/processing URL:", error);
            setUploadErrors(prev => ({
                ...prev,
                [linkId]: error?.message || "Failed to add/process URL"
            }));

            cleanup();

            setTimeout(() => {
                setUploadProgress(prev => {
                    const next = {...prev};
                    delete next[linkId];
                    return next;
                });
                setUploadStages(prev => {
                    const next = {...prev};
                    delete next[linkId];
                    return next;
                });
            }, 5000);
        } finally {
            setIsAdding(false);
        }
    };

    const handleKeyPress = (e: React.KeyboardEvent) => {
        if (e.key === 'Enter' && isValidUrl && !isAdding) {
            handleAddLink();
        }
    };

    const handleDeleteResource = async (resourceId: string) => {
        try {
            await apiService.deleteKBResource(project, tenant, resourceId);
            await loadLinkResources();
        } catch (err) {
            console.error("Error deleting link:", err);
        }
    };

    const handleDownloadResource = async (resourceId: string, filename: string) => {
        try {
            const url = apiService.getKBResourceDownloadUrl(project, tenant, resourceId);
            const link = document.createElement('a');
            link.href = url;
            link.download = filename;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        } catch (err) {
            console.error("Error downloading link:", err);
        }
    };

    const handlePreview = async (resource: KBResource) => {
        try {
            setPreviewFile({
                name: resource.name,
                size: resource.size_bytes ? `${(resource.size_bytes / 1024 / 1024).toFixed(1)} MB` : 'Unknown size',
                mimeType: resource.mime || 'text/html',
                url: apiService.getKBResourceDownloadUrl(project, tenant, resource.id),
                originalUrl: resource.uri,
                resourceId: resource.id,
                version: resource.version
            });
        } catch (err) {
            console.error("Error preparing preview:", err);
        }
    };

    const handleResourceClick = (resourceId: string) => {
        setExpandedResource(expandedResource === resourceId ? null : resourceId);
    };

    const closePreview = () => {
        setPreviewFile(null);
    };

    return (
        <div className="h-full w-full relative">
            <SimpleHTMLPreview
                isOpen={!!previewFile}
                onClose={closePreview}
                file={previewFile}
                originalUrl={previewFile?.originalUrl}
            />

            {/* Tooltip for hover */}
            {/*<LinkTooltip*/}
            {/*    resource={tooltipResource}*/}
            {/*    isVisible={!!tooltipResource}*/}
            {/*    onClose={() => {*/}
            {/*        setTooltipResource(null);*/}
            {/*        setTooltipPosition(null);*/}
            {/*    }}*/}
            {/*    position={tooltipPosition}*/}
            {/*/>*/}

            <div className="mb-4">
                <h2 className="text-xl font-semibold text-gray-800 mb-3">Links & Websites</h2>
                <div className="relative">
                    <input
                        type="url"
                        placeholder="Enter URL to add to knowledge base..."
                        value={newUrl}
                        onChange={(e) => handleUrlChange(e.target.value)}
                        onKeyPress={handleKeyPress}
                        disabled={isAdding}
                        className={`w-full px-3 py-2 pr-10 border rounded focus:outline-none focus:ring-2 ${
                            !isValidUrl
                                ? 'border-red-500 focus:ring-red-500 text-red-600'
                                : 'border-gray-300 focus:ring-blue-500'
                        } ${isAdding ? 'bg-gray-100 cursor-not-allowed' : ''}`}
                    />
                    <button
                        onClick={handleAddLink}
                        disabled={!isValidUrl || !newUrl.trim() || isAdding}
                        className={`absolute right-2 top-1/2 transform -translate-y-1/2 p-1 rounded ${
                            !isValidUrl || !newUrl.trim() || isAdding
                                ? 'text-gray-300 cursor-not-allowed'
                                : 'text-gray-500 hover:text-blue-600 hover:bg-blue-50'
                        }`}
                    >
                        {isAdding ? <div
                                className="animate-spin w-4 h-4 border-2 border-blue-500 border-t-transparent rounded-full"></div> :
                            <Plus size={16}/>}
                    </button>
                </div>
                {!isValidUrl && newUrl.trim() && (
                    <p className="text-red-500 text-xs mt-1">Please enter a valid URL</p>
                )}
            </div>

            {/* Upload Progress for Links */}
            {Object.keys(uploadProgress).length > 0 && (
                <div className="mb-4 grid grid-cols-3 gap-4">
                    {Object.entries(uploadProgress).map(([linkId, progress]) => (
                        <div key={linkId} className="bg-white rounded border p-4 flex flex-col items-center">
                            <Globe size={32} className="text-blue-500 mb-2"/>
                            <div className="text-sm text-gray-700 mb-2">
                                Adding Link
                            </div>
                            <div className="w-full bg-gray-200 rounded-full h-2 mb-1">
                                <div
                                    className="bg-blue-500 h-2 rounded-full transition-all duration-300"
                                    style={{width: `${progress}%`}}
                                />
                            </div>
                            <div className="text-xs text-gray-600">{progress}%</div>
                            {uploadStages[linkId] && (
                                <div className="text-xs text-gray-500 mt-1 text-center">
                                    {uploadStages[linkId]}
                                </div>
                            )}
                        </div>
                    ))}
                </div>
            )}

            {/* Upload Errors */}
            {Object.entries(uploadErrors).length > 0 && (
                <div className="mb-4 space-y-2">
                    {Object.entries(uploadErrors).map(([linkId, error]) => (
                        <div key={linkId} className="bg-red-50 border border-red-200 rounded p-2 flex items-center">
                            <X size={16} className="text-red-500 mr-2"/>
                            <span className="text-sm text-red-700">{error}</span>
                        </div>
                    ))}
                </div>
            )}

            {/* Enhanced Link Resources List */}
            <div className="space-y-2">
                {linkResources.length === 0 ? (
                    <div className="text-center py-8 bg-gray-50 rounded-lg">
                        <Globe size={24} className="mx-auto text-gray-400 mb-2"/>
                        <p className="text-gray-600">No links added yet</p>
                        <p className="text-gray-500 text-sm mt-1">Add URLs to make them searchable in your knowledge
                            base</p>
                    </div>
                ) : (
                    linkResources.map((resource) => {
                        const displayTitle = getDisplayTitle(resource);

                        return (
                            <div key={resource.id}
                                 className="bg-white border border-gray-400 rounded-lg overflow-hidden">
                                <div
                                    className="p-3 cursor-pointer hover:shadow-sm"
                                    onClick={() => handleResourceClick(resource.id)}
                                >
                                    <div className="flex items-center justify-between">
                                        <div className="flex items-center space-x-3">
                                            <Globe className="text-blue-500" size={20}/>
                                            <div className="flex-1">
                                                {/* Enhanced title display with hover tooltip */}
                                                <div
                                                    className="font-medium text-sm text-gray-900 flex items-center mb-1 hover:text-blue-700 transition-colors"
                                                    onMouseEnter={(e) => handleMouseEnter(resource, e)}
                                                    onMouseLeave={handleMouseLeave}
                                                >
                                                    {displayTitle}
                                                    {expandedResource === resource.id ?
                                                        <ChevronUp size={16} className="ml-2 text-gray-400"/> :
                                                        <ChevronDown size={16} className="ml-2 text-gray-400"/>
                                                    }
                                                </div>

                                                {/* Show URL only if different from title */}
                                                {displayTitle !== resource.uri && (
                                                    <div className="text-xs text-gray-500 mb-1">
                                                        <a
                                                            href={resource.uri}
                                                            target="_blank"
                                                            rel="noopener noreferrer"
                                                            className="text-blue-600 hover:text-blue-800 hover:underline truncate block"
                                                            onClick={(e) => e.stopPropagation()}
                                                            title={resource.uri}
                                                        >
                                                            {resource.uri.length > 50 ?
                                                                `${resource.uri.substring(0, 47)}...` :
                                                                resource.uri
                                                            }
                                                        </a>
                                                    </div>
                                                )}

                                                {/* Show description if available */}
                                                {resource.extraction_info && resource.extraction_info[0]?.metadata?.description && (
                                                    <div className="text-xs text-gray-600 mb-1 line-clamp-2">
                                                        {resource.extraction_info[0].metadata.description}
                                                    </div>
                                                )}

                                                <div className="flex items-center space-x-2">
                                                    <span className={`inline-block w-2 h-2 rounded-full ${
                                                        resource.fully_processed ? 'bg-green-500' : 'bg-yellow-500'
                                                    }`}></span>
                                                    <span className="text-xs text-gray-600">
                                                        {resource.fully_processed ? 'Fully Processed' : 'Processing/Pending'}
                                                    </span>
                                                    <span className="text-xs text-gray-500">
                                                        v{resource.version}
                                                    </span>
                                                </div>
                                            </div>
                                        </div>
                                        <div className="flex items-center space-x-2">
                                            <button
                                                onClick={(e) => {
                                                    e.stopPropagation();
                                                    handlePreview(resource);
                                                }}
                                                className="p-1 text-gray-500 hover:text-blue-600 hover:bg-blue-50 rounded"
                                                title="Preview website"
                                            >
                                                <Eye size={14}/>
                                            </button>
                                            <button
                                                onClick={(e) => {
                                                    e.stopPropagation();
                                                    handleDownloadResource(resource.id, resource.filename);
                                                }}
                                                className="p-1 text-gray-500 hover:text-green-600 hover:bg-green-50 rounded"
                                                title="Download HTML"
                                            >
                                                <Download size={14}/>
                                            </button>
                                            <button
                                                onClick={(e) => {
                                                    e.stopPropagation();
                                                    handleDeleteResource(resource.id);
                                                }}
                                                className="p-1 text-gray-500 hover:text-red-600 hover:bg-red-50 rounded"
                                                title="Delete link"
                                            >
                                                <Trash2 size={14}/>
                                            </button>
                                        </div>
                                    </div>
                                </div>

                                {expandedResource === resource.id && (
                                    <div className="border-t border-gray-100 p-4 bg-gray-50">
                                        <div className="space-y-3">
                                            <div>
                                                <label className="block text-sm font-medium text-gray-700 mb-1">Processing
                                                    Status</label>
                                                <div className="space-y-1">
                                                    {Object.entries(resource.processing_status).map(([stage, completed]) => (
                                                        <div key={stage}
                                                             className="flex items-center space-x-2 text-sm">
                                                            {completed ? (
                                                                <Check size={14} className="text-green-500"/>
                                                            ) : (
                                                                <X size={14} className="text-gray-400"/>
                                                            )}
                                                            <span
                                                                className={completed ? 'text-green-700' : 'text-gray-500'}>
                                                            {stage.charAt(0).toUpperCase() + stage.slice(1)}
                                                        </span>
                                                        </div>
                                                    ))}
                                                </div>
                                            </div>

                                            <div className="grid grid-cols-2 gap-4 text-sm">
                                                <div>
                                                    <label className="block font-medium text-gray-700 mb-1">Source
                                                        Type</label>
                                                    <p className="capitalize">{resource.source_type}</p>
                                                </div>
                                                <div>
                                                    <label
                                                        className="block font-medium text-gray-700 mb-1">Version</label>
                                                    <p>{resource.version}</p>
                                                </div>
                                            </div>

                                            <div>
                                                <label className="block text-sm font-medium text-gray-700 mb-1">Original
                                                    URL</label>
                                                <p className="text-xs text-gray-600 bg-gray-100 p-2 rounded break-all">{resource.uri}</p>
                                            </div>
                                        </div>
                                    </div>
                                )}
                            </div>
                        );
                    })
                )}
            </div>
        </div>
    );
};

const GoogleDrivePanel = () => {
    const {googleDriveConnected, setGoogleDriveConnected} = useDemoDataContext();

    const handleConnect = () => {
        setGoogleDriveConnected?.(true);
    };

    const handleDisconnect = () => {
        setGoogleDriveConnected?.(false);
    };

    return (
        <div className="h-full w-full">
            <div className="flex items-center justify-between mb-4">
                <h2 className="text-xl font-semibold text-gray-800">Google Drive</h2>
                {googleDriveConnected ? (
                    <button
                        onClick={handleDisconnect}
                        className="flex items-center px-3 py-1 text-sm bg-red-500 text-white rounded hover:bg-red-600"
                    >
                        <CloudDownload size={14} className="mr-1"/>
                        Disconnect
                    </button>
                ) : (
                    <button
                        onClick={handleConnect}
                        className="flex items-center px-3 py-1 text-sm bg-green-500 text-white rounded hover:bg-green-600"
                    >
                        <CloudDownload size={14} className="mr-1"/>
                        Connect
                    </button>
                )}
            </div>
            {googleDriveConnected ? (
                <div className="space-y-2">
                    <div className="p-3 bg-white rounded border hover:shadow-sm">
                        <div className="font-medium text-sm text-gray-900">Project Folder</div>
                        <div className="text-xs text-gray-500">15 files • Synced 2 hours ago</div>
                    </div>
                    <div className="p-3 bg-white rounded border hover:shadow-sm">
                        <div className="font-medium text-sm text-gray-900">Documents</div>
                        <div className="text-xs text-gray-500">8 files • Synced 1 day ago</div>
                    </div>
                </div>
            ) : (
                <div className="text-center py-8">
                    <CloudDownload size={48} className="mx-auto text-gray-400 mb-4"/>
                    <p className="text-gray-600 mb-4">Connect your Google Drive to access files directly</p>
                    <button
                        onClick={handleConnect}
                        className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
                    >
                        Authorize Google Drive
                    </button>
                </div>
            )}
        </div>
    )
}

const TabName = ({icon, name}) => (
    <div className="flex items-center">
        {icon}
        <span className="ml-2">{name}</span>
    </div>
);

const KBPanel = ({onClose}: { onClose?: () => void }) => {
    const [activeTab, setActiveTab] = useState(0);

    const tabs = [
        {name: <TabName icon={<Search size={16}/>} name="Search KB"/>, content: <IntegratedEnhancedSearchPanel/>},
        {name: <TabName icon={<FileIcon size={16}/>} name="Files & Docs"/>, content: <FilesPanel/>},
        {name: <TabName icon={<Link size={16}/>} name="Links"/>, content: <LinksPanel/>},
        {name: <TabName icon={<CloudDownload size={16}/>} name="Google Drive"/>, content: <GoogleDrivePanel/>},
    ];

    return (
        <div className="relative h-full w-full bg-gray-100 flex flex-col transition-transform">
            <div className="flex items-center justify-between p-4 bg-white border-b border-gray-400">
                <h3 className="text-lg font-semibold">Knowledge Base</h3>
                {!!onClose &&
                    (<button onClick={onClose} className="p-1 hover:bg-gray-200 rounded">
                        <X size={20}/>
                    </button>)
                }
            </div>

            <div className="flex border-b border-gray-400 bg-white overflow-x-auto">
                {tabs.map((tab, i) => (
                    <Fragment key={i}>
                        <button
                            className={
                                "px-3 py-3 text-xs font-medium border-b-2 transition-colors whitespace-nowrap " +
                                (activeTab === i
                                    ? "text-blue-600 border-blue-500 bg-blue-50"
                                    : "text-gray-500 border-transparent hover:text-gray-700 hover:border-gray-300")
                            }
                            onClick={() => setActiveTab(i)}
                        >
                            {tab.name}
                        </button>
                        {i < tabs.length - 1 && <div className="self-stretch w-px bg-gray-300"/>}
                    </Fragment>
                ))}
            </div>

            <div className="flex-1 overflow-y-auto p-6">
                {tabs[activeTab].content}
            </div>
        </div>
    )
}

export default KBPanel;