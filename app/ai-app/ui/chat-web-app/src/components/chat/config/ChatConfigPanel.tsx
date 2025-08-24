/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2025 Elena Viter
 */

import React, {useCallback, useEffect, useMemo, useState} from 'react';
import {
    AlertCircle, BookOpen, Database, Download, Play, RotateCcw, Upload, X, Loader
} from 'lucide-react';
import {useBundles} from '../hooks/useBundles';
import {BundleInfo, EmbedderInfo, EmbeddingProvider, ModelInfo} from '../types/chat';
import {getChatBaseAddress, getKBAPIBaseAddress} from '../../../AppConfig';
import { BundlesListAdmin } from "./BundlesListAdmin.tsx";
import ReactMarkdown from "react-markdown";
import className = ReactMarkdown.propTypes.className;
import {cn} from "../../../utils/utils.ts";

const server_url = `${getChatBaseAddress()}`;
const serving_server_url = 'http://localhost:5005/serving/v1';

type Props = {
    visible: boolean;
    onClose: () => void;
    authContext: any;

    // persisted config
    config: any;
    setConfigValue: (k: string, v: any) => void;
    updateConfig: (patch: Record<string, any>) => void;

    validationErrors: string[];

    // lets Chat header show selected names
    onMetaChange?: (meta: { model?: ModelInfo; embedder?: EmbedderInfo; bundle?: BundleInfo }) => void;

    // enable admin UI (you decide who is super-admin on the caller)
    canAdminBundles?: boolean;
    className?: string;
};

function needsModule(path?: string) {
    if (!path) return false;
    return /\.whl$/i.test(path) || /\.zip$/i.test(path);
}

export const ChatConfigPanel: React.FC<Props> = ({
                                                     visible, onClose, authContext,
                                                     config, setConfigValue, updateConfig,
                                                     validationErrors, onMetaChange,
                                                     canAdminBundles,
                                                 }) => {
    const [availableModels, setAvailableModels] = useState<Record<string, ModelInfo>>({});
    const [availableEmbedders, setAvailableEmbedders] = useState<Record<string, EmbedderInfo>>({});
    const [embeddingProviders, setEmbeddingProviders] = useState<Record<string, EmbeddingProvider>>({});

    const {
        bundles,
        defaultId,
        loading: bundlesLoading,
        error: bundlesError,
        reload: reloadBundles
    } = useBundles(server_url, authContext);

    const selectedModelInfo = availableModels[config.selected_model] as ModelInfo || {} as ModelInfo;
    const selectedEmbedderInfo = availableEmbedders[config.selected_embedder] as EmbedderInfo || {} as EmbedderInfo;
    const selectedBundle = config.agentic_bundle_id ? bundles[config.agentic_bundle_id] : undefined;

    // propagate header meta upward
    useEffect(() => {
        onMetaChange?.({model: selectedModelInfo, embedder: selectedEmbedderInfo, bundle: selectedBundle});
    }, [config.selected_model, config.selected_embedder, config.agentic_bundle_id, availableModels, availableEmbedders, bundles]);

    // default bundle for config (first load)
    useEffect(() => {
        if (!config.agentic_bundle_id && defaultId) {
            setConfigValue('agentic_bundle_id', defaultId);
        }
    }, [defaultId]);

    // Load models & embedders (read-only)
    useEffect(() => {
        (async () => {
            try {
                const headers: HeadersInit = [['Content-Type', 'application/json']];
                authContext.appendAuthHeader(headers);
                const modelsRes = await fetch(`${server_url}/landing/models`, {headers});
                const modelsData = await modelsRes.json();
                if (modelsRes.ok) setAvailableModels(modelsData.available_models || {});
                const embRes = await fetch(`${server_url}/landing/embedders`, {headers});
                const embData = await embRes.json();
                if (embRes.ok) {
                    setAvailableEmbedders(embData.available_embedders || {});
                    setEmbeddingProviders(embData.providers || {});
                    if (!config.selected_embedder && embData.default_embedder) {
                        setConfigValue('selected_embedder', embData.default_embedder);
                    }
                }
            } catch {
                // fall back silently; Chat header will still show something
            }
        })();
    }, []);

    const requiresCustomEndpoint = selectedEmbedderInfo?.provider === 'custom';

    // Export/import personal chat config (unchanged)
    const exportConfig = useCallback(() => {
        const blob = new Blob([JSON.stringify(config, null, 2)], {type: 'application/json'});
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a'); a.href = url; a.download = `ai-assistant-config-${new Date().toISOString().split('T')[0]}.json`;
        document.body.appendChild(a); a.click(); document.body.removeChild(a); URL.revokeObjectURL(url);
    }, [config]);

    const importConfig = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0]; if (!file) return;
        const r = new FileReader();
        r.onload = ev => {
            try {
                const s = ev.target?.result; if (typeof s === 'string') {
                    const obj = JSON.parse(s);
                    updateConfig(obj);
                    alert('Configuration imported.');
                }
            } catch { alert('Invalid configuration file.'); }
        };
        r.readAsText(file);
        e.target.value = '';
    }, [updateConfig]);

    const testEmbeddings = useCallback(async () => {
        if (selectedEmbedderInfo.provider === 'custom' && !config.custom_embedding_endpoint) {
            alert('Please enter a custom embedding endpoint'); return;
        }
        if (selectedEmbedderInfo.provider === 'openai' && !config.openai_api_key) {
            alert('Please enter your OpenAI API key'); return;
        }
        try {
            const headers: HeadersInit = [['Content-Type', 'application/json']]; authContext.appendAuthHeader(headers);
            const res = await fetch(`${server_url}/landing/test-embeddings`, {method: 'POST', headers, body: JSON.stringify(config)});
            const data = await res.json();
            if (res.ok) alert(`✅ Embeddings OK\nEmbedder: ${data.embedder_id}\nModel: ${data.model}\nDim: ${data.embedding_size}`);
            else alert(`❌ Failed:\n${data?.detail?.error || 'Unknown error'}`);
        } catch (e: any) { alert(`❌ Failed:\n${e.message}`); }
    }, [config, selectedEmbedderInfo, authContext]);

    // -------------------
    // Admin: Bundles CRUD
    // -------------------
    const [adminBusy, setAdminBusy] = useState(false);
    const [adminMsg, setAdminMsg] = useState<string | null>(null);
    const [adminErr, setAdminErr] = useState<string | null>(null);

    // local edit buffer (by id)
    type EditRow = { id: string; name?: string; path: string; module?: string; singleton?: boolean; description?: string };
    const [editRows, setEditRows] = useState<Record<string, EditRow>>({});
    const [adding, setAdding] = useState<EditRow | null>(null);

    useEffect(() => {
        // Initialize editRows from bundles (id is immutable)
        const buf: Record<string, EditRow> = {};
        Object.entries(bundles).forEach(([id, b]) => {
            buf[id] = { id, name: b.name, path: b.path, module: b.module, singleton: !!b.singleton, description: b.description };
        });
        setEditRows(buf);
    }, [bundles]);

    const adminHeaders = useMemo(() => {
        const headers: HeadersInit = [['Content-Type', 'application/json']];
        authContext.appendAuthHeader(headers);
        return headers;
    }, [authContext]);

    const postAdmin = useCallback(async (body: any) => {
        setAdminBusy(true); setAdminErr(null); setAdminMsg(null);
        try {
            const res = await fetch(`${server_url}/admin/integrations/bundles`, {method: 'POST', headers: adminHeaders, body: JSON.stringify(body)});
            const data = await res.json().catch(()=> ({}));
            if (!res.ok) {
                if (res.status === 403) throw new Error('Forbidden: you need super-admin privileges to manage bundles.');
                throw new Error(data?.detail || 'Failed to apply bundle changes');
            }
            setAdminMsg('Changes applied successfully.');
            await reloadBundles();
        } catch (e:any) {
            setAdminErr(e.message || 'Failed to apply changes');
        } finally {
            setAdminBusy(false);
        }
    }, [adminHeaders, reloadBundles]);

    const validateRow = (row: EditRow): string | null => {
        if (!row.id?.trim()) return 'Bundle ID is required';
        if (!row.path?.trim()) return 'Path is required';
        if (needsModule(row.path) && !row.module?.trim()) return 'Module is required for .whl/.zip bundles';
        return null;
    };

    const saveRow = async (id: string) => {
        const row = editRows[id];
        const err = validateRow(row); if (err) { setAdminErr(err); return; }
        await postAdmin({
            op: 'merge',
            bundles: { [id]: { id, name: row.name, path: row.path, module: row.module || undefined, singleton: !!row.singleton, description: row.description } }
        });
    };

    const removeRow = async (id: string) => {
        // Build replacement registry without this id
        const next: Record<string, any> = {};
        Object.entries(bundles).forEach(([bid, b]) => {
            if (bid === id) return;
            next[bid] = { id: b.id, name: b.name, path: b.path, module: b.module, singleton: !!b.singleton, description: b.description };
        });
        await postAdmin({ op: 'replace', bundles: next, default_bundle_id: defaultId && next[defaultId] ? defaultId : Object.keys(next)[0] });
    };

    const setDefault = async (id: string) => {
        await postAdmin({ op: 'merge', bundles: {}, default_bundle_id: id });
    };

    const saveNew = async () => {
        if (!adding) return;
        const err = validateRow(adding); if (err) { setAdminErr(err); return; }
        await postAdmin({
            op: 'merge',
            bundles: { [adding.id]: { ...adding, module: adding.module || undefined } }
        });
        setAdding(null);
    };

    const resetFromEnv = useCallback(async () => {
        if (!window.confirm('This will overwrite the current mapping from server .env. Continue?')) return;
        setAdminBusy(true); setAdminErr(null); setAdminMsg(null);
        try {
            const headers: HeadersInit = [['Content-Type', 'application/json']]; authContext.appendAuthHeader(headers);
            const res = await fetch(`${server_url}/admin/integrations/bundles/reset-from-env`, { method: 'POST', headers });
            const data = await res.json().catch(()=> ({}));
            if (!res.ok) throw new Error(data?.detail || 'Failed to reset from .env');
            setAdminMsg('Mapping was reset from .env and broadcast.');
            await reloadBundles();
        } catch (e:any) {
            setAdminErr(e.message || 'Failed to reset mapping');
        } finally {
            setAdminBusy(false);
        }
    }, [authContext, reloadBundles]);


    if (!visible) return null;

    return (
        <div className={cn(
            "bg-white border-r border-gray-200 p-6 overflow-y-auto w-[min(48vw,640px)] min-w-[360px]",
            className
        )}>
            <div className="flex items-center justify-between mb-6">
                <h2 className="text-lg font-semibold text-gray-900">Configuration</h2>
                <button onClick={onClose} className="p-2 hover:bg-gray-100 rounded-lg"><X size={16}/></button>
            </div>

            {/* Validation */}
            {validationErrors?.length > 0 && (
                <div className="border border-red-200 bg-red-50 rounded-lg p-3 mb-3">
                    <h4 className="text-sm font-medium text-red-800 mb-2">Configuration Issues:</h4>
                    <ul className="text-xs text-red-700 space-y-1">
                        {validationErrors.map((err: string, idx: number) => (
                            <li key={idx} className="flex items-start"><AlertCircle size={12} className="mr-1 mt-0.5"/>{err}</li>
                        ))}
                    </ul>
                </div>
            )}

            {/* Bundle Selection */}
            <div className="border-b pb-4 mb-4">
                <label className="block text-sm font-medium text-gray-700 mb-2">Agentic App Bundle</label>
                <select
                    value={config.agentic_bundle_id || ''}
                    onChange={(e) => setConfigValue('agentic_bundle_id', e.target.value)}
                    className="w-full p-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                >
                    {Object.entries(bundles).map(([id, b]) => (
                        <option key={id} value={id}>{b.name || id}</option>
                    ))}
                </select>
                <div className="mt-2 p-2 bg-gray-50 rounded text-xs">
                    {bundlesLoading && <div className="flex items-center"><Loader size={14} className="animate-spin mr-2"/>Loading bundles…</div>}
                    {bundlesError && <div className="text-red-600">Error: {bundlesError}</div>}
                    {selectedBundle && (
                        <>
                            <div><strong>Name:</strong> {selectedBundle.name || selectedBundle.id}</div>
                            <div><strong>Path:</strong> {selectedBundle.path}</div>
                            {selectedBundle.module && <div><strong>Module:</strong> {selectedBundle.module}</div>}
                            <div><strong>Singleton:</strong> {selectedBundle.singleton ? 'Yes' : 'No'}</div>
                        </>
                    )}
                </div>
            </div>

            {/* Model Selection */}
            <div className="border-b pb-4 mb-4">
                <label className="block text-sm font-medium text-gray-700 mb-2">AI Assistant Model</label>
                <select
                    value={config.selected_model}
                    onChange={(e) => setConfigValue('selected_model', e.target.value)}
                    className="w-full p-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                >
                    {Object.entries(availableModels).map(([id, info]) => (
                        <option key={id} value={id}>{info.description}</option>
                    ))}
                </select>
                <div className="mt-2 p-2 bg-gray-50 rounded text-xs">
                    <div><strong>Provider:</strong> {selectedModelInfo.provider || 'Unknown'}</div>
                    <div><strong>Classification:</strong> {selectedModelInfo.has_classifier ? 'Yes' : 'No'}</div>
                </div>
            </div>

            {/* Keys */}
            <div className="border-b pb-4 mb-4">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                    OpenAI API Key {selectedModelInfo.provider === 'openai' && <span className="text-red-500">*</span>}
                </label>
                <input
                    type="password"
                    value={config.openai_api_key || ''}
                    onChange={(e) => setConfigValue('openai_api_key', e.target.value)}
                    placeholder="sk-..."
                    className="w-full p-2 border border-gray-300 rounded-lg"
                />
            </div>

            {/* KB */}
            <div className="border-b pb-4 mb-4">
                <h3 className="text-sm font-medium text-gray-700 mb-3 flex items-center"><BookOpen size={16} className="mr-2"/>Knowledge Base</h3>
                <label className="block text-sm font-medium text-gray-700 mb-2">KB Search Endpoint</label>
                <input
                    type="url"
                    value={config.kb_search_endpoint || ''}
                    onChange={(e) => setConfigValue('kb_search_endpoint', e.target.value)}
                    placeholder={`${getKBAPIBaseAddress()}/api/kb`}
                    className="w-full p-2 border border-gray-300 rounded-lg mb-3"
                />
            </div>

            {/* Embeddings */}
            <div>
                <h3 className="text-sm font-medium text-gray-700 mb-3 flex items-center"><Database size={16} className="mr-2"/>Embeddings</h3>
                <label className="block text-sm font-medium text-gray-700 mb-2">Embedding Model</label>
                <select
                    value={config.selected_embedder}
                    onChange={(e) => {
                        const id = e.target.value;
                        const next = availableEmbedders[id] || ({} as EmbedderInfo);
                        updateConfig({
                            selected_embedder: id,
                            custom_embedding_endpoint: next.provider === 'openai' ? '' : (config.custom_embedding_endpoint || `${serving_server_url}/landing/embeddings`)
                        });
                    }}
                    className="w-full p-2 border border-gray-300 rounded-lg"
                >
                    {Object.entries(availableEmbedders).map(([id, info]) => (
                        <option key={id} value={id}>{info.description}</option>
                    ))}
                </select>
                <div className="mt-2 p-2 bg-gray-50 rounded text-xs space-y-1">
                    <div><strong>Provider:</strong> {selectedEmbedderInfo?.provider || 'Unknown'}</div>
                    <div><strong>Model:</strong> {selectedEmbedderInfo?.model || 'Unknown'}</div>
                    <div><strong>Dimensions:</strong> {selectedEmbedderInfo?.dimension || 'Unknown'}</div>
                </div>

                {selectedEmbedderInfo?.provider === 'custom' && (
                    <div className="mt-3">
                        <label className="block text-sm font-medium text-gray-700 mb-2">Custom Embedding Endpoint <span className="text-red-500">*</span></label>
                        <div className="flex gap-2">
                            <input
                                type="url"
                                value={config.custom_embedding_endpoint || ''}
                                onChange={(e) => setConfigValue('custom_embedding_endpoint', e.target.value)}
                                placeholder="http://localhost:5005/serving/v1/embeddings"
                                className="flex-1 p-2 border border-gray-300 rounded-lg"
                            />
                            <button
                                onClick={testEmbeddings}
                                disabled={!config.custom_embedding_endpoint}
                                className={`px-3 py-2 rounded-lg text-sm ${config.custom_embedding_endpoint ? 'bg-blue-500 text-white hover:bg-blue-600' : 'bg-gray-300 text-gray-500'}`}
                                title="Test embedding endpoint"
                            >
                                <Play size={14}/>
                            </button>
                        </div>
                    </div>
                )}

                {selectedEmbedderInfo?.provider === 'openai' && (
                    <div className="mt-3">
                        <button
                            onClick={testEmbeddings}
                            disabled={!config.openai_api_key}
                            className={`w-full px-3 py-2 rounded-lg text-sm ${config.openai_api_key ? 'bg-green-500 text-white hover:bg-green-600' : 'bg-gray-300 text-gray-500'}`}
                        >
                            <Play size={14} className="inline mr-2"/>Test OpenAI Embeddings
                        </button>
                    </div>
                )}
            </div>

            {/* Config mgmt */}
            <div className="border-t mt-4 pt-4">
                <h3 className="text-sm font-medium text-gray-700 mb-3">Config Management</h3>
                <div className="flex gap-2">
                    <button onClick={exportConfig} className="flex-1 flex items-center justify-center px-3 py-2 bg-blue-100 text-blue-700 rounded-lg hover:bg-blue-200 text-sm"><Download size={14} className="mr-1"/>Export</button>
                    <label className="flex-1 flex items-center justify-center px-3 py-2 bg-green-100 text-green-700 rounded-lg hover:bg-green-200 text-sm cursor-pointer">
                        <Upload size={14} className="mr-1"/>Import
                        <input type="file" accept=".json" onChange={importConfig} className="hidden"/>
                    </label>
                    <button onClick={() => updateConfig({})} className="flex items-center justify-center px-3 py-2 bg-red-100 text-red-700 rounded-lg hover:bg-red-200 text-sm" title="Reset to defaults">
                        <RotateCcw size={14}/>
                    </button>
                </div>
            </div>

            {/* ----------------------- */}
            {/* Admin: Manage Bundles   */}
            {/* ----------------------- */}
            <div className="border-t mt-5 pt-4">
                <div className="flex items-center justify-between mb-2">
                    <h3 className="text-sm font-semibold text-gray-800">Admin: Manage Bundles</h3>
                    {(bundlesLoading || adminBusy) && (
                        <div className="text-xs text-gray-500">Applying…</div>
                    )}
                </div>

                {adminMsg && <div className="mb-2 text-xs text-green-700 bg-green-50 border border-green-200 p-2 rounded">{adminMsg}</div>}
                {adminErr && <div className="mb-2 text-xs text-red-700 bg-red-50 border border-red-200 p-2 rounded">{adminErr}</div>}
                {bundlesError && <div className="mb-2 text-xs text-red-700 bg-red-50 border border-red-200 p-2 rounded">{bundlesError}</div>}

                <BundlesListAdmin
                    bundles={bundles}
                    defaultId={defaultId || undefined}
                    loading={bundlesLoading || adminBusy}
                    onReload={reloadBundles}
                    onResetFromEnv={resetFromEnv}   // keep the toolbar button in the list
                    onSetDefault={(id) => setDefault(id)}
                    onDelete={async (id) => {
                        const next: Record<string, any> = {};
                        Object.entries(bundles).forEach(([bid, b]) => {
                            if (bid !== id) next[bid] = {
                                id: b.id, name: b.name, path: b.path, module: b.module,
                                singleton: !!b.singleton, description: b.description
                            };
                        });
                        await postAdmin({
                            op: 'replace',
                            bundles: next,
                            default_bundle_id: (defaultId && next[defaultId]) ? defaultId : Object.keys(next)[0]
                        });
                    }}
                    onSave={async (b) => {
                        if (/.whl$|.zip$/i.test(b.path) && !b.module?.trim()) {
                            setAdminErr('Module is required for .whl/.zip bundles');
                            return;
                        }
                        await postAdmin({
                            op: 'merge',
                            bundles: {
                                [b.id]: {
                                    id: b.id,
                                    name: b.name,
                                    path: b.path,
                                    module: b.module || undefined,
                                    singleton: !!b.singleton,
                                    description: b.description
                                }
                            }
                        });
                    }}
                />
            </div>

        </div>
    );
};
