/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2025 Elena Viter
 */

import React, {useCallback, useEffect, useState} from 'react';
import {AlertCircle, BookOpen, Database, Download, Play, RotateCcw, Upload, X} from 'lucide-react';
import {useBundles} from '../hooks/useBundles';
import {ModelInfo, EmbedderInfo, EmbeddingProvider, BundleInfo} from '../types/chat';
import {getChatBaseAddress, getKBAPIBaseAddress} from '../../../AppConfig';

const server_url = `${getChatBaseAddress()}/landing`;
const serving_server_url = 'http://localhost:5005/serving/v1';

type Props = {
    visible: boolean;
    onClose: () => void;
    authContext: any;
    config: any;                          // your persisted config object
    setConfigValue: (k:string, v:any)=>void;
    updateConfig: (patch:Record<string,any>)=>void;
    validationErrors: string[];
    onMetaChange?: (meta: {model?: ModelInfo; embedder?: EmbedderInfo; bundle?: BundleInfo})=>void;
};

export const ChatConfigPanel: React.FC<Props> = ({
                                                     visible, onClose, authContext, config, setConfigValue, updateConfig, validationErrors, onMetaChange
                                                 })=>{
    const [availableModels, setAvailableModels] = useState<Record<string, ModelInfo>>({});
    const [availableEmbedders, setAvailableEmbedders] = useState<Record<string, EmbedderInfo>>({});
    const [embeddingProviders, setEmbeddingProviders] = useState<Record<string, EmbeddingProvider>>({});
    const {bundles, defaultId, loading: bundlesLoading, error: bundlesError, reload: reloadBundles} =
        useBundles(server_url, authContext);

    const selectedModelInfo = availableModels[config.selected_model] as ModelInfo || {} as ModelInfo;
    const selectedEmbedderInfo = availableEmbedders[config.selected_embedder] as EmbedderInfo || {} as EmbedderInfo;
    const selectedBundle = config.agentic_bundle_id ? bundles[config.agentic_bundle_id] : undefined;

    useEffect(()=>{
        onMetaChange?.({model: selectedModelInfo, embedder: selectedEmbedderInfo, bundle: selectedBundle});
    }, [config.selected_model, config.selected_embedder, config.agentic_bundle_id,
        availableModels, availableEmbedders, bundles]);

    // Ensure a default bundle is selected the first time
    useEffect(()=>{
        if (!config.agentic_bundle_id && defaultId) {
            setConfigValue('agentic_bundle_id', defaultId);
        }
    }, [defaultId]);

    // Load models & embedders (moved here from Chat)
    useEffect(()=>{
        (async ()=>{
            try {
                const headers: HeadersInit = [['Content-Type','application/json']];
                authContext.appendAuthHeader(headers);
                const modelsRes = await fetch(`${server_url}/models`, {headers});
                const modelsData = await modelsRes.json();
                if (modelsRes.ok) setAvailableModels(modelsData.available_models || {});
                const embRes = await fetch(`${server_url}/embedders`, {headers});
                const embData = await embRes.json();
                if (embRes.ok){
                    setAvailableEmbedders(embData.available_embedders || {});
                    setEmbeddingProviders(embData.providers || {});
                    if (!config.selected_embedder && embData.default_embedder) {
                        setConfigValue('selected_embedder', embData.default_embedder);
                    }
                }
            } catch(e){ /* fallback handled by Chat header if needed */ }
        })();
    }, []);

    const requiresCustomEndpoint = selectedEmbedderInfo?.provider === 'custom';

    const exportConfig = useCallback(()=>{
        const toSave = {...config};
        const blob = new Blob([JSON.stringify(toSave,null,2)], {type:'application/json'});
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a'); a.href=url; a.download=`ai-assistant-config-${new Date().toISOString().split('T')[0]}.json`;
        document.body.appendChild(a); a.click(); document.body.removeChild(a); URL.revokeObjectURL(url);
    }, [config]);

    const importConfig = useCallback((e: React.ChangeEvent<HTMLInputElement>)=>{
        const file = e.target.files?.[0]; if(!file) return;
        const r = new FileReader(); r.onload = ev => {
            try {
                const s = ev.target?.result; if (typeof s === 'string') {
                    const obj = JSON.parse(s);
                    // NOTE: use your ConfigProvider.import if needed; here we just patch
                    updateConfig(obj);
                    alert('Configuration imported.');
                }
            } catch { alert('Invalid configuration file.'); }
        }; r.readAsText(file); e.target.value='';
    }, [updateConfig]);

    const testEmbeddings = useCallback(async ()=>{
        if (selectedEmbedderInfo.provider === 'custom' && !config.custom_embedding_endpoint) {
            alert('Please enter a custom embedding endpoint'); return;
        }
        if (selectedEmbedderInfo.provider === 'openai' && !config.openai_api_key) {
            alert('Please enter your OpenAI API key'); return;
        }
        try {
            const headers: HeadersInit = [['Content-Type','application/json']]; authContext.appendAuthHeader(headers);
            const res = await fetch(`${server_url}/test-embeddings`, {method:'POST', headers, body: JSON.stringify(config)});
            const data = await res.json();
            if (res.ok) alert(`✅ Embeddings OK\nEmbedder: ${data.embedder_id}\nModel: ${data.model}\nDim: ${data.embedding_size}`);
            else alert(`❌ Failed:\n${data?.detail?.error || 'Unknown error'}`);
        } catch(e:any){ alert(`❌ Failed:\n${e.message}`); }
    }, [config, selectedEmbedderInfo, authContext]);

    if(!visible) return null;

    return (
        <div className="w-80 bg-white border-r border-gray-200 p-6 overflow-y-auto">
            <div className="flex items-center justify-between mb-6">
                <h2 className="text-lg font-semibold text-gray-900">Configuration</h2>
                <button onClick={onClose} className="p-2 hover:bg-gray-100 rounded-lg"><X size={16}/></button>
            </div>

            {/* Validation */}
            {validationErrors?.length>0 && (
                <div className="border border-red-200 bg-red-50 rounded-lg p-3 mb-3">
                    <h4 className="text-sm font-medium text-red-800 mb-2">Configuration Issues:</h4>
                    <ul className="text-xs text-red-700 space-y-1">
                        {validationErrors.map((err:string,idx:number)=>(
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
                    onChange={(e)=> setConfigValue('agentic_bundle_id', e.target.value)}
                    className="w-full p-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                >
                    {Object.entries(bundles).map(([id, b])=>(
                        <option key={id} value={id}>{b.name || id}</option>
                    ))}
                </select>
                <div className="mt-2 p-2 bg-gray-50 rounded text-xs">
                    {bundlesLoading && <div>Loading bundles…</div>}
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
                    onChange={(e)=> setConfigValue('selected_model', e.target.value)}
                    className="w-full p-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                >
                    {Object.entries(availableModels).map(([id, info])=>(<option key={id} value={id}>{info.description}</option>))}
                </select>
                <div className="mt-2 p-2 bg-gray-50 rounded text-xs">
                    <div><strong>Provider:</strong> {selectedModelInfo.provider || 'Unknown'}</div>
                    <div><strong>Classification:</strong> {selectedModelInfo.has_classifier ? 'Yes' : 'No'}</div>
                </div>
            </div>

            {/* Keys */}
            <div className="border-b pb-4 mb-4">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                    OpenAI API Key {selectedModelInfo.provider==='openai' && <span className="text-red-500">*</span>}
                </label>
                <input type="password" value={config.openai_api_key || ''} onChange={e=>setConfigValue('openai_api_key', e.target.value)}
                       placeholder="sk-..." className="w-full p-2 border border-gray-300 rounded-lg"/>
            </div>

            {/* KB */}
            <div className="border-b pb-4 mb-4">
                <h3 className="text-sm font-medium text-gray-700 mb-3 flex items-center"><BookOpen size={16} className="mr-2"/>Knowledge Base</h3>
                <label className="block text-sm font-medium text-gray-700 mb-2">KB Search Endpoint</label>
                <input type="url" value={config.kb_search_endpoint || ''} onChange={e=>setConfigValue('kb_search_endpoint', e.target.value)}
                       placeholder={`${getKBAPIBaseAddress()}/api/kb`} className="w-full p-2 border border-gray-300 rounded-lg mb-3"/>
            </div>

            {/* Embeddings */}
            <div>
                <h3 className="text-sm font-medium text-gray-700 mb-3 flex items-center"><Database size={16} className="mr-2"/>Embeddings</h3>
                <label className="block text-sm font-medium text-gray-700 mb-2">Embedding Model</label>
                <select
                    value={config.selected_embedder}
                    onChange={(e)=>{
                        const id = e.target.value;
                        const next = availableEmbedders[id] || ({} as EmbedderInfo);
                        updateConfig({
                            selected_embedder: id,
                            custom_embedding_endpoint: next.provider === 'openai' ? '' : (config.custom_embedding_endpoint || `${serving_server_url}/embeddings`)
                        });
                    }}
                    className="w-full p-2 border border-gray-300 rounded-lg"
                >
                    {Object.entries(availableEmbedders).map(([id,info])=>(
                        <option key={id} value={id}>{info.description}</option>
                    ))}
                </select>

                <div className="mt-2 p-2 bg-gray-50 rounded text-xs space-y-1">
                    <div><strong>Provider:</strong> {selectedEmbedderInfo?.provider || 'Unknown'}</div>
                    <div><strong>Model:</strong> {selectedEmbedderInfo?.model || 'Unknown'}</div>
                    <div><strong>Dimensions:</strong> {selectedEmbedderInfo?.dimension || 'Unknown'}</div>
                </div>

                {requiresCustomEndpoint && (
                    <div className="mt-3">
                        <label className="block text-sm font-medium text-gray-700 mb-2">Custom Embedding Endpoint <span className="text-red-500">*</span></label>
                        <div className="flex gap-2">
                            <input type="url" value={config.custom_embedding_endpoint || ''} onChange={e=>setConfigValue('custom_embedding_endpoint', e.target.value)}
                                   placeholder="http://localhost:5005/serving/v1/embeddings" className="flex-1 p-2 border border-gray-300 rounded-lg"/>
                            <button onClick={testEmbeddings} disabled={!config.custom_embedding_endpoint}
                                    className={`px-3 py-2 rounded-lg text-sm ${config.custom_embedding_endpoint ? 'bg-blue-500 text-white hover:bg-blue-600' : 'bg-gray-300 text-gray-500'}`}>
                                <Play size={14}/>
                            </button>
                        </div>
                    </div>
                )}

                {selectedEmbedderInfo?.provider==='openai' && (
                    <div className="mt-3">
                        <button onClick={testEmbeddings} disabled={!config.openai_api_key}
                                className={`w-full px-3 py-2 rounded-lg text-sm ${config.openai_api_key ? 'bg-green-500 text-white hover:bg-green-600' : 'bg-gray-300 text-gray-500'}`}>
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
                    <button onClick={()=>updateConfig({})} className="flex items-center justify-center px-3 py-2 bg-red-100 text-red-700 rounded-lg hover:bg-red-200 text-sm" title="Reset to defaults">
                        <RotateCcw size={14}/>
                    </button>
                </div>
            </div>
        </div>
    );
};
