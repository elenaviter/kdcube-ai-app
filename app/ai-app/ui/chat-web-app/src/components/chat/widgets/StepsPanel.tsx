/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2025 Elena Viter
 */

import React, {useCallback, useEffect, useRef, useState} from 'react';
import {AlertCircle, ChevronRight, Clock, Database, FileText, GripVertical, Loader, MessageSquare, Play, Search, CheckCircle2, X} from 'lucide-react';
import {StepUpdate} from '../types/chat';

const useResizable = (initial=320,min=250,max=500)=>{
    const [width,setWidth] = useState(initial);
    const [isResizing,setIsResizing] = useState(false);
    const startX = useRef(0); const startW = useRef(initial);
    const md = useCallback((e:React.MouseEvent)=>{e.preventDefault(); setIsResizing(true); startX.current=e.clientX; startW.current=width;},[width]);
    const mm = useCallback((e:MouseEvent)=>{ if(!isResizing) return; const dx=startX.current-e.clientX; setWidth(Math.min(max,Math.max(min,startW.current+dx)));},[isResizing,min,max]);
    const mu = useCallback(()=>setIsResizing(false),[]);
    useEffect(()=>{ if(!isResizing) return; document.addEventListener('mousemove',mm); document.addEventListener('mouseup',mu);
        document.body.style.cursor='col-resize'; document.body.style.userSelect='none';
        return ()=>{ document.removeEventListener('mousemove',mm); document.removeEventListener('mouseup',mu);
            document.body.style.cursor=''; document.body.style.userSelect=''; }; },[isResizing,mm,mu]);
    return {width,md};
};

export const StepsPanel: React.FC<{
    visible: boolean;
    steps: StepUpdate[];
    onClose: ()=>void;
    isProcessing: boolean;
}> = ({visible, steps, onClose, isProcessing})=>{
    const {width, md} = useResizable(320,250,500);
    if(!visible) return null;

    const icon = (name:string)=>{
        switch(name){
            case 'classifier': return <Loader size={16} className="mr-2"/>;
            case 'query_writer': return <FileText size={16}/>;
            case 'rag_retrieval': return <Database size={16}/>;
            case 'reranking': return <Search size={16}/>;
            case 'answer_generator': return <MessageSquare size={16}/>;
            case 'workflow_start': return <Play size={16}/>;
            case 'workflow_complete': return <CheckCircle2 size={16}/>;
            default: return <Clock size={16}/>;
        }
    };
    const color = (s:string)=> s==='completed' ? 'text-green-600 bg-green-50 border-green-200'
        : s==='started' ? 'text-blue-600 bg-blue-50 border-blue-200'
            : s==='error' ? 'text-red-600 bg-red-50 border-red-200' : 'text-gray-600 bg-gray-50 border-gray-400';

    const human = (s:string)=>({
        classifier:'Domain Classification',
        query_writer:'Query Generation',
        rag_retrieval:'Document Retrieval',
        reranking:'Document Reranking',
        answer_generator:'Answer Generation',
        workflow_start:'Starting Workflow',
        workflow_complete:'Workflow Complete',
        workflow_error:'Workflow Error'
    } as any)[s] || s.replace('_',' ').replace(/\b\w/g,l=>l.toUpperCase());

    return (
        <div className="bg-white border-l border-gray-400 flex flex-col relative" style={{width}}>
            <div className="absolute left-0 top-0 bottom-0 w-1 cursor-col-resize hover:bg-blue-300 group" onMouseDown={md}>
                <div className="absolute left-0 top-1/2 transform -translate-y-1/2 -translate-x-1 opacity-0 group-hover:opacity-100">
                    <GripVertical size={16} className="text-gray-400"/>
                </div>
            </div>
            <div className="px-4 py-3 border-b border-gray-400 bg-gray-50 flex items-center justify-between">
                <div>
                    <h3 className="font-semibold text-gray-900 text-sm">Execution Steps</h3>
                    <p className="text-xs text-gray-500 mt-1">
                        {steps.length>0 ? 'Real-time processing steps' : 'Steps will appear here during processing'}
                    </p>
                </div>
                <button onClick={onClose} className="p-1 hover:bg-gray-200 rounded text-gray-500 hover:text-gray-700"><X size={14}/></button>
            </div>

            <div className="flex-1 overflow-y-auto p-4">
                {steps.length===0 && !isProcessing && (
                    <div className="text-center text-gray-500 py-8">
                        <Clock size={24} className="mx-auto mb-2 opacity-50"/>
                        <p className="text-sm">No active processing</p>
                    </div>
                )}
                <div className="space-y-3">
                    {steps.map((step, i)=>(
                        <div key={`${step.step}-${i}`} className={`border rounded-lg p-3 ${color(step.status)}`}>
                            <div className="flex items-center justify-between mb-2">
                                <div className="flex items-center">
                                    {step.status==='started'? <Loader size={16} className="animate-spin mr-2"/> :
                                        step.status==='error'? <AlertCircle size={16} className="mr-2"/> : <div className="mr-2">{icon(step.step)}</div>}
                                    <span className="font-medium text-sm">{human(step.step)}</span>
                                </div>
                                {step.elapsed_time && <span className="text-xs opacity-75">{step.elapsed_time}</span>}
                            </div>
                            {step.error && (<div className="text-xs mb-2 p-2 bg-red-100 rounded border-l-2 border-red-400"><strong>Error:</strong> {step.error}</div>)}
                            {step.data && Object.keys(step.data).length>0 && (
                                <div className="text-xs space-y-1">
                                    {step.data.message && (<div><strong>Message:</strong> {step.data.message}</div>)}
                                    {step.data.model && (<div><strong>Model:</strong> {step.data.model}</div>)}
                                    {step.data.embedding_type && (<div><strong>Embeddings:</strong> {step.data.embedding_type}</div>)}
                                    {step.data.query_count && (<div><strong>Queries:</strong> {step.data.query_count}</div>)}
                                    {step.data.retrieved_count && (<div><strong>Documents:</strong> {step.data.retrieved_count}</div>)}
                                    {step.data.answer_length && (<div><strong>Answer:</strong> {step.data.answer_length} chars</div>)}
                                    {Array.isArray(step.data.queries)&&step.data.queries.length>0 && (
                                        <div><strong>Queries:</strong><ul className="ml-2 mt-1">
                                            {step.data.queries.map((q:string,idx:number)=>(
                                                <li key={idx} className="flex items-start">
                                                    <ChevronRight size={12} className="mt-0.5 mr-1 flex-shrink-0"/><span>{q}</span>
                                                </li>
                                            ))}
                                        </ul></div>
                                    )}
                                </div>
                            )}
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
};
