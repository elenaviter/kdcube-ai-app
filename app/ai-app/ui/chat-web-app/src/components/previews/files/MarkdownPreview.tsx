/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2025 Elena Viter
 */

import {useState} from "react";
import {Eye, Code, Copy} from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";
import {FileLoading, FileLoadingError, FilesPreviewProps} from "./Shared.tsx";
import ErrorBoundary from "../../ErrorBoundary.tsx";
import 'highlight.js/styles/atom-one-dark.min.css'

const MarkdownRenderError = () => {
    return (
        <div className="bg-red-50 border border-red-200 rounded-lg p-6 m-4">
            <div className="flex items-center mb-4">
                <div className="bg-red-100 p-2 rounded-full mr-3">
                    <svg className="w-6 h-6 text-red-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                              d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.732-.833-2.5 0L4.268 19.5c-.77.833.192 2.5 1.732 2.5z"/>
                    </svg>
                </div>
                <h2 className="text-lg font-semibold text-red-800">We're sorry, but we are unable to render this
                    Markdown file</h2>
            </div>
            <p className="text-red-700 mb-2">You can still view the source</p>

        </div>
    );
}

const MarkdownRenderer = (
    {viewMode, fontSize, content}:
    { viewMode: 'preview' | 'source', fontSize: number, content: string | null }
) => {
    const extractLanguageNameFromClass = (className?: string | null) => {
        if (className) {
            const classes = className.split(" ");
            const languageKey = "language-"
            for (const className of classes) {
                if (className.startsWith(languageKey))
                    return className.substring(languageKey.length).toUpperCase();
            }
        }
        return null
    }
    return (
        <div className="flex-1 overflow-auto bg-white">
            {viewMode === 'preview' ? (
                <ErrorBoundary fallback={
                    <MarkdownRenderError/>
                }>
                    <div
                        className="p-6 prose prose-slate max-w-none"
                        style={{fontSize: `${fontSize}px`}}
                    >
                        <ReactMarkdown
                            remarkPlugins={[remarkGfm]}
                            rehypePlugins={[rehypeHighlight]}
                            skipHtml={true}
                            remarkRehypeOptions={
                                {
                                    allowDangerousHtml: false
                                }
                            }
                            components={{
                                h1({node, className, children, ...props}) {
                                    return <h1
                                        className="text-4xl font-bold text-gray-900 mb-4" {...props}>{children}</h1>
                                },
                                h2({node, className, children, ...props}) {
                                    return <h1
                                        className="text-3xl font-semibold text-gray-800 mb-3" {...props}>{children}</h1>
                                },
                                h3({node, className, children, ...props}) {
                                    return <h1
                                        className="text-2xl font-semibold text-gray-700 mb-2" {...props}>{children}</h1>
                                },
                                h4({node, className, children, ...props}) {
                                    return <h1
                                        className="text-xl font-medium text-gray-600 mb-2" {...props}>{children}</h1>
                                },
                                h5({node, className, children, ...props}) {
                                    return <h1 className="text-lg font-medium text-gray-600" {...props}>{children}</h1>
                                },
                                h6({node, className, children, ...props}) {
                                    return <h1
                                        className="text-base font-medium text-gray-600" {...props}>{children}</h1>
                                },
                                code({node, inline, className, children, ...props}) {
                                    const languageName = extractLanguageNameFromClass(className)
                                    return inline ? (
                                        <code className="text-sm px-1 bg-gray-100 rounded">{children}</code>
                                    ) : (
                                        <pre className="hljs p-0 rounded-sm overflow-x-auto my-4">
                                            {languageName &&
                                                <div className="px-3 py-2 bg-gray-700 align-middle">
                                                    <span className="text-white">{languageName}</span>
                                                </div>
                                            }
                                            <code
                                                className={"text-sm text-gray-100 " + (className ? className : "hljs")}>{children}</code>
                                        </pre>
                                    )
                                },
                                a({node, children, href, ...props}) {
                                    return (
                                        <a
                                            href={href}
                                            target="_blank"
                                            rel="noopener noreferrer"
                                            className="text-blue-600 hover:text-blue-800 underline"
                                            {...props}
                                        >
                                            {children}
                                        </a>
                                    )
                                },
                                img({node, src, alt, ...props}) {
                                    return (
                                        <img
                                            src={src}
                                            alt={alt || ''}
                                            className="max-w-full h-auto rounded"
                                            {...props}
                                        />
                                    )
                                },
                                table({node, children, ...props}) {
                                    return (
                                        <div className="overflow-x-auto my-4">
                                            <table
                                                className="min-w-full border-collapse border border-gray-300" {...props}>
                                                {children}
                                            </table>
                                        </div>
                                    )
                                },
                                th({node, children, ...props}) {
                                    return (
                                        <th className="border border-gray-300 bg-gray-100 px-4 py-2 text-left">
                                            {children}
                                        </th>
                                    )
                                },
                                td({node, children, ...props}) {
                                    return (
                                        <td className="border border-gray-300 px-4 py-2">
                                            {children}
                                        </td>
                                    )
                                }
                            }}
                        >
                            {content || ''}
                        </ReactMarkdown>
                    </div>
                </ErrorBoundary>
            ) : (
                <div className="p-6">
                        <pre
                            className="font-mono text-gray-800 whitespace-pre-wrap bg-gray-50 p-4 rounded border border-gray-200 h-full"
                            style={{fontSize: `${fontSize}px`}}
                        >
                            {content || ''}
                        </pre>
                </div>
            )}
        </div>
    )
}

const MarkdownPreview = ({content, loading, error}: FilesPreviewProps) => {
    const [viewMode, setViewMode] = useState<'preview' | 'source'>('preview');
    const [fontSize, setFontSize] = useState(14);

    const copyToClipboard = () => {
        if (content) {
            navigator.clipboard.writeText(content)
                .then(() => {
                    // Show a temporary success message
                    const copyButton = document.getElementById('copy-button');
                    if (copyButton) {
                        const originalText = copyButton.innerHTML;
                        copyButton.innerHTML = `<span class="text-green-500">Copied!</span>`;
                        setTimeout(() => {
                            copyButton.innerHTML = originalText;
                        }, 2000);
                    }
                })
                .catch(err => {
                    console.error('Failed to copy text: ', err);
                });
        }
    };

    if (loading) {
        return <FileLoading/>
    }

    if (error) {
        return <FileLoadingError error={error}/>;
    }

    return (
        <div className="flex-1 min-h-0 h-full flex flex-col">
            {/* Markdown Toolbar */}
            <div className="flex items-center justify-between p-3 bg-gray-100 border-b">
                <div className="flex items-center space-x-2">
                    <button
                        onClick={() => setViewMode('preview')}
                        className={`px-3 py-1 text-sm rounded flex items-center ${
                            viewMode === 'preview'
                                ? 'bg-blue-100 text-blue-700 border border-blue-300'
                                : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                        }`}
                    >
                        <Eye size={14} className="mr-1"/>
                        Preview
                    </button>
                    <button
                        onClick={() => setViewMode('source')}
                        className={`px-3 py-1 text-sm rounded flex items-center ${
                            viewMode === 'source'
                                ? 'bg-blue-100 text-blue-700 border border-blue-300'
                                : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                        }`}
                    >
                        <Code size={14} className="mr-1"/>
                        Source
                    </button>

                    <div className="h-6 w-px bg-gray-300 mx-1"></div>

                    <span className="text-sm text-gray-600">Font size:</span>
                    <button
                        onClick={() => setFontSize(Math.max(10, fontSize - 1))}
                        className="px-2 py-1 text-sm bg-gray-200 text-gray-700 rounded hover:bg-gray-300"
                    >
                        A-
                    </button>
                    <span className="text-sm">{fontSize}px</span>
                    <button
                        onClick={() => setFontSize(Math.min(24, fontSize + 1))}
                        className="px-2 py-1 text-sm bg-gray-200 text-gray-700 rounded hover:bg-gray-300"
                    >
                        A+
                    </button>
                </div>

                <div className="flex items-center space-x-2">
                    <button
                        id="copy-button"
                        onClick={copyToClipboard}
                        className="flex items-center px-3 py-1 text-sm bg-gray-200 text-gray-700 rounded hover:bg-gray-300"
                    >
                        <Copy size={14} className="mr-1"/>
                        Copy
                    </button>
                </div>
            </div>

            {/* Markdown Content */}
            {<MarkdownRenderer content={content} viewMode={viewMode} fontSize={fontSize}/>}
        </div>
    );
};

export default MarkdownPreview;