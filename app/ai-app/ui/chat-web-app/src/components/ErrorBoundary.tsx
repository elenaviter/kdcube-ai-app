/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2025 Elena Viter
 */

import {Component, ErrorInfo, ReactNode} from "react";

interface ErrorBoundaryProps {
    children?: ReactNode | ReactNode[];
    fallback?: ReactNode | ReactNode[];
}

interface ErrorBoundaryState {
    hasError: false,
    error?: Error | null,
    errorInfo?: ErrorInfo | null
}

class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {

    constructor(props:ErrorBoundaryProps) {
        super(props);
        this.state = { hasError: false, error: null, errorInfo: null };
    }

    static getDerivedStateFromError() {
        // Update state to show fallback UI
        return { hasError: true };
    }

    componentDidCatch(error: Error, errorInfo: ErrorInfo) {
        console.error('Error caught by boundary:', error, errorInfo);
        this.setState({
            error: error,
            errorInfo: errorInfo
        });
    }

    render() {
        if (this.state.hasError) {
            return this.props.fallback || (
                <div className="bg-red-50 border border-red-200 rounded-lg p-6 m-4">
                    <div className="flex items-center mb-4">
                        <div className="bg-red-100 p-2 rounded-full mr-3">
                            <svg className="w-6 h-6 text-red-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.732-.833-2.5 0L4.268 19.5c-.77.833.192 2.5 1.732 2.5z" />
                            </svg>
                        </div>
                        <h2 className="text-lg font-semibold text-red-800">Something went wrong</h2>
                    </div>
                    <p className="text-red-700 mb-4">We're sorry, but something unexpected happened.</p>
                    <button
                        onClick={() => this.setState({ hasError: false, error: null, errorInfo: null })}
                        className="bg-red-600 text-white px-4 py-2 rounded hover:bg-red-700 mr-2"
                    >
                        Try Again
                    </button>
                    <button
                        onClick={() => window.location.reload()}
                        className="bg-gray-600 text-white px-4 py-2 rounded hover:bg-gray-700"
                    >
                        Reload Page
                    </button>

                    {/* Show error details in development */}
                    {process.env.NODE_ENV === 'development' && (
                        <details className="mt-4">
                            <summary className="cursor-pointer text-red-800 font-medium">Error Details (Development Only)</summary>
                            <pre className="mt-2 text-sm bg-red-100 p-2 rounded overflow-auto">
                {this.state.error && this.state.error.toString()}
                                <br />
                                {this.state.errorInfo?.componentStack}
              </pre>
                        </details>
                    )}
                </div>
            );
        }

        return this.props.children;
    }
}

export default ErrorBoundary;