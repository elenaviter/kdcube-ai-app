import {useEffect, useState} from "react";
import {useAuthManagerContext} from "../auth/AuthManager.tsx";


export const handleDownload = (url: string) => {
    try {
        window.open(url, '_blank');
    } catch (error) {
        console.error('Download error:', error);
    }
}

export const handleContentDownload = (fileName: string, content: string | Blob | MediaSource, mimeType: string = 'plain/text') => {
    const contentBlob = typeof content === "string" ? new Blob([content], {type: mimeType}) : content;
    const url = URL.createObjectURL(contentBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = fileName;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
}

export function getErrorMessage(error: unknown) {
    if (error instanceof Error) return error.message
    return String(error)
}

export interface FileRef {
    name: string;
    size: string;
    mimeType: string;
    url?: string;
    resourceId?: string; // KB resource ID
    version: string;
}

export const useFileContent = (file: FileRef | null) => {
    const [content, setContent] = useState<string | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const authContext = useAuthManagerContext()

    useEffect(() => {
        const fetchFileContent = async () => {
            setError(null);
            setContent(null);
            setLoading(false);

            if (!file?.url) return;

            setLoading(true);


            try {
                const headers: HeadersInit = [
                    ['Content-Type', 'application/json']
                ];

                authContext.appendAuthHeader(headers);
                const response = await fetch(file.url, {
                    headers,
                });


                // Handle different file types
                if (file.mimeType.startsWith("text/") ||
                    file.mimeType === 'application/json' ||
                    file.mimeType === 'application/yaml' ||
                    file.mimeType === 'application/x-yaml') {

                    const text = await response.text();
                    setContent(text);
                } else {
                    // For other file types that might need binary data
                    const blob = await response.blob();
                    setContent(URL.createObjectURL(blob));
                }
            } catch (err) {
                console.error('Error fetching file:', err);
                setError(getErrorMessage(err));
            } finally {
                setLoading(false);
            }
        };

        fetchFileContent();

        // Cleanup function to revoke object URLs
        return () => {
            if (content && typeof content === 'string' && content.startsWith('blob:')) {
                URL.revokeObjectURL(content);
            }
        };
    }, [file]);

    return {content, loading, error};
};
