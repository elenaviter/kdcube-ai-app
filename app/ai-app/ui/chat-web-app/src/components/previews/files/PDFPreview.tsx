import {useState} from "react";
import {FileLoading, FileLoadingError, FilesPreviewProps} from "./Shared.tsx";

const PDFPreview = ({content, loading, error}: FilesPreviewProps) => {
    if (loading) {
        return <FileLoading/>
    }

    if (error) {
        return <FileLoadingError error={error}/>;
    }

    if (!content) return null;

    return (
        <div className="h-full flex flex-col">
            {/* PDF Content */}
            <div className="flex-1 bg-gray-200 flex items-center justify-center overflow-hidden">
                <div
                    className="w-full h-full bg-white shadow-lg overflow-hidden"
                    style={{
                        transformOrigin: 'center center',
                    }}
                >
                    <iframe
                        src={content}
                        className="w-full h-full border-0"
                        title={`PDF Preview`}
                        style={{
                            minHeight: '600px'
                        }}
                        onError={(e) => {
                            console.error('Error loading PDF:', e);
                        }}
                    />
                </div>
            </div>
        </div>
    );
};

export default PDFPreview