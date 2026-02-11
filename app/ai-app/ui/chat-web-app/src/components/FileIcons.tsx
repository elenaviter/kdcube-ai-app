import {
    Archive,
    Database,
    File,
    FileAudio,
    FileCode,
    FileImage,
    FileSpreadsheet,
    FileText,
    FileVideo, LucideProps,
    Package,
    Presentation,
    Table
} from 'lucide-react';
import {ExoticComponent} from "react";

// File type to icon mapping
const fileTypeIcons: Record<string, ExoticComponent<LucideProps>> = {
    // Documents
    'application/pdf': FileText,
    'application/msword': FileText,
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document': FileText,
    'text/plain': FileText,
    'text/rtf': FileText,
    'application/rtf': FileText,

    // Spreadsheets
    'application/vnd.ms-excel': FileSpreadsheet,
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': FileSpreadsheet,
    'text/csv': Table,

    // Presentations
    'application/vnd.ms-powerpoint': Presentation,
    'application/vnd.openxmlformats-officedocument.presentationml.presentation': Presentation,

    // Images
    'image/jpeg': FileImage,
    'image/jpg': FileImage,
    'image/png': FileImage,
    'image/gif': FileImage,
    'image/webp': FileImage,
    'image/svg+xml': FileImage,
    'image/bmp': FileImage,
    'image/tiff': FileImage,

    // Videos
    'video/mp4': FileVideo,
    'video/avi': FileVideo,
    'video/mov': FileVideo,
    'video/wmv': FileVideo,
    'video/flv': FileVideo,
    'video/webm': FileVideo,
    'video/mkv': FileVideo,

    // Audio
    'audio/mpeg': FileAudio,
    'audio/mp3': FileAudio,
    'audio/wav': FileAudio,
    'audio/ogg': FileAudio,
    'audio/aac': FileAudio,
    'audio/flac': FileAudio,

    // Archives
    'application/zip': Archive,
    'application/x-rar-compressed': Archive,
    'application/x-7z-compressed': Archive,
    'application/x-tar': Archive,
    'application/gzip': Archive,

    // Code files
    'text/html': FileCode,
    'text/css': FileCode,
    'text/javascript': FileCode,
    'application/javascript': FileCode,
    'application/json': FileCode,
    'text/xml': FileCode,
    'application/xml': FileCode,

    // Database
    'application/x-sqlite3': Database,
    'application/vnd.ms-access': Database,

    // Other
    'application/octet-stream': Package,
};

// Extension to icon mapping (fallback)
const extensionIcons: Record<string, ExoticComponent<LucideProps>> = {
    // Documents
    'pdf': FileText,
    'doc': FileText,
    'docx': FileText,
    'txt': FileText,
    'rtf': FileText,
    'odt': FileText,

    // Spreadsheets
    'xls': FileSpreadsheet,
    'xlsx': FileSpreadsheet,
    'csv': Table,
    'ods': FileSpreadsheet,

    // Presentations
    'ppt': Presentation,
    'pptx': Presentation,
    'odp': Presentation,

    // Images
    'jpg': FileImage,
    'jpeg': FileImage,
    'png': FileImage,
    'gif': FileImage,
    'webp': FileImage,
    'svg': FileImage,
    'bmp': FileImage,
    'tiff': FileImage,
    'ico': FileImage,

    // Videos
    'mp4': FileVideo,
    'avi': FileVideo,
    'mov': FileVideo,
    'wmv': FileVideo,
    'flv': FileVideo,
    'webm': FileVideo,
    'mkv': FileVideo,
    'm4v': FileVideo,

    // Audio
    'mp3': FileAudio,
    'wav': FileAudio,
    'ogg': FileAudio,
    'aac': FileAudio,
    'flac': FileAudio,
    'wma': FileAudio,
    'm4a': FileAudio,

    // Archives
    'zip': Archive,
    'rar': Archive,
    '7z': Archive,
    'tar': Archive,
    'gz': Archive,
    'bz2': Archive,

    // Code files
    'html': FileCode,
    'css': FileCode,
    'js': FileCode,
    'jsx': FileCode,
    'ts': FileCode,
    'tsx': FileCode,
    'json': FileCode,
    'xml': FileCode,
    'php': FileCode,
    'py': FileCode,
    'java': FileCode,
    'cpp': FileCode,
    'c': FileCode,
    'h': FileCode,
    'cs': FileCode,
    'rb': FileCode,
    'go': FileCode,
    'rs': FileCode,
    'swift': FileCode,
    'kt': FileCode,
    'dart': FileCode,
    'vue': FileCode,
    'svelte': FileCode,
    'yml': FileCode,
    'yaml': FileCode,
    'toml': FileCode,
    'ini': FileCode,
    'conf': FileCode,
    'sh': FileCode,
    'bat': FileCode,
    'ps1': FileCode,

    // Database
    'sql': Database,
    'db': Database,
    'sqlite': Database,
    'mdb': Database,
};

// Utility function to get icon for file
export const getFileIcon = (filename: string, size = 24, mimeType?: string | null, className?: string ) => {
    // Try MIME type first
    if (mimeType && fileTypeIcons[mimeType]) {
        const IconComponent = fileTypeIcons[mimeType];
        return <IconComponent size={size}/>;
    }

    // Fallback to extension
    if (filename) {
        const extension = filename.split('.').pop()?.toLowerCase();
        if (extension && extensionIcons[extension]) {
            const IconComponent = extensionIcons[extension];
            return <IconComponent size={size} className={className}/>;
        }
    }

    // Default icon
    return <File size={size}/>;
};

export const getFileIconClass = (filename: string, mimeType?: string | null) => {
    if (mimeType && fileTypeIcons[mimeType]) {
        return  fileTypeIcons[mimeType];
    }

    if (filename) {
        const extension = filename.split('.').pop()?.toLowerCase();
        if (extension && extensionIcons[extension]) {
            return  extensionIcons[extension];
        }
    }

    return null;
};