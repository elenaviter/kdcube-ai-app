import {
    Archive,
    Database,
    File,
    FileAudio,
    FileCode,
    FileImage,
    FileSpreadsheet,
    FileText,
    FileVideo,
    Package,
    Presentation,
    Table
} from 'lucide-react';

// File type to icon mapping
const fileTypeIcons = {
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
const extensionIcons = {
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
export const getFileIcon = (filename: string, size = 24, mimeType?: string, className?: string) => {
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

export const fileExamples = [
    // Documents
    {
        fileName: "file.pdf",
        url: "file.pdf",
        size: 100500
    },
    {
        fileName: "report.docx",
        url: "documents/report.docx",
        size: 2456789
    },
    {
        fileName: "notes.txt",
        url: "notes.txt",
        size: 5432
    },
    {
        fileName: "presentation.pptx",
        url: "slides/presentation.pptx",
        size: 15678901
    },
    {
        fileName: "manual.doc",
        url: "docs/manual.doc",
        size: 987654
    },
    {
        fileName: "readme.md",
        url: "readme.md",
        size: 8765
    },
    {
        fileName: "contract.rtf",
        url: "legal/contract.rtf",
        size: 234567
    },

    // Code Files
    {
        fileName: "file.py",
        url: "file.py",
        size: 100500
    },
    {
        fileName: "app.js",
        url: "src/app.js",
        size: 45678
    },
    {
        fileName: "index.html",
        url: "public/index.html",
        size: 12345
    },
    {
        fileName: "styles.css",
        url: "assets/styles.css",
        size: 23456
    },
    {
        fileName: "component.jsx",
        url: "components/component.jsx",
        size: 34567
    },
    {
        fileName: "main.ts",
        url: "src/main.ts",
        size: 18765
    },
    {
        fileName: "config.json",
        url: "config.json",
        size: 4321
    },
    {
        fileName: "server.php",
        url: "backend/server.php",
        size: 56789
    },
    {
        fileName: "program.java",
        url: "src/program.java",
        size: 67890
    },
    {
        fileName: "script.cpp",
        url: "src/script.cpp",
        size: 78901
    },
    {
        fileName: "handler.go",
        url: "handlers/handler.go",
        size: 23450
    },
    {
        fileName: "lib.rs",
        url: "src/lib.rs",
        size: 34561
    },
    {
        fileName: "view.swift",
        url: "ios/view.swift",
        size: 12367
    },
    {
        fileName: "activity.kt",
        url: "android/activity.kt",
        size: 45672
    },
    {
        fileName: "component.vue",
        url: "src/component.vue",
        size: 23458
    },
    {
        fileName: "deploy.sh",
        url: "scripts/deploy.sh",
        size: 7890
    },
    {
        fileName: "docker-compose.yml",
        url: "docker-compose.yml",
        size: 5678
    },

    // Database
    {
        fileName: "file.sql",
        url: "file.sql",
        size: 100500
    },
    {
        fileName: "backup.db",
        url: "data/backup.db",
        size: 12345678
    },
    {
        fileName: "users.sqlite",
        url: "database/users.sqlite",
        size: 8765432
    },

    // Images
    {
        fileName: "photo.jpg",
        url: "images/photo.jpg",
        size: 3456789
    },
    {
        fileName: "logo.png",
        url: "assets/logo.png",
        size: 234567
    },
    {
        fileName: "icon.svg",
        url: "icons/icon.svg",
        size: 12345
    },
    {
        fileName: "banner.gif",
        url: "media/banner.gif",
        size: 5678901
    },
    {
        fileName: "background.webp",
        url: "images/background.webp",
        size: 876543
    },
    {
        fileName: "diagram.bmp",
        url: "docs/diagram.bmp",
        size: 4567890
    },
    {
        fileName: "thumbnail.ico",
        url: "assets/thumbnail.ico",
        size: 32768
    },
    {
        fileName: "chart.tiff",
        url: "reports/chart.tiff",
        size: 2345678
    },

    // Videos
    {
        fileName: "tutorial.mp4",
        url: "videos/tutorial.mp4",
        size: 89765432
    },
    {
        fileName: "demo.avi",
        url: "media/demo.avi",
        size: 123456789
    },
    {
        fileName: "presentation.mov",
        url: "recordings/presentation.mov",
        size: 67890123
    },
    {
        fileName: "webinar.webm",
        url: "streams/webinar.webm",
        size: 45678901
    },
    {
        fileName: "movie.mkv",
        url: "videos/movie.mkv",
        size: 234567890
    },
    {
        fileName: "clip.wmv",
        url: "clips/clip.wmv",
        size: 34567890
    },
    {
        fileName: "short.flv",
        url: "media/short.flv",
        size: 12345678
    },

    // Audio
    {
        fileName: "song.mp3",
        url: "music/song.mp3",
        size: 8765432
    },
    {
        fileName: "sound.wav",
        url: "audio/sound.wav",
        size: 23456789
    },
    {
        fileName: "track.flac",
        url: "music/track.flac",
        size: 45678901
    },
    {
        fileName: "podcast.m4a",
        url: "podcasts/podcast.m4a",
        size: 12345678
    },
    {
        fileName: "voice.ogg",
        url: "recordings/voice.ogg",
        size: 6789012
    },
    {
        fileName: "music.aac",
        url: "audio/music.aac",
        size: 4567890
    },
    {
        fileName: "alert.wma",
        url: "sounds/alert.wma",
        size: 123456
    },

    // Spreadsheets
    {
        fileName: "data.xlsx",
        url: "reports/data.xlsx",
        size: 3456789
    },
    {
        fileName: "budget.xls",
        url: "finance/budget.xls",
        size: 2345678
    },
    {
        fileName: "inventory.csv",
        url: "data/inventory.csv",
        size: 567890
    },
    {
        fileName: "report.ods",
        url: "documents/report.ods",
        size: 1234567
    },

    // Archives
    {
        fileName: "backup.zip",
        url: "backups/backup.zip",
        size: 87654321
    },
    {
        fileName: "files.rar",
        url: "archives/files.rar",
        size: 56789012
    },
    {
        fileName: "project.7z",
        url: "compressed/project.7z",
        size: 34567890
    },
    {
        fileName: "logs.tar.gz",
        url: "logs/logs.tar.gz",
        size: 23456789
    },
    {
        fileName: "assets.tar",
        url: "packages/assets.tar",
        size: 12345678
    },

    // Configuration Files
    {
        fileName: "settings.ini",
        url: "config/settings.ini",
        size: 5432
    },
    {
        fileName: "environment.env",
        url: ".env",
        size: 2345
    },
    {
        fileName: "config.toml",
        url: "config.toml",
        size: 7890
    },
    {
        fileName: "package.json",
        url: "package.json",
        size: 12345
    },
    {
        fileName: "composer.json",
        url: "composer.json",
        size: 8765
    },
    {
        fileName: "requirements.txt",
        url: "requirements.txt",
        size: 3456
    },
    {
        fileName: "Dockerfile",
        url: "Dockerfile",
        size: 4567
    },
    {
        fileName: "makefile",
        url: "makefile",
        size: 6789
    },

    // Other Files
    {
        fileName: "firmware.bin",
        url: "firmware/firmware.bin",
        size: 16777216
    },
    {
        fileName: "data.xml",
        url: "data/data.xml",
        size: 456789
    },
    {
        fileName: "certificate.pem",
        url: "certs/certificate.pem",
        size: 7890
    },
    {
        fileName: "key.p12",
        url: "keys/key.p12",
        size: 12345
    },
    {
        fileName: "font.ttf",
        url: "fonts/font.ttf",
        size: 234567
    },
    {
        fileName: "style.woff2",
        url: "fonts/style.woff2",
        size: 123456
    },
    {
        fileName: "template.xsl",
        url: "templates/template.xsl",
        size: 34567
    },
    {
        fileName: "schema.xsd",
        url: "schemas/schema.xsd",
        size: 23456
    },
    {
        fileName: "patch.diff",
        url: "patches/patch.diff",
        size: 8765
    },
    {
        fileName: "license.txt",
        url: "license.txt",
        size: 4321
    },
    {
        fileName: "changelog.log",
        url: "logs/changelog.log",
        size: 15432
    },
    {
        fileName: "unknown.xyz",
        url: "misc/unknown.xyz",
        size: 9876
    }
];

export default fileExamples;