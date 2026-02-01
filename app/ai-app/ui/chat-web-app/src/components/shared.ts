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

// Simple word-aware truncation
function truncateWords(str: string, length: number, ellipsis: string = '...'): string {
    if (str.length <= length) return str;

    const truncated = str.slice(0, length - ellipsis.length);
    const lastSpaceIndex = truncated.lastIndexOf(' ');

    // If no space found or space is too close to the beginning, just truncate normally
    if (lastSpaceIndex === -1 || lastSpaceIndex < length * 0.5) {
        return truncated + ellipsis;
    }

    return truncated.slice(0, lastSpaceIndex) + ellipsis;
}

// Truncate by number of words instead of characters
function truncateByWords(str: string, wordCount: number, ellipsis: string = '...'): string {
    const words = str.trim().split(/\s+/);

    if (words.length <= wordCount) return str;

    return words.slice(0, wordCount).join(' ') + ellipsis;
}

// Smart truncation that tries to preserve meaning
function smartTruncate(str: string, length: number, ellipsis: string = '...'): string {
    if (str.length <= length) return str;

    // Try to truncate at sentence boundaries first
    const sentences = str.split(/[.!?]+/);
    let result = '';

    for (const sentence of sentences) {
        const potential = result + (result ? '. ' : '') + sentence.trim();
        if (potential.length + ellipsis.length <= length) {
            result = potential;
        } else {
            break;
        }
    }

    // If we got a good sentence-based truncation, use it
    if (result.length > length * 0.6) {
        return result + (result.endsWith('.') ? '' : '.') + ellipsis;
    }

    // Otherwise fall back to word-aware truncation
    return truncateWords(str, length, ellipsis);
}

function openUrlSafely(url: string): boolean {
    try {
        const newWindow = window.open(url, '_blank', 'noopener,noreferrer');
        return newWindow !== null;
    } catch (error) {
        console.error('Failed to open URL:', error);
        return false;
    }
}

interface FileSelectionOptions {
    accept?: string;
    multiple?: boolean;
    maxSize?: number; // in bytes
    maxFiles?: number;
}

function selectFile(accept?: string, multiple?: boolean): Promise<FileList | null> {
    return new Promise((resolve) => {
        const input = document.createElement('input');
        input.type = 'file';

        if (accept) {
            input.accept = accept;
        }

        if (multiple) {
            input.multiple = true;
        }

        input.onchange = () => {
            resolve(input.files);
        };

        input.oncancel = () => {
            resolve(null);
        };

        input.click();
    });
}

async function selectFileAdvanced(options: FileSelectionOptions = {}): Promise<File[]> {
    const {
        accept,
        multiple = false,
        maxSize,
        maxFiles
    } = options;

    const files = await selectFile(accept, multiple);

    if (!files || files.length === 0) {
        return [];
    }

    let fileArray = Array.from(files);

    // Apply file count limit
    if (maxFiles && fileArray.length > maxFiles) {
        alert(`Maximum ${maxFiles} files allowed`);
        fileArray = fileArray.slice(0, maxFiles);
    }

    // Apply size limit
    if (maxSize) {
        const oversizedFiles = fileArray.filter(file => file.size > maxSize);
        if (oversizedFiles.length > 0) {
            alert(`Some files exceed the ${(maxSize / 1024 / 1024).toFixed(1)}MB limit`);
            fileArray = fileArray.filter(file => file.size <= maxSize);
        }
    }

    return fileArray;
}

export {truncateWords, truncateByWords, smartTruncate, openUrlSafely, selectFile, selectFileAdvanced};