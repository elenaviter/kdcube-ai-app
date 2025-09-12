function fallbackCopy(text: string) {
    const textarea = document.createElement('textarea');
    textarea.value = text;
    textarea.style.position = 'fixed';
    textarea.style.opacity = '0';
    document.body.appendChild(textarea);
    textarea.select();

    const success = document.execCommand('copy');
    document.body.removeChild(textarea);
    return success;
}

async function copyMarkdownToClipboard(mdContent?: string, htmlContent?: string) {
    // Try modern API first
    if (navigator.clipboard && window.isSecureContext) {
        try {
            const clipboardItemData:Record<string, string> = {};
            if (mdContent) {
                clipboardItemData["text/plain"] = mdContent;
            }
            if (htmlContent) {
                clipboardItemData["text/html"] = htmlContent;
            }
            const clipboardItem = [new ClipboardItem(clipboardItemData)]
            await navigator.clipboard.write(clipboardItem)
            return true;
        } catch (error) {
            console.error('Modern clipboard failed:', error);
        }
    }

    // Fallback for older browsers
    return fallbackCopy(mdContent.toString());
}

export {copyMarkdownToClipboard}