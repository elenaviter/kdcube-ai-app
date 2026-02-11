export const cleanupCode = (code: string) => {
    // ```(?=[^`]*$)(.|\n)* somehow matches last ```
    // don't ask me how
    return code.trim().replace(/^```.*\n/g, "").replace(/```(?=[^`]*$)(.|\n)*/g, "");
}
export const appendCodeMarkdown = (code: string, language: string) => {
    return `\`\`\` ${language}\n${code}\`\`\``
}