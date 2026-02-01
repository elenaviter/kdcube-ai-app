
import {ChatMessageData, StepUpdate} from "../../../features/chat/chatTypes.ts";

function getExampleAssistantMessage(id = 10): ChatMessageData {
    return {
        id: id,
        sender: "assistant",
        text: `# Welcome to My AI Assistant

Hello! I'm your **AI assistant application** and currently under *active development*.

## Features

Here are some things I can help you with:

- Answer questions about various topics
- Help with **coding** and technical problems
- Assist with \`markdown\` formatting
- Provide writing assistance

### Code Example

\`\`\`javascript
function greet(name) {
    return \`Hello, \${name}!\`;
}
\`\`\`

> **Note:** This is still a work in progress, so please be patient as we continue to improve!

For more information, visit [our documentation](https://example.com/docs).

---

*Thanks for trying out the assistant!*`,
        timestamp: new Date(id),
        metadata: {
            turn_id: `example_turn_${id}`
        },
    }
}

// if (s.step === "file" && s.status === "completed" && !!s.data?.rn && !!s.data?.filename) {
//     addItem(createDownloadItem(s))
// } else if (s.step === "citations" && s.status === "completed" && !!s.data?.count && !!s.data?.items) {
//     addItem(createSourceLinks(s))
// }

function getExampleAssistantFileSteps(id: number = 10, amount: number = 10): StepUpdate[] {
    const result = []
    for (let i = 0; i < amount; i++) {
        result.push({
            step: "file",
            status: "completed",
            timestamp: new Date(id),
            turn_id: `example_turn_${id}`,
            data: {
                rn: `example_rn_${id}_${i}`,
                filename: `example_file_${id}_${i}.txt`
            }
        } as StepUpdate)
    }
    return result;
}

function getExampleAssistantSourceSteps(id: number = 10, amount: number = 2, linksAmount: number = 3): StepUpdate[] {
    const result = []
    let t = 1
    const getVariant = (i: number) => {
        const normalized = i - Math.floor(i / 3) * 3
        switch (normalized) {
            case 0:
                return {
                    url: `https://example.com/docs/${t}`,
                    title: `Example link #${t}`,
                }
            case 1:
                return {
                    url: `https://example.com/very_long_very_long_very_long_very_long_very_long_very_long_very_long_very_long_very_long_very_long_very_long_very_long_very_long_very_long_very_long_very_long/docs/${t}`,
                    title: `Example link #${t}`,
                }
            case 2:
                return {
                    url: `https://example.com/docs/${t}`,
                }
        }
    }
    for (let i = 0; i < amount; i++) {
        const links = []
        for (let j = 0; j < linksAmount; j++) {
            links.push(getVariant(j))
            t++
        }
        result.push({
            step: "citations",
            status: "completed",
            timestamp: new Date(id),
            turn_id: `example_turn_${id}`,
            data: {
                count: amount,
                items: links
            }
        } as StepUpdate)
    }
    return result;
}

export {getExampleAssistantMessage, getExampleAssistantFileSteps, getExampleAssistantSourceSteps};