import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";
import rehypeSanitize, {defaultSchema} from "rehype-sanitize";
import type {Components} from "react-markdown";
import remarkBreaks from "remark-breaks";
import rehypeRaw from "rehype-raw";
import MermaidDiagram from "../../MermaidDiagram.tsx";
import 'highlight.js/styles/a11y-light.min.css'

const mdSanitizeSchema = {
    ...defaultSchema,
    attributes: {
        ...defaultSchema.attributes,
        // keep language-* class on <code> for highlight.js
        code: [
            ...(defaultSchema.attributes?.code || []),
            ['className', 'language-*']
        ],
        a: [
            ...(defaultSchema.attributes?.a || []),
            ['target', 'rel']
        ],
        img: [
            ...(defaultSchema.attributes?.img || []),
            ['loading', 'width', 'height']
        ],
        input: [
            ...(defaultSchema.attributes?.input || []),
            ['type', 'checked', 'disabled']
        ],
        span: [
            ...(defaultSchema.attributes?.span || []),
            ['className']
        ],
    },
};

export const remarkPlugins = [remarkGfm, remarkBreaks /*, remarkMath*/];
export const rehypePlugins = [
    rehypeRaw,
    [rehypeSanitize, mdSanitizeSchema],
    rehypeHighlight,
    /* rehypeKatex */
] as any;

const codeRender = ({inline, className, children, ...props}) => {
    const languageName = /language-(\w+)/.exec(className || '')?.[1];

    if (!inline && languageName === 'mermaid') {
        return (<MermaidDiagram chart={String(children).replace(/\n$/, '')}/>)
    }

    return inline ? (
        <code className="text-sm px-1 bg-gray-100 rounded font-mono">{children}</code>
    ) : (
        <pre className="hljs p-0 rounded-sm overflow-x-auto mt-4 mb-1 last:mb-1">
            {languageName &&
                <div className="px-2 py-1 bg-gray-200 align-middle text-[0.7rem]">
                    <span className="text-gray-800">{languageName}</span>
                </div>
            }
            <code className={"text-sm " + (className ? className : "hljs")}>{children}</code>
        </pre>
    )
}

export const markdownComponents: Components = {
    code({inline, className, children, ...props}) {
        return codeRender({inline, className, children, ...props} as any);
    },
    a({children, ...props}) {
        return (
            <a className="text-blue-600 underline" target="_blank" rel="noreferrer" {...props}>
        {children}
        </a>
    );
    },
    p({children}) {
        return <p className="mt-3 mb-2 last:mb-1 leading-7">{children}</p>;
    },
    hr() {
        return <hr className="my-6 border-t border-gray-400"/>;
    },
    blockquote({children}) {
        return (<blockquote
            className="mt-3 mb-2 last:mb-1 border-l-4 border-gray-300 pl-4 italic text-gray-700">{children}</blockquote>);
    },
    ul({children}) {
        return <ul className="mt-3 mb-2 last:mb-1 list-disc pl-6 space-y-1">{children}</ul>;
    },
    ol({children}) {
        return <ol className="mt-3 mb-2 last:mb-1 list-decimal pl-6 space-y-1">{children}</ol>;
    },
    li({children, checked}) {
        // Task list item
        if (checked !== null && checked !== undefined) {
            return (
                <li className="my-1 flex items-start gap-2">
                <input type="checkbox" checked={!!checked} readOnly className="mt-1 h-4 w-4"/>
            <span className="flex-1">{children}</span>
                </li>
        );
        }
        return <li className="my-1">{children}</li>;
    },
    table({children}) {
        return (<div className="mt-4 mb-1 last:mb-1 overflow-x-auto">
        <table className="min-w-full border-collapse">{children}</table>
            </div>);
    },
    thead({children}) {
        return <thead className="bg-gray-100">{children}</thead>;
    },
    th({children}) {
        return <th className="text-left py-2 px-3 border-b font-medium">{children}</th>;
    },
    td({children}) {
        return <td className="py-2 px-3 border-b align-top">{children}</td>;
    },
    img({src, alt, title}) {
        const s = src || "";
        return (
            <a href={s} target="_blank" rel="noreferrer">
        <img src={s} alt={alt || ""} title={title} loading="lazy" className="my-2 rounded max-w-full h-auto"/>
            </a>
    );
    },

    // Proper heading tags + consistent spacing
    h1({children, ...props}) {
        return <h1 className="text-2xl font-bold text-gray-900 mt-4 mb-3" {...props}>{children}</h1>;
    },
    h2({children, ...props}) {
        return <h2 className="text-xl font-semibold text-gray-800 mt-4 mb-3" {...props}>{children}</h2>;
    },
    h3({children, ...props}) {
        return <h3 className="text-lg font-semibold text-gray-700 mt-3 mb-2" {...props}>{children}</h3>;
    },
    h4({children, ...props}) {
        return <h4 className="text-base font-medium text-gray-700 mt-3 mb-2" {...props}>{children}</h4>;
    },
    h5({children, ...props}) {
        return <h5 className="text-base font-medium text-gray-600 mt-2 mb-1" {...props}>{children}</h5>;
    },
    h6({children, ...props}) {
        return <h6 className="text-sm font-medium text-gray-600 mt-2 mb-1" {...props}>{children}</h6>;
    },

    // Inline formatting niceties (optional)
    em({children}) {
        return <em className="italic">{children}</em>;
    },
    strong({children}) {
        return <strong className="font-semibold">{children}</strong>;
    },
    del({children}) {
        return <del className="line-through">{children}</del>;
    },
};

export const markdownComponentsTight: Components = {
    ...markdownComponents,
    p({children}) {
        return <p className="my-1 last:mb-0 leading-[1.4]">{children}</p>;
    },
    ul({children}) {
        return <ul className="mt-1 mb-1 last:mb-0 list-disc pl-6 space-y-0.5">{children}</ul>;
    },
    ol({children}) {
        return <ol className="mt-1 mb-1 last:mb-0 list-decimal pl-6 space-y-0.5">{children}</ol>;
    },
    li({children, checked}) {
        if (checked !== null && checked !== undefined) {
            return (
                <li className="my-0.5 leading-[1.4] flex items-start gap-2">
                <input type="checkbox" checked={!!checked} readOnly className="mt-0.5 h-4 w-4"/>
            <span className="flex-1">{children}</span>
                </li>
        );
        }
        return <li className="my-0.5 leading-[1.4]">{children}</li>;
    },
    blockquote({children}) {
        return (
            <blockquote
                className="mt-1 mb-1 last:mb-0 border-l-4 border-gray-300 pl-4 italic text-gray-700 leading-[1.4]">
                {children}
                </blockquote>
        );
    },
};
