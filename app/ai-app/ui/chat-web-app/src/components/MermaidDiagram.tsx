import mermaid from 'mermaid';
import {useEffect, useRef, useState} from 'react';
import {Loader2} from "lucide-react";

mermaid.initialize({
    startOnLoad: false,
    securityLevel: 'strict',  // safer; change to 'loose' if you need HTML labels
    theme: "default",
    fontFamily: 'Inter, ui-sans-serif, system-ui',
    suppressErrorRendering: true
});

const svgCache = new Map<string, string>();

function MermaidDiagram({chart}: { chart: string }) {
    const [svg, setSvg] = useState<string>(svgCache.get(chart) || "");
    const [rendering, setRendering] = useState(false);

    const idRef = useRef(`mmd-${Math.random().toString(36).slice(2)}`);

    useEffect(() => {
        if (svg) {
            console.log(`Using cached SVG (${idRef.current})`)
        } else {
            setRendering(true);
            mermaid.render(idRef.current, chart).then((svg) => {
                svgCache.set(chart, svg.svg)
                setSvg(svg.svg)
                console.log(`MermaidDiagram rendered (${idRef.current})`);
            }).catch((e) => {
                console.error(e);
            }).finally(() => setRendering(false));
        }
    }, [chart, svg]);

    // useEffect(() => {
    //     return () => {
    //         console.log("cleanup");
    //         svgCache.delete(chart);
    //     }
    // });

    if (rendering) {
        return <Loader2 className="animate-spin"/>
    }
    return <div dangerouslySetInnerHTML={{__html: svg}}/>;
}

export default MermaidDiagram;