import {useMemo, useState} from "react";
import IconContainer from "./IconContainer.tsx";
import {ExternalLink} from "lucide-react";
import {emSize} from "./shared.ts";

type IconStatus = "loaded" | "error" | null

interface IconLoaderProps {
    url: string | null | undefined;
    className?: string;
    size: number;
}

const IconLoader = ({url, className, size}: IconLoaderProps) => {
    const [status, setStatus] = useState<IconStatus>(null);

    return useMemo(() => {
        if (url) {
            switch (status) {
                case null:
                case "loaded":
                    return <img src={url} alt={"fav icon"} className={className}
                                style={{width: size * emSize, height: size * emSize}}
                                onError={(e) => {
                                    console.debug("Unable to load favicon", url, e)
                                    setStatus("error");
                                }}/>
            }
        }
        return <IconContainer size={size} icon={ExternalLink}/>
    }, [className, size, status, url]);
}

export default IconLoader;