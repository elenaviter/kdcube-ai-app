import {ExoticComponent, useMemo} from "react";
import {LucideProps} from "lucide-react";
import {emSize} from "./shared.ts";

type IconType = ExoticComponent<LucideProps>

interface IconContainerProps {
    icon: IconType
    size?: number
    className?: string
    containerClassName?: string
}

const IconContainer = ({icon, className, containerClassName, size = 4}: IconContainerProps) => {
    return useMemo(() => {
        const IconComponent = icon;
        return <div className={`w-fit h-fit ${containerClassName ?? ""}`}>
            <IconComponent size={size * emSize} className={className}/>
        </div>
    }, [className, containerClassName, icon, size]);
}

export default IconContainer;