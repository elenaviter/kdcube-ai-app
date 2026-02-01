import {ExoticComponent, useMemo} from "react";
import {LucideProps} from "lucide-react";

type IconType = ExoticComponent<LucideProps>

interface IconContainerProps {
    icon: IconType
    size?: number
    className?: string
}

const emSize = parseFloat(getComputedStyle(document.body).fontSize);

const IconContainer = ({icon, className, size = 4}: IconContainerProps) => {
    // const IconComponent = icon;
    // const sizePX = `${size * emSize}px`;
    // return <div style={{
    //     height: sizePX,
    //     width: sizePX,
    // }}>
    //     <IconComponent size={size * emSize} className={className}/>
    // </div>;

    return useMemo(() => {
        const IconComponent = icon;
        return <IconComponent size={size * emSize} className={className}/>
    }, [className, icon, size]);
}

export default IconContainer;