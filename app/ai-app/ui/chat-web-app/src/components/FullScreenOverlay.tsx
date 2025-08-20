import React from "react";

interface FullScreenOverlayProps {
    className?: string | undefined
    onClick?: React.MouseEventHandler<HTMLDivElement>;
}

const FullScreenOverlay = (props: FullScreenOverlayProps) => {

    return (
        <div id={FullScreenOverlay.name}
             className={"absolute bottom-0 left-0 w-full h-full bg-transparent pointer-events-auto " + (props.className || "")}
             onClick={props.onClick}
        />
    )
}

export default FullScreenOverlay;