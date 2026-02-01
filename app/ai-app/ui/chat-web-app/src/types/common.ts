import {ReactNode} from "react";

export interface WithReactChildren {
    children?: ReactNode | ReactNode[];
}

export interface Indexed {
    index: number;
}

export interface Timestamped {
    timestamp: number;
}

export interface ISOTimestamped {
    timestamp: string;
}