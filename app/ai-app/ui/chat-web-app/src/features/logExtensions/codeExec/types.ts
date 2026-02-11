import {Timestamped} from "../../../types/common.ts";
import {Artifact} from "../../chat/chatTypes.ts";

export interface CodeExecBase<T> extends Timestamped {
    name: string;
    title?: string | null;
    description?: string | null;
    content: T;
}

export type CodeExecProgramName = CodeExecBase<string>
export type CodeExecObjective = CodeExecBase<string>

export interface CodeExecProgram extends CodeExecBase<string> {
    language: string;
}

export interface CodeExecContractArtifact {
    "artifact_name": string,
    "mime": string,
    "filename": string,
    "description": string,
}

export type CodeExecContract = CodeExecBase<CodeExecContractArtifact[]>

export interface CodeExecStatusFormat {
    status: "gen" | "exec" | "done" | "error";
    timings: {
        "codegen": number,
        "exec": number
    };
    "error": Record<string, string>;
}

export type CodeExecStatus = CodeExecBase<CodeExecStatusFormat>

export interface CodeExecData {
    executionId: string
    program?: CodeExecProgram;
    name?: CodeExecProgramName;
    objective?: CodeExecObjective;
    contract?: CodeExecContract;
    status?: CodeExecStatus;
}

export const CodeExecArtifactType = "code_exec";

export interface CodeExecArtifact extends Artifact<CodeExecData> {
    artifactType: typeof CodeExecArtifactType;
}