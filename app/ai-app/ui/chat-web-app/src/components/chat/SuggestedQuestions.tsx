// SuggestedQuestions.tsx

import {Loader} from "lucide-react";

interface SuggestedQuestionsProps {
    isUpdating?: boolean;
    isDisabled?: boolean;
    onClick?: (question:string) => void;
    questions: string[];
}

const SuggestedQuestions = ({isUpdating, isDisabled, onClick, questions}:SuggestedQuestionsProps) => {

    return (
        <div className="flex flex-col px-6 py-4 bg-slate-100 border-b border-gray-400">
            {isUpdating ? (
                <div className="w-full flex">
                    <Loader size={28} className='animate-spin text-gray-300 mx-auto'/>
                </div>
            ) : (
                <>
                    <h4 className="text-sm font-medium text-slate-700 mb-2 mx-auto ">Try asking these questions:</h4>
                    <div className="flex flex-row flex-wrap justify-center gap-2">
                        {questions.map((q, idx) => (
                            <button key={idx} onClick={() => onClick?.(q)}
                                    disabled={isDisabled}
                                    className="px-3 py-1 text-xs bg-slate-300 hover:bg-slate-400 text-slate-700 border border-slate-200 rounded-full hover:border-slate-300 disabled:opacity-50">
                                {q}
                            </button>
                        ))}
                    </div>
                </>
            )}
        </div>
    )
}

export default SuggestedQuestions;