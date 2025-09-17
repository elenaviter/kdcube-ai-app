
# Online Python - IDE, Editor, Compiler, Interpreter
from typing import Dict, Any, List, Optional, Literal
from pydantic import BaseModel, Field, conlist
from datetime import datetime, timezone
import json


class ContractItem(BaseModel):
    rid: str = Field(..., description="Stable resource id the program MUST use in result.out[].")
    type: Literal["inline","file"] = "inline"
    # For inline
    format: Optional[Literal["markdown","text","json","url"]] = None
    # For file
    mime: Optional[str] = None
    filename_hint: Optional[str] = None

    description: str = ""
    citable: Optional[bool] = None
    source_hint: Optional[str] = Field(default=None, description="Either 'program' or an adapter id to prefer (e.g., 'generic_tools.write_pdf').")
    require_text_fallback: bool = Field(default=False, description="If type='file', require a paired inline description/content.")
    fallback_rid: Optional[str] = Field(default=None, description="Rid to use for the text fallback when required.")
    min_count: int = 1
    max_count: Optional[int] = 1

class SolvabilityOut(BaseModel):
    solvable: bool
    confidence: float = Field(0.5, ge=0.0, le=1.0)
    reasoning: str = ""
    tools_to_use: List[str] = Field(default_factory=list)
    clarifying_questions: List[str] = Field(default_factory=list)

    # NOTE: unified names — no 'single_call' anymore
    solver_mode: Literal["direct_tools_exec","codegen","llm_only"] = "llm_only"
    solver_reason: str = ""  # short justification

    # Dynamic contract the program must fulfill iff solver_mode='codegen'
    # map: slot name -> human description of what should be produced
    output_contract_dyn: Optional[Dict[str, str]] = Field(default_factory=dict)

    context_use: bool = True
    project_canvas_slot: str = "project_canvas"                  # the slot codegen MUST populate for text representation of solution
    project_log_slot: str = "project_log"
    history_select: str = "latest"                      # 'latest' | 'by_mention' | 'by_similarity'
    history_select_hint: str = ""                       # short instruction when not 'latest'
    citations_source_path: str = "program_history[].<exec>.web_links_citations.items"
    instructions_for_codegen: str = ""                  # ≤80 words, concrete steps to read context and pick version

j = '\n{\n  \"solvable\": true,\n  \"confidence\": 0.9,\n  \"reasoning\": \"The selected tools can gather relevant information, summarize it, and render the summary into a PDF document.\",\n  \"tools_to_use\": [\"generic_tools.web_search\", \"llm_tools.summarize_llm\", \"generic_tools.write_pdf\"],\n  \"clarifying_questions\": [],\n  \"solver_mode\": \"codegen\",\n  \"solver_reason\": \"Multiple tools are needed to complete the request.\",\n  \"output_contract_dyn\": {\n    \"project_canvas\": \"# Latest Advances in Fighting LLM Hallucination with Agentic Apps\\n\\n[[S:1]] Summary of the latest advances in fighting LLM hallucination with agentic apps.\",\n    \"project_log\": \"Objective: Produce a 1-2 page PDF summary of the latest advances in fighting LLM hallucination with agentic apps.\\nActions:\\n1. Perform a web search to gather relevant information.\\n2. Synthesize the gathered information into a concise summary.\\n3. Render the summary into a PDF document.\",\n    \"pdf_file\": {\n      \"path\": \"summary.pdf\",\n      \"title\": \"Latest Advances in Fighting LLM Hallucination with Agentic Apps\",\n      \"content_md\": \"[[S:1]] <TBD at runtime>\",\n      \"sources\": null,\n      \"resolve_citations\": true,\n      \"include_sources_section\": false\n    }\n  }\n}'
j = """\n{\n  \"solvable\": true,\n  \"confidence\": 0.9,\n  \"reasoning\": \"The provided tools cover the necessary functionality to address the user's request.\",\n  \"tools_to_use\": [\"generic_tools.web_search\", \"llm_tools.summarize_llm\", \"generic_tools.write_pdf\"],\n  \"clarifying_questions\": [],\n  \"solver_mode\": \"codegen\",\n  \"solver_reason\": \"Multiple tools are needed to complete the request.\",\n  \"output_contract_dyn\": {\n    \"project_canvas\": \"# Recent Advances in Fighting Hallucinations in LLMs with Agentic Applications\\n\\n[[S:1]] Hallucinations in large language models (LLMs) are a significant challenge, especially when these models are used in agentic applications where they need to make decisions and take actions. This report summarizes recent research and techniques for mitigating hallucinations in LLMs in the context of agentic AI systems.\\n\\n[[S:2]] Key topics covered include:\\n- Techniques for detecting and filtering out hallucinated outputs from LLMs\\n- Approaches to making LLMs more robust and reliable, such as using constitutional AI principles\\n- Case studies of agentic AI applications that have successfully addressed the hallucination problem\\n\\n[[S:3]] The report concludes with a discussion of the remaining challenges and future research directions in this important area of AI safety and reliability.\",\n    \"project_log\": \"Objective: Provide a summary report on recent advances in fighting hallucinations in LLMs used in agentic AI applications.\\n\\nActions:\\n1. Perform a web search to find relevant sources on the topic.\\n2. Summarize the key points from the sources using an LLM summarization tool.\\n3. Render the summary into a high-quality PDF document.\",\n    \"pdf_file\": {\n      \"path\": \"summary.pdf\",\n      \"title\": \"Recent Advances in Fighting Hallucinations in LLMs with Agentic Applications\",\n      \"sources\": \"<TBD at runtime>\",\n      \"content_md\": \"<TBD at runtime>\",\n      \"resolve_citations\": true,\n      \"include_sources_section\": true\n    }\n  }\n}"""
# j = '\n{\n  \"solvable\": true,\n  \"confidence\": 0.9,\n  \"reasoning\": \"The selected tools can gather relevant information, summarize it, and render the summary into a PDF document.\",\n  \"tools_to_use\": [\"generic_tools.web_search\", \"llm_tools.summarize_llm\", \"generic_tools.write_pdf\"],\n  \"clarifying_questions\": [],\n  \"solver_mode\": \"codegen\",\n  \"solver_reason\": \"Multiple tools are needed to complete the request.\",\n  \"output_contract_dyn\": {\n    \"project_canvas\": \"# Latest Advances in Fighting LLM Hallucination with Agentic Apps\\n\\n[[S:1]] Summary of the latest advances in fighting LLM hallucination with agentic apps.\",\n    \"project_log\": \"Objective: Produce a 1-2 page PDF summary of the latest advances in fighting LLM hallucination with agentic apps.\\nActions:\\n1. Perform a web search to gather relevant information.\\n2. Synthesize the gathered information into a concise summary.\\n3. Render the summary into a PDF document.\",\n    \"pdf_file\": \"Latest Advances in Fighting LLM Hallucination with Agentic Apps\"  }\n}'
loaded = json.loads(j)
print(json.dumps(loaded, indent=2))

print(SolvabilityOut.model_validate(loaded).model_dump())


# If tools list is empty, check that it's not the serialization issue.
# if the tools list is empty, and the solvability still decides the task is solvable, only then run codegen.
# codegen MUST know what was the previous program presentation. it does not know seems