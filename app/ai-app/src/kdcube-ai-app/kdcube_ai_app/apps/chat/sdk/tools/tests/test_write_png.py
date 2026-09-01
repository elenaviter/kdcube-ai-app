# SPDX-License-Identifier: MIT

import asyncio
import json
import os
from pathlib import Path

import pytest

from kdcube_ai_app.apps.chat.sdk.runtime.run_ctx import OUTDIR_CV, WORKDIR_CV
from kdcube_ai_app.apps.chat.sdk.tools.rendering_tools import RenderingTools

_DEFAULT_REQUEST = {
    "path": "turn_1771789172537_66qyvv/files/memory_system_readable.png",
    "content": """graph TB
      MEM["🧠 Conversation Memory System"]

      MEM --> Capture["📥 Systematic Capture"]
      MEM --> Normalize["🧹 Normalization & Cleanup"]
      MEM --> Store["🗂️ Storage & Indexing"]
      MEM --> Strength["💪 Memory Strength"]
      MEM --> Recency["⏱️ Recency & Decay"]
      MEM --> Retrieval["🔎 Memory Retrieval"]
      MEM --> Promote["⬆️ User-Level Promotion"]
      MEM --> Feedback["🔁 Feedback & Corrections"]

      Capture --> C1["Signal Extraction"]
      Capture --> C2["User-Provided Notes"]
      Capture --> C3["Session Metadata"]
      Capture --> C4["Scope Tags"]

      Normalize --> N1["Canonical Schema"]
      Normalize --> N2["Entity Resolution"]
      Normalize --> N3["PII Filtering"]
      Normalize --> N4["Deduplication"]

      Store --> S1["Memory Items"]
      Store --> S2["Embeddings"]
      Store --> S3["Keyword Index"]
      Store --> S4["Source References"]

      Strength --> ST1["Confidence Score"]
      Strength --> ST2["Repetition Boost"]
      Strength --> ST3["Source Reliability"]
      Strength --> ST4["Conflict Handling"]

      Recency --> R1["Time Decay"]
      Recency --> R2["Freshness Boosts"]
      Recency --> R3["Context Relevance"]

      Retrieval --> RE1["Memory Tool Query"]
      Retrieval --> RE2["Reranking"]
      Retrieval --> RE3["Context Injection"]
      Retrieval --> RE4["Permissions Guardrail"]

      Promote --> P1["Eligibility Rules"]
      Promote --> P2["User Consent"]
      Promote --> P3["Profile Update"]
      Promote --> P4["Strength Threshold"]

      Feedback --> F1["User Corrections"]
      Feedback --> F2["Forget Requests"]
      Feedback --> F3["Usage Telemetry"]

      style MEM fill:#2c3e50,stroke:#34495e,stroke-width:4px,color:#fff,font-size:18px
      style Capture fill:#3498db,stroke:#2980b9,stroke-width:3px,color:#fff,font-size:16px
      style Normalize fill:#9b59b6,stroke:#8e44ad,stroke-width:3px,color:#fff,font-size:16px
      style Store fill:#1abc9c,stroke:#16a085,stroke-width:3px,color:#fff,font-size:16px
      style Strength fill:#e67e22,stroke:#d35400,stroke-width:3px,color:#fff,font-size:16px
      style Recency fill:#95a5a6,stroke:#7f8c8d,stroke-width:3px,color:#fff,font-size:16px
      style Retrieval fill:#f39c12,stroke:#d68910,stroke-width:3px,color:#fff,font-size:16px
      style Promote fill:#2ecc71,stroke:#27ae60,stroke-width:3px,color:#fff,font-size:16px
      style Feedback fill:#e74c3c,stroke:#c0392b,stroke-width:3px,color:#fff,font-size:16px
    """,
    "format": "mermaid",
    "width": 4000,
    "zoom": 1.6,
    "mermaid_font_size_px": 22,
    "mermaid_scale": 1.4,
    "device_scale_factor": 3,
    "fit": "content",
    "padding_px": 60,
    "render_delay_ms": 2000,
    "background": "white",
    "mermaid_theme": "default",
}


def _prepare_dirs() -> Path:
    outdir = os.environ.get("WRITE_PNG_OUTDIR")
    if not outdir:
        outdir = Path(__file__).with_name("_out")
    outdir = Path(outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    workdir = os.environ.get("WRITE_PNG_WORKDIR")
    if not workdir:
        workdir = outdir
    workdir = Path(workdir).expanduser().resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    os.environ["OUTPUT_DIR"] = str(outdir)
    os.environ["WORKDIR"] = str(workdir)
    OUTDIR_CV.set(str(outdir))
    WORKDIR_CV.set(str(workdir))

    return outdir


def _load_request() -> dict:
    input_path = os.environ.get("WRITE_PNG_INPUT")
    if input_path:
        payload = json.loads(Path(input_path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("WRITE_PNG_INPUT must contain a JSON object")
        return payload
    return dict(_DEFAULT_REQUEST)


async def _run_write_png(request: dict) -> dict:
    tools = RenderingTools()
    return await tools.write_png(**request)


@pytest.mark.skipif(
    os.environ.get("WRITE_PNG_RUN") != "1",
    reason="Set WRITE_PNG_RUN=1 to execute the rendering test",
)
@pytest.mark.asyncio
async def test_write_png():
    outdir = _prepare_dirs()
    request = _load_request()
    result = await _run_write_png(request)
    assert result.get("ok") is True, result

    out_path = outdir / request["path"]
    assert out_path.exists(), f"PNG not created: {out_path}"
    size_bytes = out_path.stat().st_size
    print(f"write_png output: {out_path} ({size_bytes} bytes)")


if __name__ == "__main__":
    outdir = _prepare_dirs()
    request = _load_request()
    result = asyncio.run(_run_write_png(request))
    out_path = outdir / request["path"]
    print(result)
    print(f"Output: {out_path}")
    print(f"Exists: {out_path.exists()}")
