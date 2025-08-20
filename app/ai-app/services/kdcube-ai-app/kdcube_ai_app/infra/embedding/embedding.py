# infra/embedding/embedding.py
from typing import Callable

import requests, os

from kdcube_ai_app.infra.accounting import track_embedding, ServiceUsage
from kdcube_ai_app.infra.llm.llm_data_model import ModelRecord, AIProviderName, \
    AIProvider

def convert_embedding_to_string(embedding):
    if isinstance(embedding, list):
        # Use repr() instead of str() to maintain full precision
        return "[" + ",".join(repr(x) for x in embedding) + "]"
    return embedding

def parse_embedding(embedding_str):
    if embedding_str is None or not isinstance(embedding_str, str):
        return embedding_str
    if embedding_str.startswith('[') and embedding_str.endswith(']'):
        values_str = embedding_str[1:-1]
        if not values_str.strip():
            return []
        values = [float(v.strip()) for v in values_str.split(',')]
        return values
    return embedding_str

def _usage_from_result(result, model: ModelRecord, text: str, size=None, **_):
    # generic, infra-level estimate; callers can override with richer extractor if needed
    tokens = int(len(text.split()) * 1.3)
    dims = len(result) if isinstance(result, list) else 0

    return ServiceUsage(
        embedding_tokens=tokens,
        embedding_dimensions=dims,
        requests=1
    )

def _metadata_from_context(model,
                           text, size=None, **_):
    # merge any enrichment metadata from with_accounting(...)
    from kdcube_ai_app.infra.accounting import get_enrichment
    enrich = get_enrichment()
    base = {
        "text_length": len(text),
        "word_count": len(text.split()),
        "embedding_size": size,
        "processing_method": "streaming"
    }
    extra = enrich.get("metadata") or {}
    base.update(extra)
    return base

@track_embedding(
    usage_extractor=_usage_from_result,
    metadata_extractor=_metadata_from_context
)
def get_embedding(model: ModelRecord,
                  text: str,
                  size: int = None,
                  self_hosted_serving_endpoint: str = None) -> list[float]:
    if size == 1536 and model.provider.provider != AIProviderName.open_ai:
        raise ValueError(f"Invalid embedding size for provider {model.provider.provider}")

    provider_id = model.provider.provider
    api_key = model.provider.apiToken
    system_name = model.systemName

    self_hosted_serving_endpoint = self_hosted_serving_endpoint if self_hosted_serving_endpoint else os.environ.get("SELF_HOSTED_SERVING_ENDPOINT", "https://b340-88-76-169-41.ngrok-free.app")
    EMBEDDING_ENDPOINTS = {
        AIProviderName.self_hosted: f"{self_hosted_serving_endpoint}/v1/embeddings",
        AIProviderName.open_ai:      "https://api.openai.com/v1/embeddings",
        AIProviderName.hugging_face: "https://api-inference.huggingface.co/pipeline/feature-extraction/<MODEL_ID>",
    }

    # Construct the base inference URL from your dictionary
    url_template = EMBEDDING_ENDPOINTS.get(provider_id, "")
    if not url_template:
        raise ValueError(f"No known embedding endpoint for provider={provider_id}")

    # Replace <MODEL_ID> if present
    url = url_template.replace("<MODEL_ID>", system_name)

    # print(f"Embedding for {provider_id}. Key={api_key}")
    # print(f"Embedding for {provider_id}.")
    # print(f"POST => {url}")

    # Prepare request
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if provider_id == AIProviderName.open_ai:
        payload = {
            "input": text,
            "model": system_name,
            "encoding_format": "float"
        }
    elif provider_id in [AIProviderName.hugging_face, AIProviderName.self_hosted]:
        payload = {
            "inputs": text,
            "model": system_name
        }

    # Perform the request
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()  # Raise an error if not 2xx

    result = response.json()
    if provider_id == AIProviderName.open_ai:
        return result["data"][0]["embedding"]
    elif provider_id == AIProviderName.self_hosted:
        return result["embedding"]
    else:
        if isinstance(result, list) and len(result) > 0:
            if isinstance(result[0], list) and len(result[0]) > 0 and isinstance(result[0][0], float):
                # Single pass
                return result[0]
            elif (isinstance(result[0], list) and
                  isinstance(result[0][0], list) and
                  len(result[0][0]) > 0):
                # Possibly triple nesting
                return result[0][0]
            else:
                raise ValueError(f"Unexpected HF embedding format: {result}")
        else:
            raise ValueError(f"No embedding returned, or invalid format: {result}")

def embedder_model(size: int,
                   get_key_fn: Callable[[AIProviderName], str]) -> ModelRecord:

    systemName = None
    if size == 1536:
        provider = AIProvider(provider=AIProviderName.open_ai,
                              apiToken=get_key_fn(AIProviderName.open_ai))
        systemName = "text-embedding-ada-002"
    else:
        provider = AIProvider(provider=AIProviderName.hugging_face,
                              apiToken=get_key_fn(AIProviderName.hugging_face))

        if size == 1024:
            # Produces more expressive embeddings at a higher cost.
            systemName = "sentence-transformers/stsb‑roberta‑large"
        elif size == 768:
            # Provides high-quality embeddings for general use.
            systemName = "sentence-transformers/all‑mpnet‑base‑v2"
            # A distilled model that balances speed and accuracy.
            # sentence-transformers/all‑distilroberta‑v1
        elif size == 512:
            # Supports multiple languages effectively.
            systemName = "sentence-transformers/distiluse‑base‑multilingual‑cased"
        elif size == 384:
            systemName = "sentence-transformers/all‑MiniLM‑L6‑v2"

    if systemName:
        return ModelRecord(
            modelType="base",
            status="active",
            provider=provider,
            systemName=systemName,
        )
