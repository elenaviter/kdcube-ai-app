# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

from typing import List, Dict, Any, Optional, Tuple
import json
import re, hashlib

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_core.embeddings import Embeddings

from kdcube_ai_app.apps.chat.sdk.inventory import AgentLogger, CustomEmbeddings, Config
from kdcube_ai_app.apps.chat.sdk.storage.ai_bundle_storage import AIBundleStorage

class _PrecomputedThenRealEmbeddings(Embeddings):
    """
    Embeddings adapter:
      - For building the index with FAISS.from_texts: return precomputed vectors for the given texts.
      - For querying: delegate to the real embedding model.
    """
    def __init__(self, pre_map: Dict[str, List[float]], real: Embeddings):
        self._map = pre_map
        self._real = real

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._map[t] for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._real.embed_query(text)

def _builtin_docs():
    sample_docs = [
        Document(
            page_content="""
Light, Watering, and Soil Basics for Houseplants

Getting the basics right prevents 80% of problems.

- **Light**:
  - *Bright direct*: Strong sunbeams for several hours (e.g., south-facing sill).
  - *Bright indirect*: Bright room but sun not striking leaves; soft shadows (ideal for many aroids).
  - *Medium*: You can read comfortably without lamps; fuzzy shadows.
  - *Low*: You need lights on to read; many plants will only “survive,” not thrive.
  - Tip: Use the **hand-shadow test**—sharp shadow = bright; blurry = medium; barely visible = low.

- **Watering**:
  - Water thoroughly until excess drains; empty saucers after 10–15 min.
  - Let the top 2–3 cm (one knuckle) dry for most tropicals; let half or more dry for succulents/cacti.
  - Signs of **overwatering**: constant wet soil, yellowing lower leaves, mushy stems.
  - Signs of **underwatering**: dry, crispy tips/edges, wilting that perks up after watering.

- **Soil / Potting Mix**:
  - Aim for **drainage + aeration**. Start with all-purpose potting mix.
  - Add **perlite/pumice** for air; **orchid bark** for chunk; a little **coco coir** for water retention.
  - Cacti/succulents: gritty mix (more mineral, less organic).
  - Always use pots with **drainage holes**.

- **Environment**:
  - Most houseplants like 18–27°C and moderate humidity. Avoid cold drafts and hot radiators.
""",
            metadata={"source": "basics_light_watering_soil.md", "type": "documentation"}
        ),
        Document(
            page_content="""
Common Pests & Simple Integrated Pest Management (IPM)

- **Identify & Isolate**: Move the plant away from others. Confirm the pest before treating.
- **Frequent Culprits**:
  - *Spider mites*: Fine webbing; stippled, dusty leaves—often in dry air.
  - *Mealybugs*: White cottony tufts in crevices, on stems/leaf nodes.
  - *Scale*: Brown/amber bumps stuck to stems/undersides of leaves.
  - *Aphids*: Soft green/black clusters on tender new growth.
  - *Fungus gnats*: Tiny flies emerging from soil; larvae feed on roots in soggy mix.

- **Treatment Basics**:
  - Mechanical first: rinse in the shower; wipe leaves; cotton swabs with diluted alcohol for mealybugs/scale.
  - **Insecticidal soap** or **horticultural oil/neem**: Cover upper/lower leaf surfaces and stems. Repeat weekly for 3–4 weeks to break life cycles.
  - For fungus gnats: let top soil dry, use yellow sticky traps for adults; improve drainage; bottom-water when possible.

- **Prevention**:
  - Quarantine new plants 2–3 weeks.
  - Avoid overwatering; increase airflow; keep leaves dust-free.
  - Inspect undersides of leaves during routine watering.
""",
            metadata={"source": "pests_ipm.md", "type": "documentation"}
        ),
        Document(
            page_content="""
Propagation 101: Cuttings, Division, and More

- **Stem cuttings (vining aroids: pothos/philodendron/monstera)**:
  1. Cut just below a node (where leaf + aerial root emerge).
  2. Remove lower leaf; place node in water or airy soil mix.
  3. Keep warm and bright-indirect; refresh water weekly; plant up when roots are 2–5 cm long.

- **Leaf cuttings (many succulents, snake plant)**:
  1. Take a healthy leaf; let cut end callus (1–2 days for succulents).
  2. Place on/in barely moist gritty mix; mist lightly until new roots/pups appear.

- **Division (ferns, peace lily, snake plant, ZZ)**:
  1. Unpot; gently tease apart natural clumps/rhizomes.
  2. Pot divisions separately; keep evenly moist until established.

- **Air layering (woody stems/monstera)**:
  1. Wound lightly below a node; wrap moist sphagnum; cover with plastic.
  2. Once roots form, cut below the rooted section and pot.

- **Aftercare**:
  - Bright-indirect light, consistent light moisture (not soggy), and high humidity speed rooting.
  - Avoid strong fertilizer until robust new growth appears.
""",
            metadata={"source": "propagation_101.md", "type": "documentation"}
        ),
        Document(
            page_content="""
Repotting & Fertilizing Guide

- **When to repot**:
  - Roots circling the pot or poking from drainage holes
  - Water runs straight through or dries unusually fast
  - Plant is top-heavy or growth has stalled
  - Best season: **spring to early summer**

- **How to repot**:
  1. Choose a pot 2–5 cm wider than current (one size up).
  2. Loosen circling roots; remove dead/brown bits.
  3. Refresh with appropriate mix (chunkier for aroids; gritty for succulents).
  4. Water thoroughly; keep out of direct sun for a few days.

- **Fertilizing**:
  - Use a balanced liquid fertilizer (e.g., 10-10-10 or 20-20-20) **diluted to half strength**.
  - Feed during active growth (spring/summer) every 2–4 weeks; reduce or pause in winter.
  - Flush pots with plain water every 1–2 months to reduce mineral buildup.
  - Avoid fertilizing for 4–6 weeks after repotting or when plants are stressed.

- **Salts & Leaf Tips**:
  - Brown, crispy tips can indicate salt buildup or underwatering—flush soil and adjust watering.
""",
            metadata={"source": "repotting_fertilizing.md", "type": "documentation"}
        ),
        Document(
            page_content="""
Troubleshooting Leaf Symptoms

- **Yellow lower leaves**: Often overwatering or insufficient light.
  - Action: Let soil dry to the correct depth; increase light (bright-indirect).

- **Brown crispy edges/tips**: Underwatering, low humidity, or salt buildup.
  - Action: Water more deeply/consistently; consider a humidity tray; flush soil.

- **Drooping with wet soil**: Overwatering/lack of oxygen at roots.
  - Action: Improve drainage/aeration; check for root rot; repot if necessary.

- **Pale leaves/chlorosis**: Nutrient imbalance or pH issue, sometimes low light.
  - Action: Provide balanced feeding during growth; refresh potting mix; improve light.

- **Leaves curling inward**: Low humidity, underwatering, heat/draft, or pests.
  - Action: Stabilize environment; inspect undersides of leaves.

- **Sudden leaf drop after move**: Normal transplant/shock response.
  - Action: Hold steady on care; avoid overcorrection; new growth should normalize.

General rule: change **one variable at a time**, observe 1–2 weeks, and keep a simple care log.
""",
            metadata={"source": "troubleshooting_symptoms.md", "type": "documentation"}
        )
    ]
    return sample_docs

class RAGService:
    """RAG service for document retrieval with custom embeddings support"""

    def __init__(self, config: Config, storage: AIBundleStorage):
        self.config = config
        self.logger = AgentLogger("RAGService", config.log_level)
        self.vector_store = None
        self.storage = storage
        self._setup_embeddings_and_data()

    # ---------- setup ----------

    def _setup_embeddings_and_data(self):
        op = self.logger.start_operation("setup_embeddings_and_data")

        embedder_config = self.config.embedder_config
        provider = embedder_config["provider"]
        model_name = embedder_config["model_name"]
        dim = int(embedder_config["dim"])

        if provider == "openai":
            self.embeddings = OpenAIEmbeddings(
                model=model_name,
                openai_api_key=self.config.openai_api_key
            )
            self.logger.log_step("openai_embeddings_initialized", {
                "embedder_id": self.config.selected_embedder,
                "model": model_name,
                "provider": "openai",
                "dimension": dim
            })
        elif provider == "custom":
            if not self.config.custom_embedding_endpoint:
                raise ValueError(f"Custom embedder {self.config.selected_embedder} requires an endpoint")
            self.embeddings = CustomEmbeddings(
                endpoint=self.config.custom_embedding_endpoint,
                model=model_name,
                size=dim
            )
            self.logger.log_step("custom_embeddings_initialized", {
                "embedder_id": self.config.selected_embedder,
                "endpoint": self.config.custom_embedding_endpoint,
                "model": model_name,
                "provider": "custom",
                "dimension": dim
            })
        else:
            raise ValueError(f"Unknown embedding provider: {provider}")

        # --- Built-in small KB (same content as before) ---
        sample_docs = _builtin_docs()
        self.logger.log_step("sample_docs_created", {
            "total_documents": len(sample_docs),
            "doc_previews": [doc.page_content[:100] + "..." for doc in sample_docs]
        })

        # --- Load or build cached embeddings for each doc ---
        kb_items: List[Tuple[Document, List[float]]] = []
        to_embed: List[Tuple[str, Document]] = []
        pre_map: Dict[str, List[float]] = {}

        for doc in sample_docs:
            doc_id = self._doc_id_for(doc)
            key = self._kb_key(doc_id)
            cached = self._try_load_cached_doc(key)

            if cached and self._is_embedding_compatible(cached, provider, model_name, dim):
                vec = cached["embedding"]
                kb_items.append((doc, vec))
                pre_map[doc.page_content] = vec
                self.logger.log_step("kb_cache_hit", {"doc_id": doc_id, "key": key})
            else:
                to_embed.append((doc_id, doc))
                self.logger.log_step("kb_cache_miss", {"doc_id": doc_id, "key": key})

        # Embed only the missing ones
        if to_embed:
            texts = [d.page_content for _, d in to_embed]
            vecs = self.embeddings.embed_documents(texts)  # batch
            for (doc_id, doc), vec in zip(to_embed, vecs):
                kb_items.append((doc, vec))
                pre_map[doc.page_content] = vec
                self._save_doc_json(
                    self._kb_key(doc_id),
                    {
                        "id": doc_id,
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "embedding": vec,
                        "embedder": {
                            "provider": provider,
                            "model": model_name,
                            "dimension": dim,
                            "selected_embedder": self.config.selected_embedder,
                        },
                        "sha256": self._sha256(doc.page_content),
                    },
                )
                self.logger.log_step("kb_cache_write", {"doc_id": doc_id, "dimension": len(vec)})

        # --- Build FAISS without re-embedding: try from_embeddings, else adapter ---
        try:
            # Newer langchain supports FAISS.from_embeddings([(vec, doc), ...], embedding=<Embeddings>)
            pairs = [ (vec, doc) for (doc, vec) in kb_items ]
            if hasattr(FAISS, "from_embeddings"):
                self.vector_store = FAISS.from_embeddings(pairs, self.embeddings)
            else:
                # Fallback: build via from_texts with an adapter that supplies precomputed vectors for build
                adapter = _PrecomputedThenRealEmbeddings(pre_map, self.embeddings)
                texts = [doc.page_content for (doc, _) in kb_items]
                metas = [doc.metadata for (doc, _) in kb_items]
                self.vector_store = FAISS.from_texts(texts, adapter, metadatas=metas)
            self.logger.log_step("vector_store_created", {
                "store_type": "FAISS",
                "embedding_provider": provider,
                "document_count": len(kb_items)
            })
        except Exception as e:
            self.logger.log_error(e, "vector_store_creation_failed")
            self.vector_store = None

        self.logger.finish_operation(True, f"Setup complete with {len(kb_items)} cached docs")

    # ---------- public ----------

    async def retrieve_documents(self, queries: List[Dict[str, Any]], k: int = 3) -> List[Dict[str, Any]]:
        op = self.logger.start_operation(
            "retrieve_documents",
            query_count=len(queries),
            k=k,
            queries=[q.get("query", "")[:50] + "..." for q in queries]
        )

        if not self.vector_store:
            self.logger.log_step("no_vector_store", {"message": "Vector store not available"})
            self.logger.finish_operation(False, "No vector store available")
            return []

        all_docs = []
        for i, query_data in enumerate(queries):
            query_text = query_data.get("query", "")
            weight = query_data.get("weight", 1.0)
            self.logger.log_step(f"processing_query_{i}", {
                "query": query_text, "weight": weight, "query_length": len(query_text)
            })
            try:
                docs = self.vector_store.similarity_search(query_text, k=k)
                self.logger.log_step(f"query_{i}_results", {
                    "retrieved_count": len(docs),
                    "doc_previews": [doc.page_content[:100] + "..." for doc in docs]
                })
                for doc in docs:
                    all_docs.append({
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "query": query_text,
                        "weight": weight
                    })
            except Exception as e:
                self.logger.log_error(e, f"Query {i} retrieval failed")

        # Deduplicate by content head
        seen = set()
        unique_docs = []
        for doc in all_docs:
            key = doc["content"][:100]
            if key not in seen:
                seen.add(key)
                unique_docs.append(doc)

        self.logger.log_step("deduplication_complete", {
            "original_count": len(all_docs),
            "unique_count": len(unique_docs),
            "duplicates_removed": len(all_docs) - len(unique_docs)
        })
        self.logger.finish_operation(True, f"Retrieved {len(unique_docs)} unique documents")
        return unique_docs

    # ---------- helpers ----------

    def _kb_key(self, doc_id: str) -> str:
        return f"rag/kb/{doc_id}/doc.json"  # bundle root is implicit in AIBundleStorage

    def _sha256(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _slugify(self, s: str) -> str:
        s = (s or "").strip().lower()
        s = re.sub(r"[^\w\-\.]+", "-", s)        # keep word chars, dash, dot, underscore
        s = re.sub(r"-{2,}", "-", s).strip("-")
        return s or "doc"

    def _doc_id_for(self, doc: Document) -> str:
        src = (doc.metadata or {}).get("source")
        if src:
            return self._slugify(str(src).rsplit("/", 1)[-1].rsplit(".", 1)[0])
        return self._sha256(doc.page_content)[:16]

    def _try_load_cached_doc(self, key: str) -> Optional[Dict[str, Any]]:
        try:
            raw = self.storage.read(key, as_text=True)
            return json.loads(raw)
        except Exception:
            return None

    def _is_embedding_compatible(self, cached: Dict[str, Any], provider: str, model: str, dim: int) -> bool:
        try:
            emb = cached.get("embedding")
            meta = (cached.get("embedder") or {})
            return (
                    isinstance(emb, list) and len(emb) == dim and
                    meta.get("provider") == provider and
                    meta.get("model") == model
            )
        except Exception:
            return False

    def _save_doc_json(self, key: str, obj: Dict[str, Any]) -> None:
        data = json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=False)
        self.storage.write(key, data, mime="application/json")