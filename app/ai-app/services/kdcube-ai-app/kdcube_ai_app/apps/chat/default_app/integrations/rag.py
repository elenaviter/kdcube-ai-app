# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

from typing import List, Dict, Any

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from kdcube_ai_app.apps.chat.sdk.inventory import AgentLogger, CustomEmbeddings, Config


class RAGService:
    """RAG service for document retrieval with custom embeddings support"""

    def __init__(self, config: Config):
        self.config = config
        self.logger = AgentLogger("RAGService", config.log_level)
        self.vector_store = None
        self._setup_embeddings_and_data()

    def _setup_embeddings_and_data(self):
        """Setup embeddings service and sample data"""
        operation_start = self.logger.start_operation("setup_embeddings_and_data")

        embedder_config = self.config.embedder_config
        if embedder_config["provider"] == "openai":
            # Initialize OpenAI embeddings
            self.embeddings = OpenAIEmbeddings(
                model=embedder_config["model_name"],
                openai_api_key=self.config.openai_api_key
            )
            self.logger.log_step("openai_embeddings_initialized", {
                "embedder_id": self.config.selected_embedder,
                "model": embedder_config["model_name"],
                "provider": "openai",
                "dimension": embedder_config["dim"]
            })

        elif embedder_config["provider"] == "custom":
            # Initialize custom embeddings
            if not self.config.custom_embedding_endpoint:
                raise ValueError(f"Custom embedder {self.config.selected_embedder} requires an endpoint")

            self.embeddings = CustomEmbeddings(
                endpoint=self.config.custom_embedding_endpoint,
                model=embedder_config["model_name"],
                size=embedder_config["dim"]
            )
            self.logger.log_step("custom_embeddings_initialized", {
                "embedder_id": self.config.selected_embedder,
                "endpoint": self.config.custom_embedding_endpoint,
                "model": embedder_config["model_name"],
                "provider": "custom",
                "dimension": embedder_config["dim"]
            })
        else:
            raise ValueError(f"Unknown embedding provider: {embedder_config['provider']}")

        # Create sample documents
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


        self.logger.log_step("sample_docs_created", {
            "total_documents": len(sample_docs),
            "doc_previews": [doc.page_content[:100] + "..." for doc in sample_docs]
        })

        # Create FAISS vector store
        try:
            self.vector_store = FAISS.from_documents(sample_docs, self.embeddings)
            self.logger.log_step("vector_store_created", {
                "store_type": "FAISS",
                "embedding_type": "custom" if self.config.custom_embedding_endpoint else "openai",
                "document_count": len(sample_docs)
            })
        except Exception as e:
            self.logger.log_error(e, "Vector store creation failed")
            self.vector_store = None

        self.logger.finish_operation(True, f"Setup complete with {len(sample_docs)} documents")

    async def retrieve_documents(self, queries: List[Dict[str, Any]], k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve documents based on weighted queries"""
        operation_start = self.logger.start_operation("retrieve_documents",
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
                "query": query_text,
                "weight": weight,
                "query_length": len(query_text)
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

        # Remove duplicates
        seen = set()
        unique_docs = []
        for doc in all_docs:
            doc_key = doc["content"][:100]  # Use first 100 chars as key
            if doc_key not in seen:
                seen.add(doc_key)
                unique_docs.append(doc)

        self.logger.log_step("deduplication_complete", {
            "original_count": len(all_docs),
            "unique_count": len(unique_docs),
            "duplicates_removed": len(all_docs) - len(unique_docs)
        })

        self.logger.finish_operation(True, f"Retrieved {len(unique_docs)} unique documents")
        return unique_docs
