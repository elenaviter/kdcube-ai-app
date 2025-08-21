# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

"""
Simple search system that provides actual navigation data for frontend.
Removes all unnecessary complexity and focuses on what's needed for backtracking.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from kdcube_ai_app.apps.knowledge_base.db.data_models import HybridSearchParams
from kdcube_ai_app.apps.knowledge_base.db.kb_db_connector import NavigationSearchResult
from kdcube_ai_app.apps.knowledge_base.storage import KnowledgeBaseStorage
from kdcube_ai_app.apps.knowledge_base.modules.contracts.segmentation import SegmentType


@dataclass
class SearchResult:
    """Simple search result with actual navigation data."""
    query: str
    relevance_score: float
    heading: str
    subheading: str
    backtrack: Dict[str, Any]  # Actual navigation data for frontend


class SimpleSearcher:
    """Simple searcher that provides useful navigation data."""

    def __init__(self, storage: KnowledgeBaseStorage, segmentation_module):
        self.storage = storage
        self.segmentation_module = segmentation_module

    def search(self, query: str, resource_id: str, version: Optional[str] = None, top_k: int = 5) -> List[SearchResult]:
        """
        Search that returns proper navigation format.

        Returns backtrack in correct format:
        {
          "raw": {
            "citations": ["query"],
            "rn": "ef:project:knowledge_base:raw:resource_id:version"
          },
          "extraction": {
            "rn": [
              "ef:project:knowledge_base:extraction:resource_id:version:extraction_0.md",
              "ef:project:knowledge_base:extraction:resource_id:version:image_1.jpg",
              "ef:project:knowledge_base:extraction:resource_id:version:table_1.csv"
            ]
          },
          "segmentation": {
            "rn": "ef:project:knowledge_base:segmentation:retrieval:resource_id:version:segments.json",
            "navigation": [
              {
                "start_line": 145,
                "end_line": 157,
                "start_pos": 0,
                "end_pos": 1235,
                "citations": ["query"],
                "text": "actual text of this base segment",
                "heading": "heading",
                "subheading": "subheading"
              }
            ]
          }
        }
        """
        if not query.strip():
            return []

        if version is None:
            version = self.storage.get_latest_version(resource_id)
            if not version:
                return []

        # Get retrieval segments (best for search)
        segments = self.segmentation_module.get_segments_by_type(resource_id, version, SegmentType.RETRIEVAL)
        if not segments:
            return []

        # Get base segments for navigation
        base_segments = self.segmentation_module.get_base_segments(resource_id, version)
        base_lookup = {seg.guid: seg for seg in base_segments}

        query_lower = query.lower()
        scored_results = []

        for segment in segments:
            # Check if query is actually in the segment text
            text = segment.get("text", "")
            if query_lower not in text.lower():
                continue  # Skip irrelevant segments

            metadata = segment.get("metadata", {})
            base_guids = metadata.get("base_segment_guids", [])

            # Get base segments for this compound segment
            segment_base_segments = []
            for guid in base_guids:
                if guid in base_lookup:
                    segment_base_segments.append(base_lookup[guid])

            if not segment_base_segments:
                continue  # Skip if no base segments found

            # Calculate relevance score
            score = self._calculate_score(query_lower, text, metadata)

            # Create proper backtrack data

            backtrack = self._create_proper_backtrack(query,
                                                      resource_id,
                                                      version,
                                                      segment_base_segments)

            result = SearchResult(
                query=query,
                relevance_score=score,
                heading=metadata.get("heading", ""),
                subheading=metadata.get("subheading", ""),
                backtrack=backtrack
            )

            scored_results.append((result, score))

        # Sort by relevance and return top results
        scored_results.sort(key=lambda x: x[1], reverse=True)
        return [result for result, score in scored_results[:top_k]]

    def _calculate_score(self, query_lower: str, text: str, metadata: Dict[str, Any]) -> float:
        """Calculate simple relevance score."""
        text_lower = text.lower()
        heading_lower = metadata.get("heading", "").lower()
        subheading_lower = metadata.get("subheading", "").lower()

        score = 0.0

        # Text matches
        if query_lower in text_lower:
            score += 0.5
            score += min(text_lower.count(query_lower) * 0.1, 0.3)

        # Heading matches (higher weight)
        if query_lower in heading_lower:
            score += 0.8

        # Subheading matches
        if query_lower in subheading_lower:
            score += 0.6

        return score

    def _create_proper_backtrack(self, query: str, resource_id: str, version: str, base_segments: List) -> Dict[str, Any]:
        """
        Create proper navigation format that frontend can actually use.
        """
        query_lower = query.lower()

        # 1. RAW section
        raw_citations = []
        for base_seg in base_segments:
            if query_lower in base_seg.text.lower():
                raw_citations.append(query)

        raw_resource_rn = self.storage.get_version_metadata(resource_id, version).get("rn")
        extracted_resource_rn = next(iter([fi["rn"] for fi in (self.storage.get_extraction_results(resource_id, version) or []) if fi["content_file"].endswith(".md")]), None)
        # raw_resource_rn = f"ef:<tenant>:<project_id>:knowledge_base:raw:{resource_id}:{version}"
        #
        # extracted_resource_rn = f"ef:<tenant>:<project_id>:knowledge_base:extraction:{resource_id}:{version}:extraction_0.md"
        il_rn = extracted_resource_rn

        # 2. EXTRACTION section - LIST of all relevant extraction resources
        extraction_rns = []

        # Always include the main MD file
        extraction_rns.append(extracted_resource_rn)

        # Include all extraction resources referenced by these base segments
        all_extraction_rns = set()
        for base_seg in base_segments:
            all_extraction_rns.update(base_seg.extracted_data_rns)

        # Add unique extraction RNs (images, tables, etc.)
        for rn in all_extraction_rns:
            if rn not in extraction_rns:  # Avoid duplicates
                extraction_rns.append(rn)

        # 3. SEGMENTATION section - navigation with text blocks
        segmentation_rn = f"ef:running-shoes:knowledge_base:segmentation:retrieval:{resource_id}:{version}:segments.json"

        navigation = []
        for base_seg in base_segments:
            # Check if query is in this specific base segment
            base_citations = []
            if query_lower in base_seg.text.lower():
                base_citations.append(query)

            navigation_item = {
                "start_line": base_seg.start_line_num,
                "end_line": base_seg.end_line_num,
                "start_pos": base_seg.start_position,
                "end_pos": base_seg.end_position,
                "citations": base_citations,
                "text": base_seg.text,  # Actual text for frontend
                "heading": base_seg.heading,
                "subheading": base_seg.subheading
            }
            navigation.append(navigation_item)

        return {
            "raw": {
                "citations": list(set(raw_citations)),  # Remove duplicates
                "rn": raw_resource_rn
            },
            "extraction": {
                "related_rns": extraction_rns,  # LIST of all extraction resources
                "rn": il_rn
            },
            "segmentation": {
                "rn": segmentation_rn,
                "navigation": navigation  # Text blocks with navigation info
            }
        }


# Simplified API for knowledge base
class SimpleKnowledgeBaseSearch:
    """Simple search API for knowledge base."""

    def __init__(self, kb):
        self.kb = kb
        self.searcher = SimpleSearcher(
            kb.storage,
            kb.get_segmentation_module()
        )

    def search(self, query: str, resource_id: str, version: Optional[str] = None, top_k: int = 5) -> List[NavigationSearchResult]:
        """Simple search that returns useful results."""
        from kdcube_ai_app.infra.accounting import with_accounting
        with with_accounting("kb.search",
                             metadata={
                                "query": query,
                                "phase": "user_query_embedding"
                            }):
            query_embedding = self.kb.db_connector.get_embedding(query)

        resource_ids = [resource_id] if resource_id else None
        params = HybridSearchParams(
            query=query,
            top_n=20,
            min_similarity=0.2,
            text_weight=0.5,
            semantic_weight=0.5,
            embedding=query_embedding,
            match_all=False,
            rerank_threshold=0.6,
            rerank_top_k=5,
            resource_ids=resource_ids
        )
        return self.kb.db_connector.pipeline_search(params)

    def format_for_frontend(self, results: List[SearchResult]) -> List[Dict[str, Any]]:
        """Format results for frontend consumption."""
        formatted = []

        for result in results:
            formatted.append({
                "query": result.query,
                "text_blocks": result.text_blocks,  # Separate blocks for UI
                "relevance_score": result.relevance_score,
                "heading": result.heading,
                "subheading": result.subheading,
                "backtrack": result.backtrack,  # Navigation data
                "preview": " ".join(result.text_blocks)[:200] + "..."  # Combined preview
            })

        return formatted