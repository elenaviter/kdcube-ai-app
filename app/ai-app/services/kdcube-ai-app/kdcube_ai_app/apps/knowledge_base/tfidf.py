import json
import re
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import pickle

from kdcube_ai_app.apps.knowledge_base.storage import KnowledgeBaseStorage


class TfIdfService:
    """
    Standalone TF-IDF service for building and querying TF-IDF vectors across multiple datasources.

    This service can:
    - Build TF-IDF models from segments across multiple resources
    - Query for similar segments using cosine similarity
    - Save/load models for persistence
    - Handle metadata tokenization for TF-IDF computation
    """

    def __init__(self,
                 storage: KnowledgeBaseStorage,
                 project: str):
        """
        Initialize the TF-IDF service.

        Args:
            storage: Knowledge base storage instance
            project: Project identifier
        """
        self.storage = storage
        self.project = project
        self.logger = logging.getLogger(f"KnowledgeBase.TfIdfService")

        # TF-IDF components
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix: Optional[csr_matrix] = None
        self.segment_index: List[Dict[str, str]] = []  # Maps matrix rows to segments
        self.model_metadata: Dict[str, Any] = {}

    def metadata_to_token_string(self, metadata_json_str: str) -> str:
        """
        Convert a JSON string of metadata into a space-separated string of tokens.
        Each token will be in the form "key:value".

        Args:
            metadata_json_str: JSON string containing metadata list

        Returns:
            Space-separated string of key:value tokens
        """
        try:
            metadata = json.loads(metadata_json_str) if isinstance(metadata_json_str, str) else metadata_json_str
        except Exception as e:
            self.logger.warning(f"Error parsing metadata: {e}")
            return ""

        tokens = []
        # Ensure metadata is a list
        if not isinstance(metadata, list):
            metadata = [metadata]

        for item in metadata:
            if isinstance(item, dict):
                key = item.get("key", "").strip()
                value = item.get("value", "").strip()
                if key and value:
                    # Replace spaces in the value with underscores
                    clean_value = "_".join(value.split())
                    tokens.append(f"{key}:{clean_value}")
                elif value:
                    clean_value = "_".join(value.split())
                    tokens.append(f"misc:{clean_value}")
            else:
                # If the item is not a dict, convert to string and prefix with misc:
                token = f"misc:{str(item).strip()}"
                tokens.append(token)

        # Filter tokens using regex pattern
        pattern = re.compile(r"^[\w]+:[\w_]+$")
        valid_tokens = [token for token in tokens if pattern.match(token)]
        return " ".join(valid_tokens)

    def build_tfidf_model(self,
                          resource_filters: Optional[List[str]] = None,
                          include_text: bool = True,
                          include_metadata: bool = True,
                          min_df: int = 1,
                          max_df: float = 1.0,
                          max_features: Optional[int] = None) -> Dict[str, Any]:
        """
        Build a TF-IDF model from segments across specified resources.

        Args:
            resource_filters: List of resource IDs to include. If None, includes all resources
            include_text: Whether to include segment text content
            include_metadata: Whether to include metadata tokens
            min_df: Minimum document frequency for TF-IDF
            max_df: Maximum document frequency for TF-IDF
            max_features: Maximum number of features to keep

        Returns:
            Dictionary with model statistics and metadata
        """
        self.logger.info("Building TF-IDF model...")

        # Collect all segments
        all_segments = []
        segment_index = []

        # Get all resources if no filter specified
        if resource_filters is None:
            # You'll need to implement get_all_resources() in your storage
            resource_filters = self.storage.get_all_resource_ids()

        for resource_id in resource_filters:
            # Get latest version for each resource
            versions = self.storage.get_resource_versions(resource_id)
            if not versions:
                continue

            latest_version = max(versions)
            segments = self.storage.get_segments(resource_id, latest_version)

            for segment in segments:
                # Build document text
                doc_parts = []

                if include_text and segment.get("content"):
                    doc_parts.append(segment["content"])

                if include_metadata and segment.get("metadata"):
                    metadata_tokens = self.metadata_to_token_string(segment["metadata"])
                    if metadata_tokens:
                        doc_parts.append(metadata_tokens)

                if doc_parts:
                    document = " ".join(doc_parts)
                    all_segments.append(document)
                    segment_index.append({
                        "resource_id": resource_id,
                        "version": latest_version,
                        "segment_id": segment.get("id"),
                        "title": segment.get("title", "")
                    })

        if not all_segments:
            raise ValueError("No segments found for TF-IDF model building")

        # Build TF-IDF vectorizer
        token_pattern = r"(?u)\b[\w:]+\b" if include_metadata else None

        self.vectorizer = TfidfVectorizer(
            min_df=min_df,
            max_df=max_df,
            max_features=max_features,
            token_pattern=token_pattern,
            lowercase=True,
            stop_words='english'
        )

        # Fit and transform
        self.tfidf_matrix = self.vectorizer.fit_transform(all_segments)
        self.segment_index = segment_index

        # Store metadata
        self.model_metadata = {
            "created_at": datetime.now().isoformat(),
            "resource_count": len(resource_filters),
            "segment_count": len(all_segments),
            "vocabulary_size": len(self.vectorizer.get_feature_names_out()),
            "include_text": include_text,
            "include_metadata": include_metadata,
            "min_df": min_df,
            "max_df": max_df,
            "max_features": max_features,
            "resources": resource_filters
        }

        self.logger.info(f"Built TF-IDF model with {self.model_metadata['segment_count']} segments "
                         f"and {self.model_metadata['vocabulary_size']} features")

        return self.model_metadata.copy()

    def find_similar_segments(self,
                              query_text: str,
                              query_metadata: Optional[str] = None,
                              top_k: int = 10,
                              min_similarity: float = 0.1) -> List[Dict[str, Any]]:
        """
        Find segments similar to the query using cosine similarity.

        Args:
            query_text: Text content to search for
            query_metadata: Optional metadata JSON string
            top_k: Number of top results to return
            min_similarity: Minimum similarity score threshold

        Returns:
            List of similar segments with similarity scores
        """
        if self.vectorizer is None or self.tfidf_matrix is None:
            raise ValueError("TF-IDF model not built. Call build_tfidf_model() first.")

        # Build query document
        query_parts = []
        if query_text:
            query_parts.append(query_text)
        if query_metadata:
            metadata_tokens = self.metadata_to_token_string(query_metadata)
            if metadata_tokens:
                query_parts.append(metadata_tokens)

        if not query_parts:
            raise ValueError("No valid query content provided")

        query_doc = " ".join(query_parts)

        # Transform query
        query_vector = self.vectorizer.transform([query_doc])

        # Calculate similarities
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()

        # Get top results
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            similarity = similarities[idx]
            if similarity < min_similarity:
                break

            segment_info = self.segment_index[idx].copy()
            segment_info["similarity_score"] = float(similarity)
            results.append(segment_info)

        return results

    def get_segment_vector(self, segment_idx: int) -> np.ndarray:
        """
        Get the TF-IDF vector for a specific segment.

        Args:
            segment_idx: Index of the segment in the model

        Returns:
            TF-IDF vector as numpy array
        """
        if self.tfidf_matrix is None:
            raise ValueError("TF-IDF model not built")

        return self.tfidf_matrix[segment_idx].toarray().flatten()

    def get_feature_names(self) -> List[str]:
        """Get the feature names (vocabulary) from the TF-IDF model."""
        if self.vectorizer is None:
            raise ValueError("TF-IDF model not built")

        return self.vectorizer.get_feature_names_out().tolist()

    def get_top_features_for_segment(self, segment_idx: int, top_n: int = 10) -> List[Tuple[str, float]]:
        """
        Get the top TF-IDF features for a specific segment.

        Args:
            segment_idx: Index of the segment
            top_n: Number of top features to return

        Returns:
            List of (feature_name, tfidf_score) tuples
        """
        if self.vectorizer is None or self.tfidf_matrix is None:
            raise ValueError("TF-IDF model not built")

        feature_names = self.vectorizer.get_feature_names_out()
        segment_vector = self.tfidf_matrix[segment_idx].toarray().flatten()

        # Get top features
        top_indices = np.argsort(segment_vector)[::-1][:top_n]

        return [(feature_names[idx], segment_vector[idx])
                for idx in top_indices if segment_vector[idx] > 0]

    def save_model(self, model_name: str) -> str:
        """
        Save the TF-IDF model to storage.

        Args:
            model_name: Name for the saved model

        Returns:
            Storage path where model was saved
        """
        if self.vectorizer is None or self.tfidf_matrix is None:
            raise ValueError("No model to save")

        model_data = {
            "vectorizer": pickle.dumps(self.vectorizer),
            "tfidf_matrix": pickle.dumps(self.tfidf_matrix),
            "segment_index": self.segment_index,
            "metadata": self.model_metadata
        }

        # Save to storage
        storage_path = f"tfidf_models/{model_name}"
        content = pickle.dumps(model_data)

        # You'll need to implement a generic storage method
        self.storage.save_blob(storage_path, content)

        self.logger.info(f"Saved TF-IDF model to {storage_path}")
        return storage_path

    def load_model(self, model_name: str) -> Dict[str, Any]:
        """
        Load a TF-IDF model from storage.

        Args:
            model_name: Name of the model to load

        Returns:
            Model metadata
        """
        storage_path = f"tfidf_models/{model_name}"

        try:
            content = self.storage.get_blob(storage_path)
            model_data = pickle.loads(content)

            self.vectorizer = pickle.loads(model_data["vectorizer"])
            self.tfidf_matrix = pickle.loads(model_data["tfidf_matrix"])
            self.segment_index = model_data["segment_index"]
            self.model_metadata = model_data["metadata"]

            self.logger.info(f"Loaded TF-IDF model from {storage_path}")
            return self.model_metadata.copy()

        except Exception as e:
            self.logger.error(f"Failed to load model {model_name}: {e}")
            raise

    def get_model_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the currently loaded model."""
        return self.model_metadata.copy() if self.model_metadata else None

    def clear_model(self):
        """Clear the currently loaded model from memory."""
        self.vectorizer = None
        self.tfidf_matrix = None
        self.segment_index = []
        self.model_metadata = {}
        self.logger.info("Cleared TF-IDF model from memory")


# Usage example
def example_usage():
    """Example of how to use the TF-IDF service."""

    # Initialize service
    storage = KnowledgeBaseStorage(...)  # Your storage instance
    tfidf_service = TfIdfService(storage, "my_project")

    # Build model from specific resources
    model_info = tfidf_service.build_tfidf_model(
        resource_filters=["doc1", "doc2", "doc3"],
        include_text=True,
        include_metadata=True,
        min_df=2,
        max_features=10000
    )
    print(f"Built model with {model_info['segment_count']} segments")

    # Find similar segments
    similar = tfidf_service.find_similar_segments(
        query_text="data management best practices",
        query_metadata='[{"key": "domain", "value": "data management"}]',
        top_k=5
    )

    for result in similar:
        print(f"Resource: {result['resource_id']}, Similarity: {result['similarity_score']:.3f}")

    # Save model for later use
    tfidf_service.save_model("knowledge_base_v1")

    # Load model later
    tfidf_service.load_model("knowledge_base_v1")