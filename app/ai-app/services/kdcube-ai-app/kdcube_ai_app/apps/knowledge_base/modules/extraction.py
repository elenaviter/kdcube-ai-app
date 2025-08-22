# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

# modules/extraction.py
"""
Extraction module for processing raw documents into structured content and assets.
"""
import os
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

from kdcube_ai_app.apps.knowledge_base.modules.base import ProcessingModule


class ExtractionModule(ProcessingModule):
    """Module responsible for extracting content and assets from raw documents."""

    @property
    def stage_name(self) -> str:
        return "extraction"

    async def process(self, resource_id: str, version: str, force_reprocess: bool = False, **kwargs) -> Union[Any, Dict[str, Any]]:
        """Extract content and assets from raw data."""
        self.logger.info(f"EXTRACTION. Extracting content and assets from raw data. {resource_id}.{version}")
        # Check if extraction already exists
        if not force_reprocess and self.is_processed(resource_id, version):
            self.logger.info(f"Extraction already exists for {resource_id} v{version}, skipping")
            return self.get_extraction_results(resource_id, version) or []

        self.logger.info(f"Extracting content from {resource_id} v{version}")

        # Get the data element for processing
        data_element = kwargs.get("data_element")

        if not data_element:
            raise ValueError(f"Cannot reconstruct data element for {resource_id}")

        # Create data source and extract
        data_source = data_element.to_data_source()
        extraction_results = data_source.extract()

        # Convert results to serializable format and store all assets
        results_data = []
        for i, result in enumerate(extraction_results):

            # Save main content
            content_filename = f"extraction_{i}.md"
            self.storage.save_stage_content(self.stage_name, resource_id, version, content_filename, result.content)

            # Store all associated assets if they exist
            assets_stored = self._store_extraction_assets(
                resource_id, version, i, result.metadata.get("assets", {})
            )

            # Create comprehensive result metadata
            result_dict = {
                "index": i,
                "content_file": content_filename,
                "rn": f"ef:{self.tenant}:{self.project}:knowledge_base:{self.stage_name}:{resource_id}:{version}:{content_filename}",
                "metadata": {
                    k: v for k, v in result.metadata.items()
                    if k != "assets"  # Don't include raw asset content in metadata
                },
                "assets": assets_stored,
                "extraction_timestamp": datetime.now().isoformat(),
                "total_files": 1 + sum(len(assets) for assets in assets_stored.values())
            }
            results_data.append(result_dict)

        # Save extraction metadata summary
        self.save_extraction_results(resource_id, version, results_data)

        # Log operation
        self.log_operation("extraction_complete", resource_id, {
            "version": version,
            "extraction_count": len(results_data),
            "total_files": sum(r["total_files"] for r in results_data)
        })

        self.logger.info(f"Extracted {len(results_data)} results with assets from {resource_id} v{version}")
        return results_data

    def _reconstruct_data_element(self, resource_id: str, version: str):
        """Reconstruct data element from stored metadata - this would be injected from core."""
        # This is a dependency that should be provided via kwargs in process()
        raise NotImplementedError("Data element should be provided via 'data_element' kwarg in process() method")

    def _store_extraction_assets(self, resource_id: str, version: str, extraction_index: int,
                                 assets: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
        """Store all extraction assets in the extraction stage."""

        stored_assets = {
            "images": [],
            "tables": [],
            "metadata": [],
            "other": []
        }

        # Store each category of assets
        for asset_type, asset_list in assets.items():
            if asset_type not in stored_assets:
                stored_assets[asset_type] = []

            for asset in asset_list:
                if "content" in asset and "storage_filename" in asset:
                    try:
                        # Store the asset content
                        self.storage.save_stage_content(
                            self.stage_name,
                            resource_id,
                            version,
                            asset["storage_filename"],
                            asset["content"]
                        )

                        # Create metadata entry without raw content
                        stored_asset_info = {
                            k: v for k, v in asset.items()
                            if k != "content"  # Exclude raw content from metadata
                        }
                        stored_asset_info["stored"] = True
                        stored_asset_info["storage_path"] = self.storage.get_stage_file_path(
                            self.stage_name, resource_id, version, asset["storage_filename"]
                        )
                        stored_asset_info["rn"] = f"ef:{self.tenant}:{self.project}:knowledge_base:{self.stage_name}:{resource_id}:{asset_type}:{version}:{asset['storage_filename']}"

                        stored_assets[asset_type].append(stored_asset_info)

                        self.logger.debug(f"Stored {asset_type} asset: {asset['storage_filename']}")

                    except Exception as e:
                        self.logger.error(f"Failed to store {asset_type} asset {asset.get('storage_filename', 'unknown')}: {e}")
                        # Still add to metadata but mark as failed
                        failed_asset_info = {
                            k: v for k, v in asset.items()
                            if k != "content"
                        }
                        failed_asset_info["stored"] = False
                        failed_asset_info["error"] = str(e)
                        stored_assets[asset_type].append(failed_asset_info)

        return stored_assets

    def get_extraction_results(self, resource_id: str, version: str) -> Optional[List[Dict[str, Any]]]:
        """Get extraction results from storage."""
        return self.storage.get_extraction_results(resource_id, version)

    def save_extraction_results(self, resource_id: str, version: str, results: List[Dict[str, Any]]) -> None:
        """Save extraction results to storage."""
        self.storage.save_extraction_results(resource_id, version, results)

    def get_extraction_content(self, resource_id: str, version: str, extraction_index: int = 0) -> Optional[str]:
        """Get the main content for a specific extraction."""
        filename = f"extraction_{extraction_index}.md"
        return self.storage.get_stage_content(self.stage_name, resource_id, version, filename, as_text=True)

    def get_asset(self, resource_id: str, version: str, asset_filename: str) -> Optional[bytes]:
        """Retrieve a specific extraction asset by filename."""
        try:
            return self.storage.get_stage_content(
                self.stage_name, resource_id, version, asset_filename, as_text=False
            )
        except Exception as e:
            self.logger.error(f"Failed to retrieve extraction asset {asset_filename}: {e}")
            return None

    def list_assets(self, resource_id: str, version: str) -> Dict[str, List[str]]:
        """List all extraction assets for a resource version."""
        # Get all files in the extraction stage
        all_files = self.storage.list_stage_files(self.stage_name, resource_id, version)

        # Categorize files
        categorized = {
            "content": [],
            "images": [],
            "tables": [],
            "metadata": [],
            "other": []
        }

        # Extensions for categorization
        image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.svg', '.webp'}
        table_extensions = {'.csv', '.tsv', '.xlsx'}
        metadata_extensions = {'.json', '.xml', '.yaml', '.yml'}

        for filename in all_files:
            ext = os.path.splitext(filename)[1].lower()

            if filename.endswith('.md'):
                categorized["content"].append(filename)
            elif ext in image_extensions:
                categorized["images"].append(filename)
            elif ext in table_extensions:
                categorized["tables"].append(filename)
            elif ext in metadata_extensions and not filename.startswith("extraction"):
                categorized["metadata"].append(filename)
            else:
                categorized["other"].append(filename)

        return categorized

    def get_asset_url(self, resource_id: str, version: str, asset_filename: str) -> Optional[str]:
        """Get the full URL/path to an extraction asset for external access."""
        try:
            return self.storage.get_stage_full_path(self.stage_name, resource_id, version, asset_filename)
        except Exception as e:
            self.logger.error(f"Failed to get URL for extraction asset {asset_filename}: {e}")
            return None

    def get_extraction_stats(self, resource_id: str, version: str) -> Dict[str, Any]:
        """Get statistics about the extraction results."""
        results = self.get_extraction_results(resource_id, version)
        if not results:
            return {}

        stats = {
            "extraction_count": len(results),
            "total_files": sum(r.get("total_files", 0) for r in results),
            "assets_by_type": {},
            "content_files": []
        }

        # Aggregate asset statistics
        for result in results:
            stats["content_files"].append(result.get("content_file"))

            assets = result.get("assets", {})
            for asset_type, asset_list in assets.items():
                if asset_type not in stats["assets_by_type"]:
                    stats["assets_by_type"][asset_type] = 0
                stats["assets_by_type"][asset_type] += len(asset_list)

        return stats

    def delete_extraction_asset(self, resource_id: str, version: str, asset_filename: str) -> bool:
        """Delete a specific extraction asset."""
        try:
            self.storage.delete_stage_content(self.stage_name, resource_id, version, asset_filename)
            self.logger.info(f"Deleted extraction asset: {asset_filename}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to delete extraction asset {asset_filename}: {e}")
            return False

    def reprocess_assets_only(self, resource_id: str, version: str) -> Dict[str, Any]:
        """Reprocess just the asset storage part of extraction."""
        # Get existing extraction results
        extraction_results = self.get_extraction_results(resource_id, version)
        if not extraction_results:
            raise ValueError(f"No existing extraction results found for {resource_id} v{version}")

        # Re-extract and re-store assets for each result
        updated_results = []
        for result in extraction_results:
            # Get the original extraction content
            content_file = result.get("content_file")
            if content_file:
                content = self.storage.get_stage_content(self.stage_name, resource_id, version, content_file, as_text=True)

                # Here you could re-run asset extraction logic if needed
                # For now, just maintain existing structure
                updated_results.append(result)

        return {"reprocessed_assets": len(updated_results)}