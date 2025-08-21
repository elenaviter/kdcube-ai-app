# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

"""
TenantDB - Manages database schema operations for projects within a tenant

File: kdcube_ai_app/apps/knowledge_base/db/providers/tenant_db.py
"""

import logging
from typing import Optional, Dict, Any, List
from datetime import datetime

from kdcube_ai_app.ops.deployment.sql.db_deployment import (
    run as provision,
    SYSTEM_COMPONENT,
    SYSTEM_SCHEMA,
    PROJECT_COMPONENT
)

logger = logging.getLogger("TenantDB")


class TenantDB:
    """
    Manages database schema operations for projects within a tenant.

    This component is responsible for:
    - Creating project-specific database schemas
    - Deleting project database schemas
    - Managing system-level database components
    - Coordinating with TenantProjects for complete project lifecycle
    """

    def __init__(self, tenant_id: Optional[str] = None):
        """
        Initialize TenantDB.

        Args:
            tenant_id: Optional tenant identifier for logging and future multi-tenancy
        """
        self.tenant_id = tenant_id or "default"
        logger.info(f"Initialized TenantDB for tenant '{self.tenant_id}'")

    def provision_system_components(self) -> Dict[str, Any]:
        """
        Provision system-level database components.

        This should be called once during tenant initialization to ensure
        system-level database schemas and components are in place.

        Returns:
            Dictionary with operation results
        """
        try:
            logger.info(f"Provisioning system components for tenant '{self.tenant_id}'")

            # Provision system components
            result = provision("deploy", SYSTEM_COMPONENT)

            logger.info(f"Successfully provisioned system components for tenant '{self.tenant_id}'")

            return {
                "status": "success",
                "operation": "provision_system",
                "tenant_id": self.tenant_id,
                "timestamp": datetime.utcnow().isoformat(),
                "result": result
            }

        except Exception as e:
            logger.error(f"Failed to provision system components for tenant '{self.tenant_id}': {e}")
            return {
                "status": "error",
                "operation": "provision_system",
                "tenant_id": self.tenant_id,
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e)
            }

    def create_project_db(self, project_name: str, component_type: str = "knowledge_base") -> Dict[str, Any]:
        """
        Create database schema for a project.

        Args:
            project_name: Name of the project
            component_type: Type of component (default: "knowledge_base")

        Returns:
            Dictionary with operation results

        Raises:
            Exception: If database creation fails
        """
        try:
            logger.info(f"Creating database schema for project '{project_name}' (tenant: '{self.tenant_id}')")

            # Validate project name for database compatibility
            self._validate_project_name_for_db(project_name)

            project_name_effective = project_name.replace(" ", "_").lower()
            # Provision project database components
            result = provision("deploy", PROJECT_COMPONENT, tenant=self.tenant_id, project=project_name, app=component_type)

            logger.info(f"Successfully created database schema for project '{project_name}'")

            return {
                "status": "success",
                "operation": "create_project_db",
                "project_name": project_name,
                "component_type": component_type,
                "tenant_id": self.tenant_id,
                "timestamp": datetime.utcnow().isoformat(),
                "result": result
            }

        except Exception as e:
            logger.error(f"Failed to create database schema for project '{project_name}': {e}")
            raise Exception(f"Database creation failed for project '{project_name}': {str(e)}")

    def delete_project_db(self, project_name: str, component_type: str = "knowledge_base") -> Dict[str, Any]:
        """
        Delete database schema for a project.

        Args:
            project_name: Name of the project
            component_type: Type of component (default: "knowledge_base")

        Returns:
            Dictionary with operation results

        Raises:
            Exception: If database deletion fails
        """
        try:
            logger.info(f"Deleting database schema for project '{project_name}' (tenant: '{self.tenant_id}')")

            # Delete project database components
            result = provision("delete", PROJECT_COMPONENT, tenant=self.tenant_id, project=project_name, app=component_type)

            logger.info(f"Successfully deleted database schema for project '{project_name}'")

            return {
                "status": "success",
                "operation": "delete_project_db",
                "project_name": project_name,
                "component_type": component_type,
                "tenant_id": self.tenant_id,
                "timestamp": datetime.utcnow().isoformat(),
                "result": result
            }

        except Exception as e:
            logger.error(f"Failed to delete database schema for project '{project_name}': {e}")
            raise Exception(f"Database deletion failed for project '{project_name}': {str(e)}")

    def project_db_exists(self, project_name: str, component_type: str = "knowledge_base") -> bool:
        """
        Check if database schema exists for a project.

        Args:
            project_name: Name of the project
            component_type: Type of component (default: "knowledge_base")

        Returns:
            True if database schema exists, False otherwise
        """
        try:
            # This would need to be implemented based on your db_deployment module
            # For now, we'll assume it exists if no exception is thrown during a query operation

            # You might want to add a "check" operation to your provision function
            # result = provision("check", PROJECT_COMPONENT, self.tenant_id, project_name, component_type)

            # For now, we'll implement a basic check (you might need to adjust this)
            logger.debug(f"Checking if database schema exists for project '{project_name}'")

            # This is a placeholder - you'll need to implement the actual check
            # based on your database setup and db_deployment module capabilities
            return True  # Placeholder

        except Exception as e:
            logger.warning(f"Error checking database existence for project '{project_name}': {e}")
            return False

    def list_project_dbs(self) -> List[Dict[str, Any]]:
        """
        List all project databases for this tenant.

        Returns:
            List of dictionaries with project database information
        """
        try:
            logger.debug(f"Listing project databases for tenant '{self.tenant_id}'")

            # This would need to be implemented based on your db_deployment module
            # You might need to add a "list" operation to your provision function
            # result = provision("list", PROJECT_COMPONENT)

            # For now, return empty list as placeholder
            # You'll need to implement this based on your database setup
            return []

        except Exception as e:
            logger.error(f"Error listing project databases for tenant '{self.tenant_id}': {e}")
            return []

    def cleanup_orphaned_dbs(self, existing_projects: List[str]) -> Dict[str, Any]:
        """
        Clean up database schemas for projects that no longer exist in storage.

        Args:
            existing_projects: List of project names that exist in storage

        Returns:
            Dictionary with cleanup results
        """
        try:
            logger.info(f"Cleaning up orphaned databases for tenant '{self.tenant_id}'")

            # Get list of project databases
            project_dbs = self.list_project_dbs()

            cleanup_results = []
            errors = []

            for db_info in project_dbs:
                project_name = db_info.get("project_name")
                if project_name and project_name not in existing_projects:
                    logger.info(f"Found orphaned database for project '{project_name}', deleting...")
                    try:
                        result = self.delete_project_db(project_name)
                        cleanup_results.append(result)
                    except Exception as e:
                        error_info = {
                            "project_name": project_name,
                            "error": str(e)
                        }
                        errors.append(error_info)
                        logger.error(f"Failed to cleanup orphaned database for '{project_name}': {e}")

            return {
                "status": "success",
                "operation": "cleanup_orphaned_dbs",
                "tenant_id": self.tenant_id,
                "timestamp": datetime.utcnow().isoformat(),
                "cleaned_up": len(cleanup_results),
                "errors": len(errors),
                "cleanup_results": cleanup_results,
                "cleanup_errors": errors
            }

        except Exception as e:
            logger.error(f"Error during database cleanup for tenant '{self.tenant_id}': {e}")
            return {
                "status": "error",
                "operation": "cleanup_orphaned_dbs",
                "tenant_id": self.tenant_id,
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e)
            }

    def get_db_health(self) -> Dict[str, Any]:
        """
        Get health status of the database components.

        Returns:
            Dictionary with database health information
        """
        try:
            # This would check database connectivity and component status
            # You might want to add a "health" operation to your provision function

            health_info = {
                "status": "healthy",
                "tenant_id": self.tenant_id,
                "timestamp": datetime.utcnow().isoformat(),
                "system_components": "available",  # Check system components
                "connectivity": "ok"  # Check database connectivity
            }

            return health_info

        except Exception as e:
            logger.error(f"Error checking database health for tenant '{self.tenant_id}': {e}")
            return {
                "status": "unhealthy",
                "tenant_id": self.tenant_id,
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e)
            }

    def _validate_project_name_for_db(self, project_name: str) -> None:
        """
        Validate project name for database compatibility.

        Args:
            project_name: Project name to validate

        Raises:
            ValueError: If project name is not suitable for database use
        """
        if not project_name:
            raise ValueError("Project name cannot be empty")

        # Check for SQL injection patterns or invalid characters
        import re
        if not re.match(r'^[a-zA-Z0-9_-]+$', project_name):
            raise ValueError(
                "Project name can only contain letters, numbers, hyphens, and underscores (database safe)"
            )

        # Check length constraints (database identifiers usually have limits)
        if len(project_name) > 63:  # PostgreSQL identifier limit
            raise ValueError("Project name too long for database identifier (max 63 characters)")

        # Check for reserved words (you might want to expand this list)
        reserved_words = {
            'select', 'insert', 'update', 'delete', 'create', 'drop', 'alter',
            'table', 'index', 'view', 'schema', 'database', 'user', 'role',
            'system', 'admin', 'root', 'public'
        }

        if project_name.lower() in reserved_words:
            raise ValueError(f"Project name '{project_name}' is a reserved word")


class TenantDBError(Exception):
    """Custom exception for TenantDB operations."""

    def __init__(self, message: str, operation: str = None, project_name: str = None):
        super().__init__(message)
        self.operation = operation
        self.project_name = project_name


# ================================================================================
#                          TESTING HELPERS
# ================================================================================

def try_tenant_db_operations(tenant_id: str = "test", test_project: str = "test-project"):
    """
    Test function for TenantDB operations.

    WARNING: This creates and deletes actual database schemas!
    Only use in test environments.

    Args:
        tenant_id: Test tenant ID
        test_project: Test project name
    """
    logger.warning(f"Running TenantDB test operations for tenant '{tenant_id}'")

    tenant_db = TenantDB(tenant_id)

    try:
        # Test system provisioning
        logger.info("Testing system provisioning...")
        system_result = tenant_db.provision_system_components()
        logger.info(f"System provisioning result: {system_result}")

        # Test project creation
        logger.info(f"Testing project creation for '{test_project}'...")
        create_result = tenant_db.create_project_db(test_project)
        logger.info(f"Project creation result: {create_result}")

        # Test project existence check
        logger.info(f"Testing project existence check for '{test_project}'...")
        exists = tenant_db.project_db_exists(test_project)
        logger.info(f"Project exists: {exists}")

        # Test project deletion
        logger.info(f"Testing project deletion for '{test_project}'...")
        delete_result = tenant_db.delete_project_db(test_project)
        logger.info(f"Project deletion result: {delete_result}")

        logger.info("TenantDB test operations completed successfully")

    except Exception as e:
        logger.error(f"TenantDB test operations failed: {e}")
        raise