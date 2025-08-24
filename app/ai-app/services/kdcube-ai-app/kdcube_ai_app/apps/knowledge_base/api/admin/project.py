# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

"""
Updated Project API using TenantProjects component

File: api/admin/project.py
"""
from datetime import datetime
from fastapi import Request, Depends, APIRouter, HTTPException
from typing import Optional, Callable, List, Dict, Any
import json
import traceback
import logging

from kdcube_ai_app.apps.knowledge_base.api.resolvers import (
    get_project,
    get_tenant_projects,
    get_tenant_db,
    ENABLE_DATABASE, get_system_health, get_kb_admin_with_acct_dep
)
from kdcube_ai_app.apps.knowledge_base.tenant import TenantProjects
from kdcube_ai_app.apps.knowledge_base.db.providers.tenant_db import TenantDB
from kdcube_ai_app.auth.sessions import UserSession

logger = logging.getLogger("KBAdmin.project.API")

# Create router
router = APIRouter()

# Store TenantProjects reference in router for access in partial functions
router.tenant_projects = None

# ================================================================================
#                          PROJECT MANAGEMENT ENDPOINTS
# ================================================================================

@router.post("/projects")
async def create_project(
    request: Request,
    tenant_projects: TenantProjects = Depends(get_tenant_projects),
        session: UserSession = Depends(get_kb_admin_with_acct_dep()),
):
    """Create a new knowledge base project."""
    data = await request.json()

    if not data or "name" not in data:
        raise HTTPException(status_code=400, detail="Missing project name")

    project_name = data["name"].strip()
    description = data.get("description", "").strip()
    created_by = data.get("created_by", "<user>")

    try:
        # Use TenantProjects singleton to create the project
        project_metadata = tenant_projects.create_project(
            project_name=project_name,
            description=description,
            created_by=created_by
        )

        logger.info(f"Created project '{project_name}' with description: {description}")

        return {
            "status": "success",
            "project": project_metadata.to_dict(),
            "message": f"Project '{project_name}' created successfully",
            "database_enabled": ENABLE_DATABASE
        }

    except ValueError as e:
        # Handle validation errors
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to create project '{project_name}': {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to create project: {str(e)}")


@router.get("/projects")
async def list_projects(
        session: UserSession = Depends(get_kb_admin_with_acct_dep()),
        tenant_projects: TenantProjects = Depends(get_tenant_projects)
):
    """List all knowledge base projects using TenantProjects singleton."""
    try:
        # Use TenantProjects singleton to list projects
        projects_metadata = tenant_projects.list_projects()

        # Convert to API response format and add stats
        projects = []
        for project_meta in projects_metadata:
            project_dict = project_meta.to_dict()

            # Add runtime stats if available
            try:
                stats = tenant_projects.get_project_stats(project_meta.name)
                if stats:
                    project_dict["stats"] = stats
            except Exception as e:
                logger.warning(f"Could not get stats for project {project_meta.name}: {e}")
                project_dict["stats"] = {"error": str(e)}

            projects.append(project_dict)

        return {
            "status": "success",
            "projects": projects,
            "total_projects": len(projects),
            "tenant_id": tenant_projects.tenant_id,
            "database_enabled": ENABLE_DATABASE
        }

    except Exception as e:
        logger.error(f"Failed to list projects: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list projects: {str(e)}")


@router.get("/projects/{project}")
async def get_project_info(
    project: str = Depends(get_project),
    tenant_projects: TenantProjects = Depends(get_tenant_projects),
        session: UserSession = Depends(get_kb_admin_with_acct_dep()),
):
    """Get detailed information about a specific project."""
    try:
        # Get project metadata
        project_metadata = tenant_projects.get_project_metadata(project)
        if not project_metadata:
            raise HTTPException(status_code=404, detail=f"Project '{project}' not found")

        project_dict = project_metadata.to_dict()

        # Add detailed stats and resources
        try:
            kb = tenant_projects.create_kb_for_project(project)
            if kb:
                project_dict["stats"] = kb.get_stats()
                project_dict["resources"] = [
                    resource.model_dump() for resource in kb.list_resources()
                ]

                # Add content index stats if available
                try:
                    project_dict["content_index_stats"] = kb.get_content_index_stats()
                except Exception as e:
                    logger.warning(f"Could not get content index stats for {project}: {e}")
                    project_dict["content_index_stats"] = {"error": str(e)}
            else:
                logger.warning(f"Could not create KB instance for project {project}")
                project_dict["stats"] = {"error": "Could not create KB instance"}
                project_dict["resources"] = []

        except Exception as e:
            logger.warning(f"Error getting KB data for project {project}: {e}")
            project_dict["stats"] = {"error": str(e)}
            project_dict["resources"] = []

        return {
            "status": "success",
            "project": project_dict,
            "database_enabled": ENABLE_DATABASE
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get project '{project}': {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get project: {str(e)}")


@router.put("/projects/{project}")
async def update_project(
    request: Request,
    project: str = Depends(get_project),
    tenant_projects: TenantProjects = Depends(get_tenant_projects),
        session: UserSession = Depends(get_kb_admin_with_acct_dep()),
):
    """Update project metadata (description, etc.)."""
    try:
        data = await request.json()
        updated_by = data.get("updated_by", "<user>")

        # Use TenantProjects singleton to update the project
        updated_metadata = tenant_projects.update_project(
            project_name=project,
            updates=data,
            updated_by=updated_by
        )

        if not updated_metadata:
            raise HTTPException(status_code=404, detail=f"Project '{project}' not found")

        logger.info(f"Updated project '{project}' metadata")

        return {
            "status": "success",
            "project": updated_metadata.to_dict(),
            "message": f"Project '{project}' updated successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update project '{project}': {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update project: {str(e)}")


@router.delete("/projects/{project}")
async def delete_project(
    project: str = Depends(get_project),
    tenant_projects: TenantProjects = Depends(get_tenant_projects),
        session: UserSession = Depends(get_kb_admin_with_acct_dep()),
):
    """Delete a project and all its data."""
    if not router.tenant_projects:
        raise HTTPException(status_code=500, detail="TenantProjects not configured")

    try:
        # Get project stats before deletion for logging
        stats = {}
        try:
            stats = tenant_projects.get_project_stats(project)
        except Exception as e:
            logger.warning(f"Could not get stats before deletion: {e}")
            stats = {"error": "Could not get stats"}

        # Use TenantProjects singleton to delete the project
        success = tenant_projects.delete_project(project)

        if not success:
            raise HTTPException(status_code=404, detail=f"Project '{project}' not found or deletion failed")

        logger.info(f"Deleted project '{project}' with stats: {stats}")

        return {
            "status": "success",
            "message": f"Project '{project}' deleted successfully",
            "deleted_stats": stats
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete project '{project}': {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete project: {str(e)}")


# ================================================================================
#                          ADMIN ENDPOINTS
# ================================================================================

@router.get("/admin/tenant/health")
async def get_tenant_health(
    tenant_projects: TenantProjects = Depends(get_tenant_projects),
        session: UserSession = Depends(get_kb_admin_with_acct_dep()),
):
    """Get detailed tenant health including storage and database."""
    try:
        return tenant_projects.get_tenant_health()
    except Exception as e:
        logger.error(f"Tenant health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Tenant health check failed: {str(e)}")


@router.get("/admin/system/health")
async def get_system_health_endpoint(session: UserSession = Depends(get_kb_admin_with_acct_dep()),):
    """Get comprehensive system health."""
    try:
        return get_system_health()
    except Exception as e:
        logger.error(f"System health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"System health check failed: {str(e)}")


@router.post("/admin/tenant/cleanup")
async def cleanup_orphaned_resources(
    tenant_projects: TenantProjects = Depends(get_tenant_projects),
        session: UserSession = Depends(get_kb_admin_with_acct_dep()),
):
    """Clean up orphaned database schemas and storage inconsistencies."""
    try:
        return tenant_projects.cleanup_orphaned_resources()
    except Exception as e:
        logger.error(f"Orphaned resource cleanup failed: {e}")
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")


@router.get("/admin/projects/{project}/validate")
async def validate_project_consistency(
    project: str = Depends(get_project),
    tenant_projects: TenantProjects = Depends(get_tenant_projects),
        session: UserSession = Depends(get_kb_admin_with_acct_dep()),
):
    """Validate consistency between storage and database for a project."""
    try:
        return tenant_projects.validate_project_consistency(project)
    except Exception as e:
        logger.error(f"Project validation failed for '{project}': {e}")
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")


@router.post("/admin/tenant/provision-system")
async def provision_system_components(
    tenant_projects: TenantProjects = Depends(get_tenant_projects),
        session: UserSession = Depends(get_kb_admin_with_acct_dep()),
):
    """Provision system-level database components."""
    try:
        return tenant_projects.provision_system_components()
    except Exception as e:
        logger.error(f"System provisioning failed: {e}")
        raise HTTPException(status_code=500, detail=f"System provisioning failed: {str(e)}")


# ================================================================================
#                          DATABASE-SPECIFIC ADMIN ENDPOINTS (Optional)
# ================================================================================

@router.get("/admin/database/health")
async def get_database_health(session: UserSession = Depends(get_kb_admin_with_acct_dep()),):
    """Get database health (only available if database support is enabled)."""
    if not ENABLE_DATABASE:
        raise HTTPException(status_code=404, detail="Database support is disabled")

    try:
        tenant_db = get_tenant_db()
        return tenant_db.get_db_health()
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Database health check failed: {str(e)}")


@router.post("/admin/database/provision-system")
async def provision_database_system(session: UserSession = Depends(get_kb_admin_with_acct_dep()),):
    """Provision system-level database components directly."""
    if not ENABLE_DATABASE:
        raise HTTPException(status_code=404, detail="Database support is disabled")

    try:
        tenant_db = get_tenant_db()
        return tenant_db.provision_system_components()
    except Exception as e:
        logger.error(f"Database system provisioning failed: {e}")
        raise HTTPException(status_code=500, detail=f"Database system provisioning failed: {str(e)}")


# ================================================================================
#                          SIMPLE MOUNT FUNCTION
# ================================================================================

def mount_project_routes(app):
    """
    Simple mount function that just adds the router.
    All singletons are managed in resolvers.py
    """
    app.include_router(router, prefix="/api/kb", tags=["projects"])
    logger.info("Mounted project management routes")
