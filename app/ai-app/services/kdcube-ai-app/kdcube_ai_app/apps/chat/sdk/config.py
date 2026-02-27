# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

# chatbot/sdk/config.py
from __future__ import annotations
import os
from pydantic import Field
from pydantic import BaseModel, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache

class CorsConfig(BaseModel):
    allow_origins: list[str] = Field(default_factory=lambda: ["*"])
    allow_methods: list[str] = Field(default_factory=lambda: ["*"])
    allow_headers: list[str] = Field(default_factory=lambda: ["*"])
    allow_credentials: bool = True

    @field_validator("allow_origins", "allow_methods", "allow_headers", mode="before")
    @classmethod
    def _coerce_list(cls, v):
        if v is None:
            return ["*"]
        if isinstance(v, str):
            if v.strip() == "*":
                return ["*"]
            return [s.strip() for s in v.split(",") if s.strip()]
        if isinstance(v, list):
            return v
        return [str(v)]


class Settings(BaseSettings):
    # API
    PORT: int = 8011
    CORS_CONFIG: str | None = None
    CORS_CONFIG_OBJ: CorsConfig | None = None

    OPENAI_API_KEY: str | None = None
    ANTHROPIC_API_KEY: str | None = None
    GOOGLE_API_KEY: str | None = Field(default=None, alias="GEMINI_API_KEY")

    # Postgres
    PGHOST: str = Field(default="localhost", alias="POSTGRES_HOST")
    PGPORT: int = Field(default=5434, alias="POSTGRES_PORT")
    PGDATABASE: str = Field(default="postgres", alias="POSTGRES_DATABASE")
    PGUSER: str = Field(default="postgres", alias="POSTGRES_USER")
    PGPASSWORD: str = Field(default="postgres", alias="POSTGRES_PASSWORD")
    PGSSL: bool = Field(default=False, alias="POSTGRES_SSL")

    # Neo4j
    NEO4J_URI: str = Field(default="bolt://neo4j:7687", alias="APP_NEO4J_URI")
    NEO4J_USER: str = Field(default="neo4j", alias="APP_NEO4J_USERNAME")
    NEO4J_PASSWORD: str = Field(default="neo4j", alias="APP_NEO4J_PASSWORD")
    APP_GRAPH_ENABLED: bool = Field(default=False, alias="APP_GRAPH_ENABLED")

    # S3
    AWS_REGION: str = "us-east-1"
    AWS_S3_BUCKET: str = "your-conv-bucket"
    AWS_PROFILE: str | None = None
    AWS_SHARED_CREDENTIALS_FILE: str | None = None
    AWS_CONFIG_FILE: str | None = None

    # Redis
    REDIS_URL: str = Field(default="redis://localhost:6379/0")
    REDIS_HOST: str = Field(default="localhost")
    REDIS_PORT: int = Field(default=6379)
    REDIS_PASSWORD: str | None = Field(default=None)
    REDIS_DB: int = Field(default=0)

    STORAGE_PATH: str | None = Field(default=None, alias="KDCUBE_STORAGE_PATH")

    TENANT: str = Field(default="home", alias="TENANT_ID")
    PROJECT: str = Field(default="default-project", alias="PROJECT_ID")
    INSTANCE_ID: str = Field(default="home-instance-1", alias="INSTANCE_ID")

    DEFAULT_MODEL_LLM: str | None = "claude-3-7-sonnet-20250219"

    # OPEX aggregation scheduler
    OPEX_AGG_CRON: str = Field(default="0 3 * * *")

    # Subscription rollover scheduler
    SUBSCRIPTION_ROLLOVER_ENABLED: bool = Field(default=True)
    SUBSCRIPTION_ROLLOVER_CRON: str = Field(default="15 * * * *")
    SUBSCRIPTION_ROLLOVER_LOCK_TTL_SECONDS: int = Field(default=900)
    SUBSCRIPTION_ROLLOVER_SWEEP_LIMIT: int = Field(default=500)

    # Bundle cleanup + ref tracking
    BUNDLE_CLEANUP_ENABLED: bool = Field(default=True)
    BUNDLE_CLEANUP_INTERVAL_SECONDS: int = Field(default=3600)
    BUNDLE_CLEANUP_LOCK_TTL_SECONDS: int = Field(default=900)
    BUNDLE_REF_TTL_SECONDS: int = Field(default=3600)
    # Force bundles registry to be overwritten from AGENTIC_BUNDLES_JSON at startup (processor only).
    BUNDLES_FORCE_ENV_ON_STARTUP: bool = Field(default=False)
    BUNDLES_FORCE_ENV_LOCK_TTL_SECONDS: int = Field(default=60)

    def model_post_init(self, __context) -> None:
        # Override tenant/project from GATEWAY_CONFIG_JSON if present.
        cfg_json = os.getenv("GATEWAY_CONFIG_JSON")
        if isinstance(cfg_json, str) and cfg_json.strip():
            try:
                import json
                cfg = json.loads(cfg_json)
                tenant = cfg.get("tenant_id") or cfg.get("tenant")
                project = cfg.get("project_id") or cfg.get("project")
                if tenant:
                    self.TENANT = tenant
                if project:
                    self.PROJECT = project
            except Exception:
                # Keep env-derived values on parse failure
                pass

        # Parse CORS_CONFIG JSON (if provided)
        if isinstance(self.CORS_CONFIG, str) and self.CORS_CONFIG.strip():
            try:
                import json
                self.CORS_CONFIG_OBJ = CorsConfig.model_validate(json.loads(self.CORS_CONFIG))
            except Exception:
                # Leave None on parse failure
                self.CORS_CONFIG_OBJ = None

        # If REDIS_URL is not explicitly set, build it from host/port/password/db.
        if not os.getenv("REDIS_URL"):
            auth = f":{self.REDIS_PASSWORD}@" if self.REDIS_PASSWORD else ""
            self.REDIS_URL = f"redis://{auth}{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"

@lru_cache()
def get_settings() -> Settings:
    return Settings()
