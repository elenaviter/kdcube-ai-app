import os
from kdcube_ai_app.apps.chat.sdk.config import get_settings

# Redis connection settings
REDIS_URL = get_settings().REDIS_URL

INSTANCE_ID = os.environ.get("INSTANCE_ID", "home-instance-1")
HEARTBEAT_INTERVAL = int(os.environ.get("HEARTBEAT_INTERVAL", 10))  # seconds
TENANT_ID = os.environ.get("TENANT_ID", "home")
PROJECT_ID = os.environ.get("DEFAULT_PROJECT_NAME", "home")
