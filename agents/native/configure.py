#!/usr/bin/env python3
"""Create this native agent example's local descriptors and services config."""

import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
SDK_ROOT = HERE.parents[1] / "app" / "ai-app" / "src" / "kdcube-ai-app"
if str(SDK_ROOT) not in sys.path:
    sys.path.insert(0, str(SDK_ROOT))

from kdcube_ai_app.apps.chat.sdk.runtime.direct_hosting.local_setup import main


if __name__ == "__main__":
    main(root=HERE, default_provider="openai")
