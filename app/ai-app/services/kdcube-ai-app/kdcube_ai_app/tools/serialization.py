# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

import base64
from datetime import datetime
from typing import Any, Dict

def json_safe(x: Any) -> Any:
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    if isinstance(x, datetime):
        return x.isoformat()
    if isinstance(x, (bytes, bytearray)):
        return base64.b64encode(bytes(x)).decode("ascii")
    if isinstance(x, dict):
        return {str(k): json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [json_safe(v) for v in x]
    # last resort
    return repr(x)