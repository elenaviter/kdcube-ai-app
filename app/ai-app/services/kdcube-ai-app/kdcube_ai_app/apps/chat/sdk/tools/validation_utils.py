# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

# chat/sdk/tools/validation_utils.py

import json, re
from html.parser import HTMLParser
import yaml

def _valid_json(s: str) -> bool:
    try: json.loads(s); return True
    except Exception: return False

def _valid_yaml(s: str) -> bool:
    try:
        # accept multi-doc; ensure at least one document
        docs = list(yaml.safe_load_all(s))
        return len(docs) >= 1
    except Exception:
        return False

class _HTMLCloseChecker(HTMLParser):
    def __init__(self): super().__init__(); self.stack=[]
    def handle_starttag(self, tag, attrs): self.stack.append(tag)
    def handle_endtag(self, tag):
        # naive, but catches obvious truncations
        if self.stack and self.stack[-1]==tag: self.stack.pop()
def _valid_html(s: str) -> bool:
    try:
        p=_HTMLCloseChecker(); p.feed(s)
        return len(p.stack)==0 and ("</html>" in s.lower())
    except Exception:
        return False

def _valid_markdown(s: str) -> bool:
    # minimal: non-empty + balanced code fences if any
    if not s.strip(): return False
    fences = len(re.findall(r"^```", s, flags=re.M))
    return (fences % 2)==0

VALIDATORS = {
    "json": _valid_json,
    "yaml": _valid_yaml,
    "html": _valid_html,
    "markdown": _valid_markdown,
    "text": lambda s: bool(s.strip()),
}
