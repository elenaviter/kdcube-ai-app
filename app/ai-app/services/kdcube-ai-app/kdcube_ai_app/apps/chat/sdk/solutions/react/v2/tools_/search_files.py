from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, Iterator, List, Optional

MAX_SCANNED_FILES = 2000


def _iter_files(root: str) -> Iterator[str]:
    count = 0
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        # Skip hidden dirs
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        dirnames.sort()
        filenames = sorted(f for f in filenames if not f.startswith("."))
        for fname in filenames:
            yield os.path.join(dirpath, fname)
            count += 1
            if count >= MAX_SCANNED_FILES:
                return


def search_files(
        *,
        root: str,
        name_regex: Optional[str] = None,
        content_regex: Optional[str] = None,
        max_bytes: int = 1_000_000,
        max_hits: int = 200,
) -> List[Dict[str, object]]:
    """
    Safe file search under a root directory.
    - no symlink following
    - hidden files/dirs skipped
    - file size capped per file
    - traversal capped by an internal safety limit
    """
    if not root:
        return []
    root_path = Path(root)
    name_re = re.compile(name_regex) if name_regex else None
    content_re = re.compile(content_regex) if content_regex else None
    hits: List[Dict[str, object]] = []
    for path in _iter_files(root):
        try:
            size = os.path.getsize(path)
        except Exception:
            continue
        if name_re and not name_re.search(os.path.basename(path)):
            continue
        if content_re:
            if size > max_bytes:
                continue
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read(max_bytes)
            except Exception:
                continue
            if not content_re.search(text):
                continue
        try:
            rel_path = Path(path).relative_to(root_path).as_posix()
        except Exception:
            rel_path = os.path.relpath(path, root).replace(os.sep, "/")
        hits.append({
            "path": rel_path,
            "size_bytes": int(size),
        })
        if len(hits) >= max_hits:
            break
    return hits
