from typing import Any, Dict, Optional
import json, re

js = """
```json
{
  "phase1": [5],
  "phase2": {
    "5": [
      {
        "s": "„Ein Gespräch setzt voraus, dass der andere Recht haben könnte." – Jugend debattiert 2026 am CFG",
        "e": "Valentinstagsaktion beim Wuppertaler SV"
      },
      {
        "s": "Gartenschau 2031: Bergische Städte wollen bei Buga zusammenarbeiten",
        "e": "Bergische Wirtschaft: Axalta schließt seine Restrukturierung ab"
      }
    ]
  }
}
```
"""
def _fix_json_quotes(text: str) -> str:
    """Fix JSON by escaping internal quotes and replacing Unicode delimiters."""
    result = []
    in_string = False
    i = 0

    while i < len(text):
        c = text[i]
        prev_char = text[i-1] if i > 0 else ''

        if c == '"' and prev_char != '\\':
            if not in_string:
                # Opening quote
                in_string = True
                result.append(c)
            else:
                # Could be closing quote or content - look ahead
                next_chars = text[i+1:i+10].lstrip()
                if next_chars and next_chars[0] in ',}]:':
                    # Followed by delimiter - it's closing
                    in_string = False
                    result.append(c)
                else:
                    # It's content - escape it
                    result.append('\\' + c)
        elif c in '\u201C\u201D\u201E':
            # Unicode quotes - replace with ASCII
            if not in_string:
                result.append('"')
                in_string = True
            else:
                # Check if closing
                next_chars = text[i+1:i+10].lstrip()
                if next_chars and next_chars[0] in ',}]:':
                    result.append('"')
                    in_string = False
                else:
                    # Keep as content
                    result.append(c)
        else:
            result.append(c)

        i += 1

    return ''.join(result)

def _extract_json_block(text: str) -> Optional[str]:
    """Strip ```json fences and return the innermost {...} block."""
    if not text:
        return None
    t = text.strip()

    # Remove code fences
    if t.startswith("```"):
        t = re.sub(r"^```[ \t]*([jJ][sS][oO][nN])?[ \t]*\n?", "", t)
        t = re.sub(r"\n?```$", "", t).strip()

    # Find JSON block
    start = t.find("{")
    end = t.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    block = t[start:end + 1]
    block = _fix_json_quotes(block)

    return block

def _json_loads_loose(text: str):
    """Best-effort JSON loader that tolerates code fences and chatter."""
    # Try parsing as-is first
    try:
        return json.loads(text)
    except Exception:
        pass

    # Extract JSON block
    block = _extract_json_block(text)
    if not block:
        print("DEBUG: No block extracted")
        return None

    # Try parsing the block
    try:
        return json.loads(block)
    except Exception as e:
        print(f"JSON parse error: {e}")
        print(f"Block length: {len(block)}")

        start = 50
        end = 85
        snippet = block[start:end]
        print(f"\nCharacters {start}-{end}:")
        print(repr(snippet))
        print("Hex codes:")
        print([f"{ord(c):04x} ({c!r})" for c in snippet])

        # Show characters around position 131
        start = max(0, 125)
        end = min(len(block), 140)
        snippet = block[start:end]
        print(f"\nCharacters {start}-{end}:")
        print(repr(snippet))
        print("Hex codes:")
        print([f"{ord(c):04x} ({c!r})" for c in snippet])

        # Try removing trailing commas
        block_nc = re.sub(r",(\s*[}\]])", r"\1", block)
        try:
            return json.loads(block_nc)
        except Exception as e2:
            print(f"Still failed after comma removal: {e2}")
            return None

a = _json_loads_loose(js)
print(a)