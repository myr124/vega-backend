#!/usr/bin/env python3
import json
from pathlib import Path
from typing import Any
import copy

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def _extract_json_objects_from_text(text: str) -> list:
    """Attempt to extract any JSON objects/arrays embedded in a text blob.

    Uses json.JSONDecoder.raw_decode to robustly find JSON starting at any position.
    Returns a list of parsed Python objects.
    """
    objs = []
    try:
        decoder = json.JSONDecoder()
        idx = 0
        L = len(text or "")
        while idx < L:
            # find next possible JSON start
            next_start = None
            for i in range(idx, L):
                if text[i] in "{[":
                    next_start = i
                    break
            if next_start is None:
                break
            try:
                obj, end = decoder.raw_decode(text[next_start:])
                objs.append(obj)
                idx = next_start + end
            except Exception:
                # Move forward and keep searching
                idx = next_start + 1
    except Exception:
        pass
    return objs


def _sanitize_json_like_string(s: str) -> str:
    """
    Heuristically escape inner double-quotes that appear inside JSON string values
    so the blob becomes valid JSON. Keeps existing escapes intact.
    """
    if not isinstance(s, str):
        return s
    out_chars = []
    in_string = False
    escape = False
    for i, ch in enumerate(s):
        if in_string:
            if escape:
                out_chars.append(ch)
                escape = False
                continue
            if ch == "\\":
                out_chars.append(ch)
                escape = True
                continue
            if ch == '"':
                # Look ahead to decide if this ends the string (next non-space is a JSON delimiter)
                j = i + 1
                while j < len(s) and s[j] in (" ", "\n", "\r", "\t"):
                    j += 1
                if j >= len(s) or s[j] in [",", "}", "]", ":"]:
                    # Treat as closing quote
                    out_chars.append(ch)
                    in_string = False
                else:
                    # Likely an inner quote; escape it
                    out_chars.append('\\"')
                continue
            # regular char inside string
            out_chars.append(ch)
        else:
            if ch == '"':
                out_chars.append(ch)
                in_string = True
            else:
                out_chars.append(ch)
    return "".join(out_chars)


def _try_parse_json_like(txt: str):
    if not isinstance(txt, str):
        return None
    try:
        return json.loads(txt)
    except Exception:
        try:
            fixed = _sanitize_json_like_string(txt)
            return json.loads(fixed)
        except Exception:
            return None


def _normalize_result_structure(value: Any) -> Any:
    """
    Convert JSON-like strings to structured objects and de-nest any 'output' field
    that itself contains JSON. Prevents double-encoding like '\\n'.
    Also attempts to extract embedded JSON from noisy strings.
    """
    parsed = value

    # Try top-level normalization with fallback to embedded extraction
    for _ in range(3):
        if isinstance(parsed, str):
            txt = parsed.strip()
            if txt:
                if (txt[0] in "{[") and (txt[-1] in "}]"):
                    candidate = _try_parse_json_like(txt)
                    if candidate is not None:
                        parsed = candidate
                        continue
                    try:
                        found = _extract_json_objects_from_text(txt)
                        if found:
                            parsed = found[0]
                            continue
                    except Exception:
                        pass
        break

    # If dict with 'output' as JSON-like string, try to parse or extract
    if isinstance(parsed, dict):
        out = parsed.get("output")
        if isinstance(out, str):
            out_txt = out.strip()
            if out_txt:
                if (out_txt[0] in "{[") and (out_txt[-1] in "}]"):
                    candidate = _try_parse_json_like(out_txt)
                    if candidate is not None:
                        parsed["output"] = candidate
                    else:
                        try:
                            found = _extract_json_objects_from_text(out_txt)
                            if found:
                                parsed["output"] = found[0]
                        except Exception:
                            pass
                else:
                    # Even if it doesn't start with { or [, try to extract any JSON object inside
                    try:
                        found = _extract_json_objects_from_text(out_txt)
                        if found:
                            parsed["output"] = found[0]
                    except Exception:
                        pass
    return parsed


def process_file(fp: Path) -> bool:
    """
    Normalize the 'result' field in a session JSON file if it's double-encoded.
    Returns True if file was modified.
    """
    try:
        with fp.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[SKIP] Failed to read {fp.name}: {e}")
        return False

    if not isinstance(data, dict) or "result" not in data:
        # not a session payload
        return False

    # Snapshot before normalization (stringify for deep equality)
    try:
        before = json.dumps(data["result"], ensure_ascii=False, sort_keys=True)
    except Exception:
        before = repr(data["result"])

    # Normalize on a deep copy to avoid mutating the original object in place
    normalized = _normalize_result_structure(copy.deepcopy(data["result"]))

    try:
        after = json.dumps(normalized, ensure_ascii=False, sort_keys=True)
    except Exception:
        after = repr(normalized)

    if before == after:
        # No change in serialized structure/content
        return False

    data["result"] = normalized
    try:
        with fp.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write("\n")
        print(f"[FIXED] {fp.name}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to write {fp.name}: {e}")
        return False


def main():
    if not DATA_DIR.exists():
        print(f"[INFO] Data directory not found: {DATA_DIR}")
        return

    modified = 0
    total = 0
    for fp in sorted(DATA_DIR.glob("*.json")):
        # Skip non-session aggregate files if any
        if fp.name == "json_objects.json":
            continue
        total += 1
        if process_file(fp):
            modified += 1

    print(f"[DONE] Scanned {total} file(s); modified {modified} file(s).")


if __name__ == "__main__":
    main()
