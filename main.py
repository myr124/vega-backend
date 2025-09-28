import os
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from multi_tool_agent.agent import root_agent
from google.adk.runners import Runner, types
from google.adk.sessions import InMemorySessionService

import base64
import json
import datetime
import tempfile
import subprocess
import shutil
import asyncio
import httpx


# Load environment variables
load_dotenv()
env_path = Path(__file__).with_name(".env")
if env_path.exists():
    load_dotenv(dotenv_path=env_path)

# Import the existing agent and Google ADK components


class RunRequest(BaseModel):
    prompt: Optional[str] = Form(None)
    video: Optional[UploadFile] = File(None)
    video_uri: Optional[str] = Form(None)


class RunResponse(BaseModel):
    ok: bool
    result: Optional[Any] = None
    error: Optional[str] = None


def _extract_result(raw: Any) -> Any:
    """
    Extract the result from the agent's raw output.
    """
    if isinstance(raw, tuple) and len(raw) == 2:
        return raw[0]  # Assume first element is the result
    elif hasattr(raw, "result") or hasattr(raw, "output"):
        return getattr(raw, "result", getattr(raw, "output", raw))
    return raw


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
    that itself contains JSON. This prevents double-encoding like '\\n' in files.
    Also attempts to extract embedded JSON from noisy strings.
    """
    parsed = value

    # Try to parse top-level if it's a JSON string, with fallback to embedded extraction
    for _ in range(3):
        if isinstance(parsed, str):
            txt = parsed.strip()
            if txt:
                if (txt[0] in "{[") and (txt[-1] in "}]"):
                    candidate = _try_parse_json_like(txt)
                    if candidate is not None:
                        parsed = candidate
                        continue
                    # fall back to extracting embedded JSON from the string
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


app = FastAPI(title="Agent Backend", version="0.1.0")

# CORS configuration
frontend_origin = os.getenv("FRONTEND_ORIGIN", "http://localhost:3000")
allow_origins = [
    frontend_origin,
    "http://localhost:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/agent/run", response_model=RunResponse)
async def run_agent(
    prompt: Optional[str] = Form(None),
    video: Optional[UploadFile] = File(None),
    video_uri: Optional[str] = Form(None),
) -> RunResponse:
    print(
        f"[run_agent] Received prompt={bool(prompt)} video_present={bool(video)} video_uri_present={bool(video_uri)}"
    )
    if not prompt and not video and not video_uri:
        raise HTTPException(
            status_code=400, detail="Either prompt, video, or video_uri is required"
        )

    try:
        session_service = InMemorySessionService()
        runner = Runner(
            app_name="vega-agent", agent=root_agent, session_service=session_service
        )
        user_id = "test_user"
        session_id = "test_session"
        # Create session if not exists
        session_service.create_session_sync(
            app_name="vega-agent", user_id=user_id, session_id=session_id
        )

        parts = []
        if prompt:
            parts.append(types.Part(text=prompt.strip()))

        video_bytes: Optional[bytes] = None
        mime_type: Optional[str] = None
        original_filename: Optional[str] = (
            getattr(video, "filename", None) if video else None
        )

        if video_uri:
            try:
                async with httpx.AsyncClient(
                    follow_redirects=True, timeout=60.0
                ) as client:
                    resp = await client.get(video_uri)
                    resp.raise_for_status()
                    video_bytes = resp.content
                    ct = resp.headers.get("Content-Type")
                    if ct:
                        mime_type = ct
                    print(
                        f"[run_agent] Fetched video from URI; bytes={len(video_bytes) if video_bytes else 0} mime={mime_type}"
                    )
            except Exception as e:
                print(f"WARNING: failed to fetch video from URI: {e}")
                video_bytes = None

            if not mime_type:
                if isinstance(video_uri, str) and video_uri.lower().endswith(".mp4"):
                    mime_type = "video/mp4"
                else:
                    mime_type = "application/octet-stream"

        elif video:
            video_bytes = await video.read()
            # Force video/mp4 for MP4 files, as content_type may default to octet-stream
            if video.filename and video.filename.lower().endswith(".mp4"):
                mime_type = "video/mp4"
            else:
                mime_type = video.content_type or "video/mp4"
            print(
                f"[run_agent] Received video file upload; bytes={len(video_bytes) if video_bytes else 0} mime={mime_type}"
            )

        new_message = types.Content(parts=parts, role="user")
        events = list(
            runner.run(user_id=user_id, session_id=session_id, new_message=new_message)
        )
        print("Events:", [str(e) for e in events])  # Debug print
        # Extract result from events
        result = None
        for event in reversed(events):
            if event.content and event.content.parts:
                result = event.content.parts[0].text
                break
        if result is None:
            result = "No response generated"

        # --- New: extract JSON objects produced by enjoyer/reviewer agents and persist them ---
        found_json_objects: list = []
        try:
            # Gather texts from all event parts to search for JSON
            all_text = "\n".join(
                p.text
                for e in events
                if getattr(e, "content", None)
                for p in e.content.parts
                if getattr(p, "text", None)
            )
            found = _extract_json_objects_from_text(all_text)
            if found:
                found_json_objects = found
                # Ensure data dir exists
                json_file = Path(__file__).parent / "data" / "json_objects.json"
                json_file.parent.mkdir(parents=True, exist_ok=True)
                # Load existing list if present
                try:
                    if json_file.exists():
                        with open(json_file, "r", encoding="utf-8") as jf:
                            existing = json.load(jf)
                            if isinstance(existing, list):
                                jsonObjectList.extend(existing)
                except Exception as e:
                    print(f"WARNING: failed to load existing json objects: {e}")

                # Append and deduplicate simple by string representation
                for obj in found:
                    try:
                        jsonObjectList.append(obj)
                    except Exception:
                        pass

                # Persist the list
                try:
                    with open(json_file, "w", encoding="utf-8") as jf:
                        json.dump(jsonObjectList, jf, ensure_ascii=False, indent=2)
                    print(f"Appended {len(found)} json object(s) to {json_file}")
                except Exception as e:
                    print(f"WARNING: failed to persist json object list: {e}")
        except Exception as e:
            print(f"WARNING: json extraction failed: {e}")

        # Ensure JSON-serializable (normalize potential JSON strings first)
        normalized_result = _normalize_result_structure(result)
        # If still string-like or contains string 'output', prefer extracted JSON if available
        if (
            (
                isinstance(normalized_result, str)
                or (
                    isinstance(normalized_result, dict)
                    and isinstance(normalized_result.get("output"), str)
                )
            )
            and "found_json_objects" in locals()
            and found_json_objects
        ):
            try:
                normalized_result = found_json_objects[-1]
            except Exception:
                pass
        safe_result = jsonable_encoder(normalized_result, custom_encoder={set: list})

        # Persist the result to a JSON file under src/backend/data
        try:
            data_dir = Path(__file__).parent / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{session_id}.json"
            filepath = data_dir / filename
            payload = {
                "meta": {
                    "app_name": "vega-agent",
                    "user_id": user_id,
                    "session_id": session_id,
                    "created_at": datetime.datetime.utcnow().isoformat() + "Z",
                },
                "result": safe_result,
            }
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            print(f"Saved run result to {filepath}")
        except Exception as e:
            print(f"WARNING: failed to write run result to file: {e}")

        return RunResponse(
            ok=True,
            result=safe_result,
        )
    except Exception as e:
        return RunResponse(ok=False, error=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="localhost", port=2000, reload=True)


jsonObjectList = []
inputJsonObject = {}
