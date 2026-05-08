import json
import re
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

from models.schemas import IdentifyRequest, IdentifyResponse, Provider
from services import classifier

router = APIRouter(tags=["identify"])

_MANIFEST_PATH = Path(__file__).parent.parent / "data" / "classes" / "manifest.json"

MIME_MAP = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}

STAGE1_PROMPT = """Look at this vehicle image and identify it.
Return ONLY a valid JSON object with no markdown, no backticks, no explanation:
{
  "make": "manufacturer name e.g. Ford",
  "model": "model name e.g. Ranger",
  "year_estimate": "year or range e.g. 2018-2022",
  "confidence": "high|medium|low",
  "confidence_score": 0.85,
  "reasoning": "2-3 sentences explaining which visual features identify this vehicle"
}"""


def _load_manifest() -> list[dict]:
    if _MANIFEST_PATH.exists():
        return json.loads(_MANIFEST_PATH.read_text())
    return []


def _match_manifest(make: str, model: str) -> dict | None:
    for entry in _load_manifest():
        if (entry["make"].lower() == make.lower() and
                entry["model"].lower() == model.lower()):
            return entry
    return None


def _strip_json(raw: str) -> str:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    raw = raw.strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end != -1:
        raw = raw[start: end + 1]
    return raw


import base64

def _encode_image(path: Path) -> tuple[str, str]:
    mime = MIME_MAP.get(path.suffix.lower(), "image/jpeg")
    b64 = base64.standard_b64encode(path.read_bytes()).decode()
    return b64, mime


@router.post("/identify", response_model=IdentifyResponse)
async def identify(req: IdentifyRequest):
    image_path = Path(req.image_path)
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")

    b64, mime = _encode_image(image_path)

    try:
        if req.provider == Provider.anthropic:
            raw = await classifier.call_anthropic(b64, mime, STAGE1_PROMPT, req.api_key, req.model)
        elif req.provider == Provider.google:
            raw = await classifier.call_google(b64, mime, STAGE1_PROMPT, req.api_key, req.model)
        elif req.provider == Provider.openai:
            raw = await classifier.call_openai(b64, mime, STAGE1_PROMPT, req.api_key, req.model)
        else:
            raw = await classifier.call_ollama(b64, mime, STAGE1_PROMPT, req.ollama_endpoint, req.model)
    except ValueError as e:
        msg = str(e)
        status = 401 if ("401" in msg or "api_key" in msg.lower() or "invalid" in msg.lower()) else 502
        raise HTTPException(status_code=status, detail=msg)

    try:
        parsed: dict[str, Any] = json.loads(_strip_json(raw))
    except json.JSONDecodeError:
        raise HTTPException(status_code=422, detail=f"AI returned unparseable JSON: {raw[:300]}")

    make = parsed.get("make", "Unknown")
    model = parsed.get("model", "Unknown")
    match = _match_manifest(make, model)

    return IdentifyResponse(
        make=make,
        model=model,
        year_estimate=parsed.get("year_estimate", "Unknown"),
        confidence=parsed.get("confidence", "low"),
        confidence_score=float(parsed.get("confidence_score", 0.0)),
        reasoning=parsed.get("reasoning", ""),
        in_database=match is not None,
        knowledge_base_id=match["id"] if match else None,
        knowledge_base_label=match["label"] if match else None,
    )
