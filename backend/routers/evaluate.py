import base64
import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import quote

from fastapi import APIRouter, HTTPException

from models.schemas import EvaluateRequest, EvaluateResponse, EvaluationMode, Provider, SimilarImage
from services import classifier, embeddings, vector_store

router = APIRouter(tags=["evaluate"])


def invalidate_kb_cache():
    global _KB, _KB_CACHE
    _KB = None
    _KB_CACHE = {}


@router.get("/classes")
async def get_classes():
    return [{"id": cls["id"], "label": cls["label"]} for cls in _get_kb()["classes"]]

_KB_PATH = Path(__file__).parent.parent / "data" / "vehicle_classes.json"
_MANIFEST_PATH = Path(__file__).parent.parent / "data" / "classes" / "manifest.json"
_KB: dict | None = None
_KB_CACHE: dict[str, dict] = {}

MIME_MAP = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}


def _get_kb(knowledge_base_id: str | None = None) -> dict:
    global _KB
    if knowledge_base_id and _MANIFEST_PATH.exists():
        if knowledge_base_id not in _KB_CACHE:
            manifest = json.loads(_MANIFEST_PATH.read_text())
            entry = next((e for e in manifest if e["id"] == knowledge_base_id), None)
            if entry:
                kb_path = _KB_PATH.parent / entry["file"]
                _KB_CACHE[knowledge_base_id] = json.loads(kb_path.read_text())
        if knowledge_base_id in _KB_CACHE:
            return _KB_CACHE[knowledge_base_id]
    if _KB is None:
        with open(_KB_PATH) as f:
            _KB = json.load(f)
    return _KB


def _format_kb(kb: dict) -> str:
    lines = [f"Vehicle: {kb['vehicle']} ({kb['market']} market)\n"]
    for cls in kb["classes"]:
        lines.append(f"CLASS ID: {cls['id']}")
        lines.append(f"LABEL: {cls['label']}")
        lines.append(f"Years: {cls['years']}")
        if cls.get("key_visual_features_enabled", True) and cls.get("key_visual_features"):
            lines.append("Key Visual Features:")
            for feat in cls["key_visual_features"]:
                lines.append(f"  - {feat}")
        if cls.get("rear_features_enabled", True) and cls.get("rear_features"):
            lines.append("Rear Features:")
            for feat in cls["rear_features"]:
                lines.append(f"  - {feat}")
        if cls.get("side_features_enabled", True) and cls.get("side_features"):
            lines.append("Side Features:")
            for feat in cls["side_features"]:
                lines.append(f"  - {feat}")
        if cls.get("distinguishing_tips_enabled", True) and cls.get("distinguishing_tips"):
            lines.append(f"Distinguishing Tips: {cls['distinguishing_tips']}")
        if cls.get("common_confusables_enabled", True) and cls.get("common_confusables"):
            lines.append(f"Often confused with: {', '.join(cls['common_confusables'])}")
        lines.append("")
    return "\n".join(lines)


def _strip_json(raw: str) -> str:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    raw = raw.strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1:
        raw = raw[start : end + 1]
    return raw


def _encode_image(path: Path) -> tuple[str, str]:
    mime = MIME_MAP.get(path.suffix.lower(), "image/jpeg")
    b64 = base64.standard_b64encode(path.read_bytes()).decode()
    return b64, mime


@router.post("/evaluate", response_model=EvaluateResponse)
async def evaluate(req: EvaluateRequest):
    image_path = Path(req.image_path)
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")

    kb = _get_kb(req.knowledge_base_id)
    prompt = req.prompt.replace("{CLASS_KNOWLEDGE_BASE}", _format_kb(kb))

    b64, mime = _encode_image(image_path)

    similar_images = None
    if req.mode == EvaluationMode.rag:
        if not embeddings.is_loaded():
            raise HTTPException(status_code=503, detail="CLIP model not loaded — RAG unavailable")
        if vector_store.count() == 0:
            raise HTTPException(status_code=503, detail="Index is empty — build the index first")
        img_bytes = image_path.read_bytes()
        emb = await _run_in_thread(embeddings.generate_embedding_from_bytes, img_bytes)
        raw_similars = vector_store.find_similar(emb, top_k=5)
        if raw_similars:
            context_lines = ["\n\nSIMILAR REFERENCE IMAGES FROM DATABASE (use as supporting evidence):"]
            for s in raw_similars:
                context_lines.append(
                    f"  - class_id={s['class_id']}, angle={s['angle']}, similarity={s['similarity_score']:.0%}"
                )
            prompt = prompt + "\n" + "\n".join(context_lines)
        similar_images = [
            SimilarImage(
                path=s["path"],
                thumb_url=f"/api/thumb?path={quote(s['path'])}",
                class_id=s["class_id"],
                angle=s["angle"],
                similarity_score=s["similarity_score"],
            )
            for s in raw_similars
        ]

    try:
        if req.provider == Provider.anthropic:
            raw = await classifier.call_anthropic(b64, mime, prompt, req.api_key, req.model)
        elif req.provider == Provider.google:
            raw = await classifier.call_google(b64, mime, prompt, req.api_key, req.model)
        elif req.provider == Provider.openai:
            raw = await classifier.call_openai(b64, mime, prompt, req.api_key, req.model)
        else:  # ollama
            raw = await classifier.call_ollama(b64, mime, prompt, req.ollama_endpoint, req.model)
    except ValueError as e:
        msg = str(e)
        status = 401 if ("401" in msg or "api_key" in msg.lower() or "invalid" in msg.lower()) else 502
        raise HTTPException(status_code=status, detail=msg)

    try:
        parsed: dict[str, Any] = json.loads(_strip_json(raw))
    except json.JSONDecodeError:
        raise HTTPException(status_code=422, detail=f"AI returned unparseable JSON: {raw[:300]}")

    return EvaluateResponse(
        class_id=parsed.get("class_id", "unknown"),
        class_label=parsed.get("class_label", "Unknown"),
        confidence=parsed.get("confidence", "low"),
        confidence_score=float(parsed.get("confidence_score", 0.0)),
        reasoning=parsed.get("reasoning", ""),
        visible_features=parsed.get("visible_features", []),
        similar_images=similar_images,
    )


async def _run_in_thread(fn, *args):
    import asyncio
    return await asyncio.to_thread(fn, *args)
