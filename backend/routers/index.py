import asyncio
import json
import time
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException

from models.schemas import BuildIndexRequest, IndexClassGroup, IndexEntry, IndexInspectResponse, IndexStatusResponse
from services import embeddings, vector_store

router = APIRouter(tags=["index"])

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
ANGLE_KEYWORDS    = {"front_3q", "rear_3q", "side", "front", "rear", "interior", "other"}
STATUS_PATH       = Path("data/index_status.json")


def _extract_metadata(img_path: Path, root: Path) -> tuple[str, str]:
    """Walk path innermost→outermost; first angle-keyword folder = angle, first other = class_id."""
    parts = img_path.relative_to(root).parts[:-1]
    class_id = "unknown"
    angle    = "unknown"
    for part in reversed(parts):
        if part.lower() in ANGLE_KEYWORDS and angle == "unknown":
            angle = part.lower()
        elif class_id == "unknown":
            class_id = part
    return class_id, angle


def _run_build_index(dataset_path: str) -> dict:
    root = Path(dataset_path)
    if not root.exists():
        raise ValueError(f"Dataset path not found: {dataset_path}")

    images = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    if not images:
        raise ValueError("No images found in dataset path")

    vector_store.clear()

    start    = time.time()
    batch_sz = 32
    indexed  = 0

    for i in range(0, len(images), batch_sz):
        batch      = images[i : i + batch_sz]
        embs, metas, ids = [], [], []
        for img_path in batch:
            try:
                class_id, angle = _extract_metadata(img_path, root)
                emb = embeddings.generate_embedding(str(img_path))
                embs.append(emb)
                metas.append({
                    "path":     str(img_path),
                    "class_id": class_id,
                    "angle":    angle,
                    "filename": img_path.name,
                })
                ids.append(str(img_path))
                indexed += 1
            except Exception as e:
                print(f"Skipping {img_path.name}: {e}")

        if embs:
            vector_store.add_images(embs, metas, ids)

    elapsed = round(time.time() - start, 1)

    STATUS_PATH.parent.mkdir(exist_ok=True)
    STATUS_PATH.write_text(json.dumps({
        "built":        True,
        "total_images": indexed,
        "index_path":   vector_store.VECTOR_STORE_PATH,
        "last_built":   datetime.utcnow().isoformat(),
    }))

    return {"message": f"Indexed {indexed} images", "total_indexed": indexed, "time_seconds": elapsed}


@router.post("/build-index")
async def build_index(request: BuildIndexRequest):
    if not embeddings.is_loaded():
        raise HTTPException(status_code=503, detail="CLIP model not loaded — check server startup logs")
    try:
        result = await asyncio.to_thread(_run_build_index, request.dataset_path)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return result


@router.get("/index-status", response_model=IndexStatusResponse)
async def index_status():
    db_count = vector_store.count()
    if STATUS_PATH.exists():
        status = json.loads(STATUS_PATH.read_text())
        return IndexStatusResponse(
            built=db_count > 0,
            total_images=db_count,
            index_path=status.get("index_path", vector_store.VECTOR_STORE_PATH),
            last_built=status.get("last_built"),
        )
    return IndexStatusResponse(
        built=db_count > 0,
        total_images=db_count,
        index_path=vector_store.VECTOR_STORE_PATH,
        last_built=None,
    )


@router.get("/index-inspect", response_model=IndexInspectResponse)
async def index_inspect():
    from collections import defaultdict
    col = vector_store.get_collection()
    total = col.count()
    if total == 0:
        return IndexInspectResponse(total=0, by_class=[])
    results = col.get(include=["metadatas"])
    grouped: dict[str, list[IndexEntry]] = defaultdict(list)
    for meta in results["metadatas"]:
        grouped[meta.get("class_id", "unknown")].append(
            IndexEntry(
                filename=meta.get("filename", ""),
                path=meta.get("path", ""),
                angle=meta.get("angle", "unknown"),
            )
        )
    by_class = [
        IndexClassGroup(class_id=cid, count=len(entries), entries=entries)
        for cid, entries in sorted(grouped.items())
    ]
    return IndexInspectResponse(total=total, by_class=by_class)


@router.post("/clear-index")
async def clear_index():
    vector_store.clear()
    if STATUS_PATH.exists():
        STATUS_PATH.unlink()
    return {"message": "Index cleared"}
