import hashlib
import os
from pathlib import Path
from urllib.parse import quote

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, Response

from models.schemas import BrowseTreeResponse, FileNode

router = APIRouter(tags=["filesystem"])

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
CONTENT_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}

THUMBS_DIR = Path(os.getenv("THUMBNAILS_CACHE_PATH", "./data/thumbs"))


def _build_tree(directory: Path) -> tuple[list[FileNode], int]:
    nodes: list[FileNode] = []
    total = 0

    try:
        items = sorted(directory.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
    except PermissionError:
        return nodes, total

    for item in items:
        if item.name.startswith("."):
            continue
        if item.is_dir():
            children, count = _build_tree(item)
            total += count
            nodes.append(FileNode(name=item.name, type="folder", image_count=count, children=children))
        elif item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS:
            path_str = str(item)
            nodes.append(FileNode(
                name=item.name,
                type="image",
                path=path_str,
                thumb_url=f"/api/thumb?path={quote(path_str)}",
            ))
            total += 1

    return nodes, total


@router.get("/browse-tree", response_model=BrowseTreeResponse)
async def browse_tree(path: str):
    root = Path(path)
    if not root.exists():
        raise HTTPException(status_code=404, detail="Path not found")
    if not root.is_dir():
        raise HTTPException(status_code=400, detail="Path is not a directory")

    tree, total = _build_tree(root)
    return BrowseTreeResponse(tree=tree, total_images=total, root_path=str(root))


@router.get("/thumb")
async def get_thumb(path: str):
    image_path = Path(path)
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")

    cache_key = hashlib.md5(path.encode()).hexdigest()
    THUMBS_DIR.mkdir(parents=True, exist_ok=True)
    cached = THUMBS_DIR / f"{cache_key}.jpg"

    if not cached.exists():
        try:
            from PIL import Image
            with Image.open(image_path) as img:
                img.thumbnail((200, 200))
                img = img.convert("RGB")
                img.save(cached, "JPEG", quality=70)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Thumbnail generation failed: {e}")

    return FileResponse(str(cached), media_type="image/jpeg", headers={"Cache-Control": "max-age=86400"})


@router.get("/image")
async def get_image(path: str):
    image_path = Path(path)
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")

    media_type = CONTENT_TYPES.get(image_path.suffix.lower(), "application/octet-stream")
    return FileResponse(str(image_path), media_type=media_type, headers={"Cache-Control": "max-age=3600"})
