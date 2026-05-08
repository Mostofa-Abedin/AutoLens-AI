import aiohttp
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

from services import google_drive

router = APIRouter(tags=["drive"])


@router.get("/api/drive/image")
async def drive_image(file_id: str = Query(...), api_key: str = Query(...)):
    """Proxy Drive image bytes to the browser (avoids CORS on direct download URLs)."""
    try:
        img_bytes = await google_drive.fetch_image_bytes(file_id, api_key)
    except ValueError as e:
        raise HTTPException(status_code=502, detail=str(e))
    content_type = "image/jpeg"
    if img_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        content_type = "image/png"
    elif img_bytes[:4] == b"RIFF" and img_bytes[8:12] == b"WEBP":
        content_type = "image/webp"
    return Response(content=img_bytes, media_type=content_type)


@router.get("/api/drive/cache")
async def drive_cache(
    file_id: str = Query(...),
    api_key: str = Query(...),
    mime:    str = Query("image/jpeg"),
):
    """Download a Drive file to the local cache and return its path for /api/evaluate."""
    try:
        local_path = await google_drive.cache_image(file_id, api_key, mime)
    except ValueError as e:
        raise HTTPException(status_code=502, detail=str(e))
    return {"local_path": local_path}
