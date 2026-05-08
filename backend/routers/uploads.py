import asyncio
import hashlib
import os
from pathlib import Path
from urllib.parse import urlparse

import aiohttp
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from typing import Optional
from pydantic import BaseModel

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "./data/uploads"))
IMAGE_MIMES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
_MIME_EXT   = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp", "image/bmp": ".bmp"}

router = APIRouter(tags=["uploads"])


@router.post("/upload")
async def upload_image(file: UploadFile = File(...), rel_path: Optional[str] = Form(None)):
    mime = (file.content_type or "").split(";")[0].strip()
    if mime not in IMAGE_MIMES:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {mime}")
    content = await file.read()
    ext = _MIME_EXT.get(mime, ".jpg")
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    if rel_path:
        local_path = (UPLOAD_DIR / rel_path).resolve()
        if not str(local_path).startswith(str(UPLOAD_DIR.resolve())):
            raise HTTPException(status_code=400, detail="Invalid path")
        local_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        local_path = UPLOAD_DIR / (hashlib.md5(content).hexdigest() + ext)
    local_path.write_bytes(content)
    return {"local_path": str(local_path), "filename": file.filename or local_path.name}


class FetchUrlRequest(BaseModel):
    url: str


@router.post("/fetch-url")
async def fetch_url(req: FetchUrlRequest):
    parsed = urlparse(req.url)
    if parsed.scheme not in ("http", "https"):
        raise HTTPException(status_code=400, detail="Only http/https URLs are supported")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(req.url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if not resp.ok:
                    raise HTTPException(status_code=502, detail=f"Failed to fetch URL: HTTP {resp.status}")
                mime = (resp.content_type or "image/jpeg").split(";")[0].strip()
                if not mime.startswith("image/"):
                    raise HTTPException(status_code=400, detail=f"URL does not point to an image ({mime})")
                content = await resp.read()
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        raise HTTPException(status_code=502, detail=f"Network error: {e}")
    ext = _MIME_EXT.get(mime, ".jpg")
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    local_path = UPLOAD_DIR / (hashlib.md5(content).hexdigest() + ext)
    local_path.write_bytes(content)
    url_filename = Path(parsed.path).name or local_path.name
    return {"local_path": str(local_path), "filename": url_filename}
