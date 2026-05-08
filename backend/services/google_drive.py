import os
from pathlib import Path

import aiohttp

DRIVE_API = "https://www.googleapis.com/drive/v3"
CACHE_DIR  = Path(os.getenv("DRIVE_CACHE_PATH", "./data/drive_cache"))

_MIME_EXT = {
    "image/jpeg": ".jpg",
    "image/png":  ".png",
    "image/webp": ".webp",
    "image/bmp":  ".bmp",
    "image/gif":  ".gif",
}


async def fetch_image_bytes(file_id: str, api_key: str) -> bytes:
    async with aiohttp.ClientSession() as session:
        async with session.get(
            f"{DRIVE_API}/files/{file_id}",
            params={"alt": "media", "key": api_key},
        ) as resp:
            if resp.status == 403:
                raise ValueError("Access denied — check API key and file sharing settings")
            if not resp.ok:
                raise ValueError(f"Drive error {resp.status}: {(await resp.text())[:200]}")
            return await resp.read()


async def cache_image(file_id: str, api_key: str, mime: str) -> str:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ext = _MIME_EXT.get(mime, ".jpg")
    local_path = CACHE_DIR / f"{file_id}{ext}"
    if not local_path.exists():
        img_bytes = await fetch_image_bytes(file_id, api_key)
        local_path.write_bytes(img_bytes)
    return str(local_path)
