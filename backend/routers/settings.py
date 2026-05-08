from fastapi import APIRouter
from pydantic import BaseModel

from services import key_store

router = APIRouter(tags=["settings"])


class KeySaveRequest(BaseModel):
    provider: str
    api_key: str


@router.get("/settings/saved-keys")
async def get_saved_keys():
    return {"saved": key_store.list_saved()}


@router.get("/settings/key")
async def get_key(provider: str):
    return {"key": key_store.load_key(provider)}


@router.post("/settings/key")
async def save_key(req: KeySaveRequest):
    key_store.save_key(req.provider, req.api_key)
    return {"saved": True}


@router.delete("/settings/key")
async def delete_key(provider: str):
    key_store.delete_key(provider)
    return {"deleted": True}
