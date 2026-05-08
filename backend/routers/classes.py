import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(tags=["classes"])

_KB_PATH = Path(__file__).parent.parent / "data" / "vehicle_classes.json"


class VehicleClassData(BaseModel):
    id: str
    label: str
    years: str
    generation_code: str = ""
    key_visual_features: list[str] = []
    rear_features: list[str] = []
    side_features: list[str] = []
    distinguishing_tips: str = ""
    common_confusables: list[str] = []
    key_visual_features_enabled: bool = True
    rear_features_enabled: bool = True
    side_features_enabled: bool = True
    distinguishing_tips_enabled: bool = True
    common_confusables_enabled: bool = True


def _load() -> dict:
    return json.loads(_KB_PATH.read_text())


def _save(data: dict) -> None:
    _KB_PATH.write_text(json.dumps(data, indent=2))
    from routers.evaluate import invalidate_kb_cache
    invalidate_kb_cache()


@router.get("/vehicle-classes")
async def get_vehicle_classes():
    return _load()


@router.post("/vehicle-classes", status_code=201)
async def create_vehicle_class(cls: VehicleClassData):
    data = _load()
    if any(c["id"] == cls.id for c in data["classes"]):
        raise HTTPException(status_code=409, detail=f"Class '{cls.id}' already exists")
    data["classes"].append(cls.model_dump())
    _save(data)
    return cls


@router.put("/vehicle-classes/{class_id}")
async def update_vehicle_class(class_id: str, cls: VehicleClassData):
    data = _load()
    idx = next((i for i, c in enumerate(data["classes"]) if c["id"] == class_id), None)
    if idx is None:
        raise HTTPException(status_code=404, detail="Class not found")
    data["classes"][idx] = cls.model_dump()
    _save(data)
    return cls


@router.delete("/vehicle-classes/{class_id}")
async def delete_vehicle_class(class_id: str):
    data = _load()
    original_len = len(data["classes"])
    data["classes"] = [c for c in data["classes"] if c["id"] != class_id]
    if len(data["classes"]) == original_len:
        raise HTTPException(status_code=404, detail="Class not found")
    _save(data)
    return {"deleted": True}
