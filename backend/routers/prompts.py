from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from database.db import Prompt, get_db
from database import crud
from models.schemas import PromptCreate, PromptUpdate, PromptResponse

router = APIRouter(tags=["prompts"])

DEFAULT_PROMPTS = [
    {
        "name": "Default Prompt",
        "is_default": True,
        "content": (
            "You are an expert vehicle classification system.\n\n"
            "For this image, identify both the vehicle generation AND the camera angle.\n\n"
            "{CLASS_KNOWLEDGE_BASE}\n\n"
            "Return a valid JSON object:\n"
            "{\n"
            '  "class_id": "exact_class_id",\n'
            '  "class_label": "human readable label",\n'
            '  "angle": "front_3q|rear_3q|side|front|rear|other",\n'
            '  "confidence": "high|medium|low",\n'
            '  "confidence_score": 0.0,\n'
            '  "reasoning": "2-4 sentences on generation AND angle identification",\n'
            '  "visible_features": ["features", "you", "can", "see"]\n'
            "}"
        ),
    },
]


def _to_response(p: Prompt) -> PromptResponse:
    return PromptResponse(
        id=p.id,
        name=p.name,
        content=p.content,
        is_default=p.is_default,
        created_at=p.created_at.isoformat() if p.created_at else "",
    )


@router.get("/prompts", response_model=list[PromptResponse])
async def get_prompts(db: AsyncSession = Depends(get_db)):
    prompts = await crud.get_all_prompts(db)
    return [_to_response(p) for p in prompts]


@router.post("/prompts/seed")
async def seed_prompts(db: AsyncSession = Depends(get_db)):
    if await crud.prompts_exist(db):
        return {"seeded": 0, "message": "Prompts already exist"}
    count = await crud.seed_prompts(db, DEFAULT_PROMPTS)
    return {"seeded": count}


@router.post("/prompts", response_model=PromptResponse, status_code=201)
async def create_prompt(data: PromptCreate, db: AsyncSession = Depends(get_db)):
    prompt = await crud.create_prompt(db, data)
    return _to_response(prompt)


@router.put("/prompts/{prompt_id}", response_model=PromptResponse)
async def update_prompt(prompt_id: int, data: PromptUpdate, db: AsyncSession = Depends(get_db)):
    prompt = await crud.update_prompt(db, prompt_id, data)
    if not prompt:
        raise HTTPException(status_code=404, detail="Prompt not found")
    return _to_response(prompt)


@router.delete("/prompts/{prompt_id}")
async def delete_prompt(prompt_id: int, db: AsyncSession = Depends(get_db)):
    deleted = await crud.delete_prompt(db, prompt_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Prompt not found")
    return {"deleted": True}
