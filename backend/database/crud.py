from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete
from database.db import Prompt, EvaluationResult
from models.schemas import PromptCreate, PromptUpdate, SessionResultRequest
from datetime import datetime


async def get_all_prompts(db: AsyncSession) -> list[Prompt]:
    result = await db.execute(select(Prompt).order_by(Prompt.created_at))
    return result.scalars().all()


async def get_prompt(db: AsyncSession, prompt_id: int) -> Prompt | None:
    result = await db.execute(select(Prompt).where(Prompt.id == prompt_id))
    return result.scalar_one_or_none()


async def clear_default_flags(db: AsyncSession):
    await db.execute(update(Prompt).values(is_default=False))


async def create_prompt(db: AsyncSession, data: PromptCreate) -> Prompt:
    if data.is_default:
        await clear_default_flags(db)
    prompt = Prompt(name=data.name, content=data.content, is_default=data.is_default)
    db.add(prompt)
    await db.commit()
    await db.refresh(prompt)
    return prompt


async def update_prompt(db: AsyncSession, prompt_id: int, data: PromptUpdate) -> Prompt | None:
    prompt = await get_prompt(db, prompt_id)
    if not prompt:
        return None
    if data.is_default:
        await clear_default_flags(db)
    if data.name is not None:
        prompt.name = data.name
    if data.content is not None:
        prompt.content = data.content
    if data.is_default is not None:
        prompt.is_default = data.is_default
    await db.commit()
    await db.refresh(prompt)
    return prompt


async def delete_prompt(db: AsyncSession, prompt_id: int) -> bool:
    result = await db.execute(delete(Prompt).where(Prompt.id == prompt_id))
    await db.commit()
    return result.rowcount > 0


async def prompts_exist(db: AsyncSession) -> bool:
    result = await db.execute(select(Prompt).limit(1))
    return result.scalar_one_or_none() is not None


async def seed_prompts(db: AsyncSession, prompts: list[dict]) -> int:
    for p in prompts:
        db.add(Prompt(name=p["name"], content=p["content"], is_default=p.get("is_default", False)))
    await db.commit()
    return len(prompts)


async def create_evaluation_result(db: AsyncSession, data: SessionResultRequest) -> EvaluationResult:
    result = EvaluationResult(
        session_id=data.session_id,
        image_path=data.image_path,
        image_filename=data.image_filename,
        predicted_class=data.predicted_class,
        actual_class=data.actual_class,
        correct=data.correct,
        mode=data.mode,
        model_used=data.model_used,
        confidence_score=data.confidence_score,
    )
    db.add(result)
    await db.commit()
    await db.refresh(result)
    return result


async def get_session_results(db: AsyncSession, session_id: str) -> list[EvaluationResult]:
    result = await db.execute(
        select(EvaluationResult)
        .where(EvaluationResult.session_id == session_id)
        .order_by(EvaluationResult.created_at)
    )
    return result.scalars().all()


async def delete_session_results(db: AsyncSession, session_id: str):
    await db.execute(delete(EvaluationResult).where(EvaluationResult.session_id == session_id))
    await db.commit()
