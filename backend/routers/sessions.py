import csv
import io
import uuid

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from database.db import get_db
from database import crud
from models.schemas import SessionResultRequest, SessionStats, ClassStat
from routers.evaluate import _get_kb

router = APIRouter(tags=["sessions"])


def _build_stats(session_id: str, results: list) -> SessionStats:
    labels = {cls["id"]: cls["label"] for cls in _get_kb()["classes"]}
    total = len(results)
    correct = sum(1 for r in results if r.correct)
    accuracy = (correct / total) if total else 0.0

    counts: dict[str, dict] = {}
    for r in results:
        cls = r.actual_class
        if cls not in counts:
            counts[cls] = {"total": 0, "correct": 0}
        counts[cls]["total"] += 1
        if r.correct:
            counts[cls]["correct"] += 1

    by_class = sorted(
        [
            ClassStat(
                class_id=cls_id,
                class_label=labels.get(cls_id, cls_id),
                total=s["total"],
                correct=s["correct"],
                accuracy=s["correct"] / s["total"] if s["total"] else 0.0,
            )
            for cls_id, s in counts.items()
        ],
        key=lambda x: x.class_id,
    )

    return SessionStats(
        session_id=session_id,
        total=total,
        correct=correct,
        accuracy=accuracy,
        by_class=by_class,
    )


@router.post("/session/start")
async def start_session():
    return {"session_id": str(uuid.uuid4())}


@router.post("/session/result", response_model=SessionStats)
async def submit_result(data: SessionResultRequest, db: AsyncSession = Depends(get_db)):
    await crud.create_evaluation_result(db, data)
    results = await crud.get_session_results(db, data.session_id)
    return _build_stats(data.session_id, results)


@router.get("/session/stats", response_model=SessionStats)
async def get_stats(session_id: str, db: AsyncSession = Depends(get_db)):
    results = await crud.get_session_results(db, session_id)
    return _build_stats(session_id, results)


@router.get("/session/export")
async def export_session(session_id: str, db: AsyncSession = Depends(get_db)):
    results = await crud.get_session_results(db, session_id)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["filename", "predicted_class", "actual_class", "correct",
                "confidence_score", "mode", "model_used", "timestamp"])
    for r in results:
        w.writerow([r.image_filename, r.predicted_class, r.actual_class, r.correct,
                    r.confidence_score, r.mode, r.model_used, r.created_at.isoformat()])
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="session_{session_id[:8]}.csv"'},
    )


@router.delete("/session")
async def reset_session(session_id: str, db: AsyncSession = Depends(get_db)):
    await crud.delete_session_results(db, session_id)
    return {"cleared": True}
