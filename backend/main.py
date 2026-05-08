import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

load_dotenv()

from database.db import engine, Base, AsyncSessionLocal
from database import crud
from routers import auth, classes, evaluate, filesystem, identify, index, prompts, sessions, settings, uploads
from routers.prompts import DEFAULT_PROMPTS

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    Path("data/chroma").mkdir(parents=True, exist_ok=True)
    Path("data/thumbs").mkdir(parents=True, exist_ok=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    async with AsyncSessionLocal() as db:
        if not await crud.prompts_exist(db):
            await crud.seed_prompts(db, DEFAULT_PROMPTS)
    try:
        from services import embeddings
        await asyncio.to_thread(embeddings.load_model)
    except Exception as e:
        print(f"CLIP not loaded (RAG/indexing unavailable): {e}")
    yield


app = FastAPI(title="AutoLens AI", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(filesystem.router, prefix="/api")
app.include_router(identify.router, prefix="/api")
app.include_router(evaluate.router, prefix="/api")
app.include_router(index.router, prefix="/api")
app.include_router(prompts.router, prefix="/api")
app.include_router(classes.router, prefix="/api")
app.include_router(settings.router, prefix="/api")
app.include_router(sessions.router, prefix="/api")
app.include_router(uploads.router, prefix="/api")
app.include_router(auth.router)

assets_dir = FRONTEND_DIR / "assets"
if assets_dir.exists():
    app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")


@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}


@app.get("/{full_path:path}")
async def serve_frontend(full_path: str):
    return FileResponse(str(FRONTEND_DIR / "index.html"))
