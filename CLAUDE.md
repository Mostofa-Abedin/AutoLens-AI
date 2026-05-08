# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

AutoLens AI is a one-image-at-a-time vehicle identification workbench. A user browses a local folder of images, selects one, sends it to an AI vision model, and sees structured reasoning about which vehicle generation is shown. The MVP classifies Ford Ranger generations (5 classes, Australian market) but the architecture is generic.

It is NOT a batch sorting tool.

## Running the app

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Open: http://localhost:8000  
API docs: http://localhost:8000/docs

## Tech stack

- **Backend:** FastAPI (Python 3.11+), SQLite via SQLAlchemy async, ChromaDB (local)
- **Frontend:** Vanilla HTML/CSS/JS — single file (`frontend/index.html`), all CSS/JS inline
- **AI providers:** Anthropic, Google Gemini, OpenAI, Ollama — all pluggable via `services/classifier.py`
- **Embeddings:** CLIP `openai/clip-vit-base-patch32` via HuggingFace — load once at startup

## Architecture

```
backend/
  main.py              FastAPI app — lifespan inits DB + data dirs, mounts all routers
  routers/             One file per domain; all included in main.py with /api prefix
                       (auth.py has no prefix — its /api/drive/* paths are explicit)
  services/            Pure logic, no FastAPI — classifier.py, embeddings.py,
                       vector_store.py, google_drive.py
  models/schemas.py    All Pydantic request/response models
  database/
    db.py              SQLAlchemy engine + ORM models (Prompt, EvaluationResult)
    crud.py            Async DB operations
  data/
    vehicle_classes.json  Knowledge base injected into every AI prompt
frontend/
  index.html           Entire frontend — three-column layout, all CSS/JS inline
```

## Key implementation rules (from spec)

- **CLIP model** — load once in the FastAPI `lifespan` event, never per-request
- **Thumbnails** — always check `data/thumbs/{md5_of_path}.jpg` cache before generating
- **`{CLASS_KNOWLEDGE_BASE}` placeholder** — replaced server-side in `evaluate.py` before the prompt is sent to the AI; never send the placeholder to the model
- **AI JSON parsing** — always wrap in `try/except`; strip markdown fences/backticks before `json.loads()`
- **Paths** — use `pathlib.Path` throughout, never string concatenation
- **ChromaDB collection** — must use cosine similarity metric, not euclidean
- **API keys** — always from request body or `.env`; never hardcoded

## Build order (steps)

1. ✅ Scaffold — structure, schemas, DB models, stub routers, running `/health`
2. Filesystem endpoints — browse-tree, thumb, image
3. Classifier service + evaluate endpoint (cold mode)
4. Prompts CRUD + DB seeding (3 default prompts)
5. Frontend — full three-column layout wired to backend
6. Session endpoints + feedback UI
7. CLIP indexing — embeddings service + ChromaDB wrapper + build-index endpoint
8. RAG mode in evaluate + similar images panel
9. Google OAuth + Drive tab
10. Drag-and-drop + URL tabs

## Frontend layout

Three columns: `[Explorer 320px] [Analysis flex] [Session 300px]`  
Collapses to bottom panel < 1200px, single column with tabs < 768px.  
Font: IBM Plex Mono + IBM Plex Sans. Accent colour: `#c0340e`.

## Environment variables

See `backend/.env.example`. Key ones: `DATABASE_URL`, `VECTOR_STORE_PATH`, `THUMBNAILS_CACHE_PATH`.
