from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.core.config import settings
from app.core.errors import generic_exception_handler, validation_exception_handler
from app.api import ingest, embed, vectorstore, retriever, chat
from app.api.v1 import router as v1_router, limiter

app = FastAPI(
    title       = settings.app_name,
    version     = settings.app_version,
    description = "Personal Knowledge Base Copilot — RAG API",
    debug       = settings.debug,
)

# ── Rate limiter state ────────────────────────────────────────────────────────
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ── Global error handlers ─────────────────────────────────────────────────────
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(Exception, generic_exception_handler)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["http://localhost:3000"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
# v1 (clean, production API — what the frontend uses)
app.include_router(v1_router)

# Legacy routers (kept for debugging via Swagger during development)
app.include_router(ingest.router)
app.include_router(embed.router)
app.include_router(vectorstore.router)
app.include_router(retriever.router)
app.include_router(chat.router)


@app.get("/")
def root():
    return {
        "message": f"{settings.app_name} is running",
        "version": settings.app_version,
        "docs":    "/docs",
        "api":     "/api/v1",
    }