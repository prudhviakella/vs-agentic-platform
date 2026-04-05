"""
main.py — VS Agentic Platform Entry Point
==========================================
Assembles the FastAPI application, registers middleware, mounts routers,
and starts the uvicorn server.

Start the server:
  uvicorn platform.main:app --host 0.0.0.0 --port 8000 --reload

Production (ECS):
  uvicorn platform.main:app --host 0.0.0.0 --port 8000 --workers 4

Environment variables:
  APP_ENV        — prod | staging | dev (default: prod)
  LOG_LEVEL      — DEBUG | INFO | WARNING | ERROR (default: INFO)
  OPENAI_API_KEY — required for LLM calls

AWS credentials are fetched at agent initialisation time via boto3 from
the ECS task role / EC2 instance profile. No credential env vars needed.

API surface:
  POST /api/v1/{agent}/chat              — send a message
  POST /api/v1/{agent}/resume            — resume HITL conversation
  GET  /api/v1/health                    — liveness check
  GET  /api/v1/prompts/{agent}/{env}     — list prompt versions
  POST /api/v1/prompts/{agent}/{env}/activate  — activate a version
  POST /api/v1/prompts/{agent}/{env}/rollback  — rollback to previous

  GET  /docs                             — Swagger UI (disable in prod)
  GET  /openapi.json                     — OpenAPI schema
"""

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from platform.observability.logger import configure_logging
from platform.gateway.middleware import RequestContextMiddleware, TimingMiddleware
from platform.gateway.router import router as agent_router
from platform.prompt_versioning.router import router as prompt_router

# ── Logging ────────────────────────────────────────────────────────────────────
configure_logging(level=os.environ.get("LOG_LEVEL", "INFO"))

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "VS Agentic Platform",
    description = "Vidya Sankalp multi-agent AI platform — clinical trial agent and beyond.",
    version     = "0.1.0",
    # Disable docs in production — expose only on staging/dev
    docs_url    = "/docs"    if os.environ.get("APP_ENV", "prod") != "prod" else None,
    redoc_url   = "/redoc"   if os.environ.get("APP_ENV", "prod") != "prod" else None,
    openapi_url = "/openapi.json" if os.environ.get("APP_ENV", "prod") != "prod" else None,
)

# ── Middleware (outermost → innermost) ─────────────────────────────────────────
app.add_middleware(RequestContextMiddleware)
app.add_middleware(TimingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],   # Tighten to specific domains in production
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ── Routers ────────────────────────────────────────────────────────────────────
app.include_router(agent_router)
app.include_router(prompt_router)
