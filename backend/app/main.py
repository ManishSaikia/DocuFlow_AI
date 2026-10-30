from app.core.config import settings
from fastapi import FastAPI
from app.api.v1.router import api_router

from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from app.core.tracing import tracer

app = FastAPI(
    title=settings.PROJECT_NAME,
    version="1.0.0",
    description="Event-Driven Unstructured Document Extraction & Hybrid RAG Engine"
)

app.include_router(api_router,prefix='/api/v1')
FastAPIInstrumentor.instrument_app(app)

@app.get("/health", tags=["System"])
async def health():
    return {
        "status": "healthy",
        "project": settings.PROJECT_NAME,
        "version": "1.0.0"
    }