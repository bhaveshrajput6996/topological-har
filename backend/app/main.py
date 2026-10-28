"""FastAPI application factory."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from har.config.schema import ConfigError
from har.experiments.inspection import InspectionError
from har.inference.predictor import PredictorError
from har.models.loader import ArtifactError
from har.tda.engine import TdaEngineError
from har.utils.logging import get_logger

from .api.routes import router

logger = get_logger(__name__)

_STATUS_BY_ERROR = {
    ConfigError: 400,
    TdaEngineError: 400,
    PredictorError: 400,
    ArtifactError: 404,
    InspectionError: 400,
    FileNotFoundError: 404,
    KeyError: 404,
}


def create_app() -> FastAPI:
    app = FastAPI(
        title="Topological Deep Learning HAR backend",
        version=__import__("har.version", fromlist=["__version__"]).__version__,
        description=(
            "Human Activity Recognition backend with a Vietoris-Rips "
            "persistent-homology branch fused into a DeepConvLSTM model."
        ),
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)

    @app.exception_handler(ConfigError)
    async def _config_error(request: Request, exc: ConfigError) -> JSONResponse:
        return JSONResponse({"error": "invalid configuration", "detail": str(exc)}, status_code=400)

    @app.exception_handler(KeyError)
    async def _key_error(request: Request, exc: KeyError) -> JSONResponse:
        return JSONResponse({"error": "not found", "detail": str(exc)}, status_code=404)

    @app.exception_handler(Exception)
    async def _generic_error(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled API error")
        return JSONResponse(
            {"error": type(exc).__name__, "detail": str(exc)}, status_code=500
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse({"error": "validation error", "detail": exc.errors()}, status_code=422)

    @app.on_event("startup")
    async def _startup() -> None:
        logger.info("tdl-har backend starting")

    @app.get("/")
    async def root() -> dict[str, Any]:
        return {
            "service": "tdl-har-backend",
            "docs": "/docs",
            "health": "/health",
        }

    return app


app = create_app()
