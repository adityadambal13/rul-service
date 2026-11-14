"""FastAPI application."""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response

from rul_service import config
from rul_service.model import load_bundle
from rul_service.predictor import InputError, Predictor
from rul_service.schemas import ModelInfo, PredictRequest, PredictResponse
from rul_service.utils import logger


def create_app(bundle_path: Path | None = None) -> FastAPI:
    path = bundle_path or config.model_path()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.predictor = None
        if path.exists():
            app.state.predictor = Predictor(load_bundle(path))
            logger.info(json.dumps({"event": "model_loaded", "path": str(path)}))
        else:
            logger.warning(json.dumps({"event": "model_missing", "path": str(path)}))
        yield

    app = FastAPI(title="RUL Service", version="1.0.0", lifespan=lifespan)

    @app.middleware("http")
    async def log_requests(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = request.headers.get("X-Request-ID", uuid.uuid4().hex)
        start = time.perf_counter()
        response = await call_next(request)
        latency_ms = round((time.perf_counter() - start) * 1000, 1)
        response.headers["X-Request-ID"] = request_id
        logger.info(
            json.dumps(
                {
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status": response.status_code,
                    "latency_ms": latency_ms,
                }
            )
        )
        return response

    def get_predictor(request: Request) -> Predictor:
        predictor: Predictor | None = request.app.state.predictor
        if predictor is None:
            raise HTTPException(status_code=503, detail="model not loaded")
        return predictor

    @app.get("/health")
    def health() -> dict[str, str]:
        """Liveness: the process is up."""
        return {"status": "ok"}

    @app.get("/ready")
    def ready(request: Request) -> dict[str, str]:
        """Readiness: the model is loaded and the service can answer predictions."""
        predictor = get_predictor(request)
        return {"status": "ready", "model_version": predictor.bundle.version}

    @app.get("/v1/model-info", response_model=ModelInfo)
    def model_info(request: Request) -> ModelInfo:
        meta = get_predictor(request).bundle.metadata
        return ModelInfo(
            model_version=str(meta["model_version"]),
            subset=str(meta["subset"]),
            trained_at=str(meta["trained_at"]),
            window=int(meta["window"]),
            rul_cap=int(meta["rul_cap"]),
            interval=str(meta["interval"]),
            n_train_engines=int(meta["n_train_engines"]),
            sensors_kept=list(meta["sensors_kept"]),
        )

    # A plain `def` endpoint runs in a worker thread, so model inference
    # does not block the server's event loop.
    @app.post("/v1/predict", response_model=PredictResponse)
    def predict(body: PredictRequest, request: Request) -> PredictResponse:
        predictor = get_predictor(request)
        readings = [r.model_dump() for r in body.readings]
        try:
            result = predictor.predict(body.engine_id, readings)
        except InputError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return PredictResponse(**result.__dict__)

    return app


app = create_app()
