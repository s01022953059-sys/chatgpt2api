from __future__ import annotations

import re
from contextlib import asynccontextmanager
from threading import Event
from urllib.parse import unquote

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from starlette.middleware.base import BaseHTTPMiddleware

from api import accounts, ai, image_tasks, system
from api.errors import install_exception_handlers
from api.support import resolve_web_asset, start_limited_account_watcher
from services.backup_service import backup_service
from services.config import config
from services.image_service import start_image_cleanup_scheduler


def create_app() -> FastAPI:
    app_version = config.app_version

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        stop_event = Event()
        thread = start_limited_account_watcher(stop_event)
        cleanup_thread = start_image_cleanup_scheduler(stop_event)
        backup_service.start()
        config.cleanup_old_images()
        try:
            yield
        finally:
            stop_event.set()
            thread.join(timeout=1)
            cleanup_thread.join(timeout=1)
            backup_service.stop()

    app = FastAPI(title="chatgpt2api", version=app_version, lifespan=lifespan)
    install_exception_handlers(app)

    class PathRewriteMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            raw_path = request.scope.get("path") or "/"
            decoded = unquote(raw_path)
            stripped = re.match(r"^https?://[^/]+(/.*)$", decoded)
            if stripped:
                request.scope["path"] = stripped.group(1)
                request.scope["raw_path"] = None
            return await call_next(request)

    # Register FIRST so that it executes LAST (Starlette wraps middleware in
    # reverse order around the application). It must run before the CORS
    # middleware so rewritten paths are observable to downstream handlers.
    app.add_middleware(PathRewriteMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(ai.create_router())
    app.include_router(accounts.create_router())
    app.include_router(image_tasks.create_router())
    app.include_router(system.create_router(app_version))

    @app.api_route("/{full_path:path}", methods=["GET", "HEAD"], include_in_schema=False)
    async def serve_web(full_path: str):
        asset = resolve_web_asset(full_path)
        if asset is not None:
            response = FileResponse(asset)
            if full_path.strip("/").startswith("_next/"):
                response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            return response
        if full_path.strip("/").startswith("_next/"):
            raise HTTPException(status_code=404, detail="Not Found")
        fallback = resolve_web_asset("")
        if fallback is None:
            raise HTTPException(status_code=404, detail="Not Found")
        response = FileResponse(fallback)
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        return response

    return app
