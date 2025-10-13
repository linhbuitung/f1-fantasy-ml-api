import uvicorn
import os
from fastapi import FastAPI, Request, HTTPException
from app.api.routers import predict_mainrace, predict_qualifying, predict_status
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from app.utils import dvc_pull_with_gcp_key

dvc_pull_with_gcp_key()
# APP_MODE controls whether docs/openapi are exposed. Default to dev for local runs.
APP_MODE: str = os.getenv("APP_MODE", "dev")
# API_KEY is optional at import time to allow local/dev runs. Middleware will enforce it only if set.
API_KEY: str | None = os.getenv("API_KEY")

if APP_MODE == "dev":
    load_dotenv()

def create_app() -> FastAPI:
    docs_url = None if APP_MODE == "prod" else "/docs"  # disables docs
    redoc_url = None if APP_MODE == "prod" else "/redoc"  # disables redoc
    openapi_url = None if APP_MODE == "prod" else "/openapi.json"  # disables openapi.json suggested by tobias comment.
    created_app = FastAPI(title="f1-fantasy-ml-api",docs_url=docs_url, redoc_url=redoc_url, openapi_url=openapi_url)

    created_app.include_router(predict_mainrace.router)
    created_app.include_router(predict_qualifying.router)
    created_app.include_router(predict_status.router)

    # lightweight health endpoint that does not require an API key
    @created_app.get("/health", include_in_schema=False)
    def _health():
        return {"status": "ok"}
    return created_app

app = create_app()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def verify_api_key(request: Request, call_next):
    # let health endpoint through without API key
    if request.url.path == "/_health":
        return await call_next(request)

    # If API_KEY is set in environment, require matching header. If not set, allow requests (useful for dev).
    if APP_MODE == "prod" and API_KEY:
        header = request.headers.get("Ml-API-Key") or request.headers.get("ml-api-key")
        if not header or header != API_KEY:
            raise HTTPException(status_code=401, detail="Missing or invalid API key")
    return await call_next(request)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port)