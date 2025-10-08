import json
import os
from fastapi import FastAPI
from src.api.routers import predict_mainrace, predict_qualifying, predict_status
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from src.preprocess.preprocess_qualifying import serve_qualifying_df

load_dotenv()

serve_qualifying_df()
ALLOWED_ORIGINS: list = json.loads(os.getenv("ALLOWED_ORIGINS"))
APP_MODE: str = os.getenv("APP_MODE", "dev")

def create_app() -> FastAPI:
    docs_url = None if APP_MODE == "prod" else "/docs"  # disables docs
    redoc_url = None if APP_MODE == "prod" else "/redoc"  # disables redoc
    openapi_url = None if APP_MODE == "prod" else "/openapi.json"  # disables openapi.json suggested by tobias comment.
    created_app = FastAPI(title="f1-fantasy-ml-api",docs_url=docs_url, redoc_url=redoc_url, openapi_url=openapi_url)

    created_app.include_router(predict_mainrace.router, prefix="/api")
    created_app.include_router(predict_qualifying.router, prefix="/api")
    created_app.include_router(predict_status.router, prefix="/api")
    return created_app

app = create_app()



app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
