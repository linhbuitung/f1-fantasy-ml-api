import json
import os
from fastapi import FastAPI, Request, HTTPException
from app.api.routers import predict_mainrace, predict_qualifying, predict_status
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

APP_MODE: str = os.getenv("APP_MODE", "dev")
API_KEY: str = os.getenv("API_KEY")
if(API_KEY is None or API_KEY == ""):
    raise ValueError("API_KEY environment variable must be set")

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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def verify_api_key(request: Request, call_next):
    if  request.headers.get("X-API-Key") is None or request.headers.get("X-API-Key") != API_KEY:
        raise HTTPException(status_code=403, detail="Forbidden")
    return await call_next(request)