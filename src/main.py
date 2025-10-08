from fastapi import FastAPI
from src.api.routers import predict

def create_app() -> FastAPI:
    created_app = FastAPI(title="f1-fantasy-ml-api")
    created_app.include_router(predict.router, prefix="/api")
    return created_app

app = create_app()