from fastapi import FastAPI
from core.config import config
from src.api.face.face_controller import router_face
from src.api.home.home_controller import router_root

app = FastAPI(
    title="Hide",
    description="Hide API",
    version="1.0.0",
    docs_url=None if config.ENV == "production" else "/docs",
    redoc_url=None if config.ENV == "production" else "/redoc",
)

app.include_router(router_root)
app.include_router(router_face, prefix="/api/face")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="26.115.12.45", port=8000)
