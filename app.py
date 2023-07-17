from fastapi import FastAPI
from core.config import config
from core.utils.token_helper import get_mac_address
from src.api.face.face_controller import router_face
from src.api.home.home_controller import router_root
from src.database.connect import connection, DeviceService

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
    connection.connect()
    service = DeviceService(connection)
    device = service.get_info_by_mac_address(get_mac_address())
    print(device)
    connection.disconnect()
    uvicorn.run(app, host=device[0][0], port=device[0][1])
