import threading
from core.utils.token_helper import get_mac_address
from sockets.services.socket_services import FaceServices
from fastapi import WebSocket
from fastapi import APIRouter

from src.database.connect import connection, DeviceService

router_face = APIRouter()

face_service = None
real_time_thread = None


def run_real_time_check_in(net, th):
    global face_service
    face_service = FaceServices(threshold=th, network=net, update=True)
    face_service.real_time_check_in()


def start_real_time_check_in(net, th):
    global real_time_thread
    real_time_thread = threading.Thread(target=run_real_time_check_in, args=(net, th))
    real_time_thread.start()


def stop_real_time_check_in():
    global face_service, real_time_thread
    if face_service is not None:
        face_service.stop_real_time_check_in()
        face_service = None
    if real_time_thread is not None:
        real_time_thread.join()
        real_time_thread = None


@router_face.get("/start-realtime")
def start_realtime():
    stop_real_time_check_in()
    connection.connect()
    service = DeviceService(connection)
    device = service.get_by_mac_address(get_mac_address())
    print(device)
    connection.disconnect()
    if len(device) > 0:
        start_real_time_check_in(device[0][-2], device[0][-1])
        return {"status": "success", "message": "Real-time check-in started"}
    else:
        return {"status": "error", "message": "You not have permission"}



@router_face.get("/stop-realtime")
def stop_realtime():
    stop_real_time_check_in()
    return {"status": "success", "message": "Real-time check-in stopped"}


@router_face.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        if data == "start_realtime":
            start_realtime()
        elif data == "stop_realtime":
            stop_realtime()
        await websocket.send_text(f"Received: {data}")