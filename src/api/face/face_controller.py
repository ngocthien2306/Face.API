import shutil
import threading
from typing import List

from core.utils.token_helper import get_mac_address
from sockets.services.socket_services import FaceServices
from fastapi import WebSocket
from fastapi import APIRouter
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import os
from src.database.connect import connection, DeviceService

router_face = APIRouter()

face_service = None
real_time_thread = None

print("Mac Address: ", get_mac_address())

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


@router_face.post('/addusertofolder')
async def add_user_to_folder(user_ids: List[str], files: List[UploadFile] = File(...)):
    try:
        folder_path = "./public/images/users"  # Replace with the actual folder path
        os.makedirs(os.path.join(folder_path, user_ids[0].split('_')[1]),
                    exist_ok=True)  # Create the folder if it doesn't exist

        for user_id, file in zip(user_ids, files):
            file_path = os.path.join(folder_path, f"{user_id}_{file.filename}")
            with open(file_path, "wb") as f:
                f.write(await file.read())

        return JSONResponse(content={"message": "Files uploaded successfully", 'status': True})
    except Exception as e:
        return JSONResponse(content={"message": "Files uploaded fail", 'status': False})


@router_face.get('/listfiles')
async def list_files():
    folder_path = "./public/images/users"  # Replace with the actual folder path
    files = os.listdir(folder_path)
    return {"files": files}


@router_face.get('/listusers')
async def list_users():
    folder_path = "./public/images/users"  # Replace with the actual folder path
    files = os.listdir(folder_path)
    user_ids = [file.split("_")[0] for file in files]
    return {"user_ids": user_ids}


@router_face.post('/addtracktofolder')
async def add_track_to_folder(user_ids: List[str], files: List[UploadFile] = File(...)):
    try:
        folder_path = "./public/images/nouser"  # Replace with the actual folder path
        os.makedirs(os.path.join(folder_path, user_ids[0].split('_')[1]),
                    exist_ok=True)  # Create the folder if it doesn't exist

        for user_id, file in zip(user_ids, files):
            file_path = os.path.join(folder_path, f"{user_id}_{file.filename}")
            with open(file_path, "wb") as f:
                f.write(await file.read())

        return JSONResponse(content={"message": "Files uploaded successfully", 'status': True})
    except:
        return JSONResponse(content={"message": "Files uploaded fail", 'status': False})


@router_face.post('/deleteusertofolder')
async def delete_user_to_folder(user_ids: List[str]):
    folder_path = "./public/images/nouser"  # Replace with the actual folder path
    for user_id in user_ids:
        user_folder_path = os.path.join(folder_path, user_id.split('_')[1])
        shutil.rmtree(user_folder_path, ignore_errors=True)

    return JSONResponse(content={"message": "Folders deleted successfully"})


@router_face.post('/deletetracktofolder')
async def delete_track_to_folder(user_ids: List[str]):
    folder_path = "./public/images/users"  # Replace with the actual folder path
    for user_id in user_ids:
        user_folder_path = os.path.join(folder_path, user_id.split('_')[1])
        shutil.rmtree(user_folder_path, ignore_errors=True)

    return JSONResponse(content={"message": "Folders deleted successfully"})


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
