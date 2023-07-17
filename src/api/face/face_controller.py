import shutil
import threading
import time
import zipfile
from typing import List
import os
import base64
import requests
from core.utils.token_helper import get_mac_address
from sockets.services.socket_services import FaceServices
from fastapi import WebSocket
from fastapi import APIRouter
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from src.database.connect import connection, DeviceService


router_face = APIRouter()

face_service = None
real_time_thread = None

print("Mac Address: ", get_mac_address())


def add_images_to_folders(folder_path, api_url):

    # Get a list of folder names in the specified directory
    folder_names = os.listdir(folder_path)

    # Make a POST request to the server API to check folder contents
    response = requests.post(api_url, json={"folders": folder_names})
    if response.status_code != 200:
        return

    data = response.json()
    start_time = time.time()

    for idx, folder in enumerate(data['result']):
        image_base64 = data["result"][folder][idx]['image_base64']
        image_data = base64.b64decode(image_base64)

        os.makedirs(os.path.join(folder_path, folder), exist_ok=True)
        # Save the image to the specified folder
        image_path = os.path.join(folder_path, folder,
                                  data["result"][folder][0]['file_name'])  # Change the extension if needed
        with open(image_path, "wb") as f:
            f.write(image_data)


        print(image_data[:10])
        os.makedirs(os.path.join(folder_path, folder), exist_ok=True)
        # Save the image to the specified folder
        image_path = os.path.join(folder_path, folder, data["result"][folder][0]['file_name'])  # Change the extension if needed
        with open(image_path, "wb") as f:
            f.write(image_data)

    end_time = time.time()
    print(end_time - start_time)
    print("Images added to the folders successfully.")



def download_zip():
    folder_user = "./public/images/users"
    endpoint = "http://26.115.12.45:8005/api/v1/users/downloadfolder"
    folder_names = os.listdir(folder_user)
    print(folder_names)
    response = requests.post(endpoint, json={"folders": folder_names}, stream=True)

    if response.status_code == 200:
        save_path = os.path.join(folder_user, "/downloaded.zip")
        with open(save_path, "wb") as file:
            response.raw.decode_content = True
            shutil.copyfileobj(response.raw, file)
        print("Zip file downloaded successfully.")

        with zipfile.ZipFile(save_path, 'r') as zip_ref:
            zip_ref.extractall(folder_user)
    else:
        print("Error downloading zip file:", response.status_code)


# folder_path = "./public/images/users"
# api_url = "http://26.115.12.45:8005/api/v1/users/checkfaceuserfolder"
# add_images_to_folders(folder_path, api_url)

download_zip()



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
        start_real_time_check_in(device[0][-5], device[0][-4])
        return {"status": "success", "message": "Real-time check-in started"}
    else:
        return {"status": "error", "message": "You not have permission"}


@router_face.post('/addusertofolder')
async def add_user_to_folder(user_ids: List[str], files: List[UploadFile] = File(...)):
    folder_path = "./public/images/users"  # Replace with the actual folder path
    os.makedirs(folder_path, exist_ok=True)  # Create the main folder if it doesn't exist
    for user_id, file in zip(user_ids, files):
        user_folder_path = os.path.join(folder_path, user_id)
        os.makedirs(user_folder_path, exist_ok=True)  # Create the user's folder if it doesn't exist

        file_path = os.path.join(user_folder_path, f"{user_id}_{file.filename}")
        with open(file_path, "wb") as f:
            f.write(await file.read())

    return JSONResponse(content={"message": "Files uploaded successfully"})


@router_face.get('/synchronous-user')
async def synchronous_user():
    try:
        download_zip()
        return {"status": True}
    except:
        return {"status": False}

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
