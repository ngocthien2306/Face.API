import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File
from starlette.responses import JSONResponse
import traceback
from api.face.request.face import FaceExtractRequest, ChangeModelRequest
from sockets.services.socket_services import FaceServices
import traceback

from src.Learner import face_learner

face_router = APIRouter()

face_service = FaceServices()

@face_router.post("/face-extract")
async def face_extract(request: FaceExtractRequest):
    if not request.facebase64 or request.facebase64 == "":
        return {"id": "Not Found"}
    try:
        print(request.facebase64[:10])
        img = face_service.convertbase64(request.facebase64)
        res = face_service.recognize_user(img[:, :, ::-1])
        return res
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(content={"error": str(e)}, status_code=400)


@face_router.post("/face-extract-v2")
async def face_extract(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return face_service.recognize_user(image)
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(content={"error": str(e)}, status_code=400)


@face_router.get('/open-program')
async def open_program():
    face_service.threshold = 1.3
    face_service.conf.network = 'r18'
    face_service.conf.update = True
    await face_service.real_time_check_in()
    return {'status': 'success', 'message': "open program success"}


@face_router.post("/change-model")
async def change_model(request: ChangeModelRequest):
    try:
        if not request.network or request.network == "":
            return {
                "status": "error",
                "message": "network field is required"
            }
        if request.network not in ['r100', 'r34', 'mbf', 'vit', 'r18', 'ir_se50', 'mobilefacenet']:
            return {
                "status": "error",
                "message": "model is invalid"
            }
        face_service.conf.network = request.network
        face_service.learner = face_learner(face_service.conf, True)
        return {
            "status": "success",
            "message": "synchronous successfully, please wait a second to reload"
        }
    except Exception as e:
        return JSONResponse(content={'status': 'error', "message": str(e)}, status_code=400)
