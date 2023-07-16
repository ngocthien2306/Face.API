from fastapi import APIRouter, HTTPException
from api.track.v1.request.track import CheckVehicleReport, CheckVehicleRequest
from app.track.helper.embedding import Embedding
from app.track.schemas import ExceptionTrackResponseSchema
from app.track.schemas.track import TrackReportResposeSchemas, TrackVehicleResposeSchemas
from app.track.services.track import TrackingServices
from fastapi.responses import JSONResponse
## Image Recognition
import base64
import numpy as np
import cv2
from deepface import DeepFace
from datetime import datetime
from typing import Union
import os
from os.path import join, dirname, isdir
from dotenv import load_dotenv
load_dotenv()
TARGET = os.environ.get("TARGET")
MODEL_PATH = os.environ.get("MODEL_PATH")
NETWORK = os.environ.get("NETWORK")

batch_size = 32

embedding = Embedding(MODEL_PATH, (3, 112, 112), 128, network=NETWORK)
#End: Deeplearning


track_router = APIRouter()
track_services = TrackingServices(TARGET)

@track_router.post(
    "/trackingVehicle",
    response_model=TrackVehicleResposeSchemas,
    responses={"400": {"model": ExceptionTrackResponseSchema}},
)
async def trackVehicle(request: CheckVehicleRequest):
    try:
        ## Check LP 
        if not request.platenum or request.platenum == "": 
            return -1 ## ExceptionTrack
        # Check Face
        ## Read Image --> cv2 
        # face = cv2.imdecode(np.fromstring(file.file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        # lpImage = cv2.imdecode(np.fromstring(filelp.file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        face64 = request.stringFace
        lp = request.stringlp
        face = track_services.convertbase64(face64)
        lpImage = track_services.convertbase64(lp)
        ## Get path image
        curDT = datetime.now()
        date_time = curDT.strftime("%m%d%Y-%H%M%S")
        path_img_detected = TARGET + str(request.platenum)  

        if isdir(path_img_detected) == False:     
            os.makedirs(path_img_detected)
        img_detected = path_img_detected + "/face_{}.jpg".format(date_time)
        lpImage_path = path_img_detected + "/lp_{}.jpg".format(date_time)
        ## Image DetectFace
        ### If it can't be detected return exception
        print(img_detected)        
        face_detected = DeepFace.detectFace(face, detector_backend='ssd', enforce_detection=True)
        ## Save Image to path  --- Save Image Detect
        cv2.imwrite(img_detected, face_detected[:,:,::-1]*255)
        cv2.imwrite(lpImage_path, lpImage)
        print(img_detected)
        print(lpImage_path)
        ## Using track_services to write in Database
        result = await track_services.create_track_vehicle_async(
            embedding.model,
            face_detected[:,:,::-1]*255,
            lpImage_path,
            img_detected,
            request)
        print(result["status"])
        return TrackVehicleResposeSchemas(status=result["status"], fee=result["fee"])
    except Exception as e:
        print(str(e))
        return JSONResponse(content={"error": str(e)}, status_code=400)
@track_router.post(
    "/trackingReports",
    response_model=TrackReportResposeSchemas,
    responses={"400": {"model": ExceptionTrackResponseSchema}},
)
async def trackVehicle(request: CheckVehicleReport):
    try:
        face64 = request.stringFace
        lp = request.stringlp
        face = track_services.convertbase64(face64)
        lpImage = track_services.convertbase64(lp)
        ## Get path image
        curDT = datetime.now()
        date_time = curDT.strftime("%m%d%Y-%H%M%S")
        path_img_detected = TARGET + str(request.platenum)  

        if isdir(path_img_detected) == False:     
            os.makedirs(path_img_detected)
        img_detected = path_img_detected + "/face_{}.jpg".format(date_time)
        lpImage_path = path_img_detected + "/lp_{}.jpg".format(date_time)
        face_detected = DeepFace.detectFace(face, detector_backend='ssd', enforce_detection=True)
        ## Save Image to path  --- Save Image Detect
        cv2.imwrite(img_detected, face_detected[:,:,::-1]*255)
        cv2.imwrite(lpImage_path, lpImage)
        result = await track_services.create_track_report(request.siteId,request.platenum,request.typeTransport,request.typeLicensePlate,img_detected,lpImage_path)
        print(result) 
        if result == True:
            return TrackReportResposeSchemas(status=True)
        return TrackReportResposeSchemas(status=False)
    except Exception as e:
        print(str(e))
        return JSONResponse(content={"error": str(e)}, status_code=400)
