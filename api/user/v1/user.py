from typing import List
import base64
import numpy as np
import cv2
import io
from PIL import Image
from fastapi import APIRouter, Depends, Query, UploadFile,File
from fastapi.responses import JSONResponse
from api.user.v1.request.user import LoginRequest, RegisterIdVietNamRequest, VerifyIdVietNameRequest, VerifylicensePlateRequest
from api.user.v1.response.user import LoginResponse, ResgisterIdVietNamResponse, VerifylIdVietNamResponse, VerifylicensePlateResponse
from app.user.schemas import (
    ExceptionResponseSchema,
    GetUserListResponseSchema,
    CreateUserRequestSchema,
    CreateUserResponseSchema,
)
from app.user.services import UserService
from core.fastapi.dependencies import (
    PermissionDependency,
    IsAdmin,
    AllowAll
)

user_router = APIRouter()


@user_router.get(
    "",
    response_model=List[GetUserListResponseSchema],
    response_model_exclude={"id"},
    responses={"400": {"model": ExceptionResponseSchema}},
    dependencies=[Depends(PermissionDependency([IsAdmin]))],
)
async def get_user_list(
    limit: int = Query(10, description="Limit"),
    prev: int = Query(None, description="Prev ID"),
):
    return await UserService().get_user_list(limit=limit, prev=prev)


@user_router.post(
    "",
    response_model=CreateUserResponseSchema,
    responses={"400": {"model": ExceptionResponseSchema}},
)
async def create_user(request: CreateUserRequestSchema):
    await UserService().create_user(**request.dict())
    return {"email": request.email, "nickname": request.nickname}


@user_router.post(
    "/login",
    response_model=LoginResponse,
    responses={"404": {"model": ExceptionResponseSchema}},
)
async def login(request: LoginRequest):
    token = await UserService().login(email=request.email, password=request.password)
    return {"token": token.token, "refresh_token": token.refresh_token}
@user_router.post(
    "/verifylicenseplates",
    response_model=VerifylicensePlateResponse,
    responses={"404": {"model": ExceptionResponseSchema}},
)
async def verify(request: VerifylicensePlateRequest):
    if not request.stringlp or request.stringlp == "": 
            return {"license": "Not Found"} ## ExceptionTrack
    try:
        # Convert string to Image
        decoded_data = base64.b64decode(request.stringlp)
        np_data = np.fromstring(decoded_data, np.uint8)
        image = cv2.imdecode(np_data, cv2.IMREAD_UNCHANGED)
        
        
        # Lp returned to LP Dictionary so Check accuracy here
        lp = await UserService().verify_lp(imagelp=image)
        
        return {"license": str(lp)}
    except Exception as e:
        print(str(e))
        return JSONResponse(content={"error": str(e)}, status_code=400)
@user_router.post(
    "/verifyidvn",
    response_model=VerifylIdVietNamResponse,
    responses={"404": {"model": ExceptionResponseSchema}},
)    
async def verify(request: VerifyIdVietNameRequest):
    if not request.stringidvn or request.stringidvn == "": 
            return {"id": "Not Found"} ## ExceptionTrack
    try:
        # Convert string to Image
        decoded_data = base64.b64decode(request.stringidvn)
        # np_data = np.fromstring(decoded_data, np.uint8)
        # image = cv2.imdecode(np_data, cv2.IMREAD_UNCHANGED)
        # image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        image_io = io.BytesIO(decoded_data)
        image = Image.open(image_io)
        img = np.asarray(image)
        result = await UserService().verify_id(imageId=img)
        return {"id": str(result)}
    except Exception as e:
        print(str(e))
        return JSONResponse(content={"error": str(e)}, status_code=400)
    
@user_router.post(
    "/verifyidvn2"
)    
async def verify(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"

    
    try:
        image = UserService().read_image_file(await file.read())
        img = np.asarray(image)
        result = await UserService().verify_id(imageId=img)
        return {"id": str(result)}
    except Exception as e:
        print(str(e))
        return JSONResponse(content={"error": str(e)}, status_code=400)
@user_router.post(
    "/registerCCCD",
    response_model=ResgisterIdVietNamResponse,
    responses={"404": {"model": ExceptionResponseSchema}},
)    
async def registerCCCD(request: RegisterIdVietNamRequest):
    if not request.BirthDay or request.BirthDay == "" or not request.FullName or request.FullName =="" or not request.Gender  or request.Gender == "" or not request.Cccd or len(request.Cccd) != 12: 
            return JSONResponse(content={"error": "Null Data"}, status_code=400)
    try:
        # Convert string to Image
        decoded_data = base64.b64decode(request.Stringidvn)
        np_data = np.fromstring(decoded_data, np.uint8)
        imageid = cv2.imdecode(np_data, cv2.IMREAD_UNCHANGED)
        imageid = cv2.cvtColor(imageid, cv2.COLOR_RGBA2RGB)
        
        decoded_data = base64.b64decode(request.PictureFace)
        np_data = np.fromstring(decoded_data, np.uint8)
        imageface = cv2.imdecode(np_data, cv2.IMREAD_UNCHANGED)
        imageface = cv2.cvtColor(imageface, cv2.COLOR_RGBA2RGB)
        result = await UserService().register_user(cccd=request.Cccd, fullname=request.FullName,Gender = request.Gender,imageId=imageid,imageFace=imageface )
        return {"result": result}
    except Exception as e:
        print(str(e))
        return JSONResponse(content={"error": str(e)}, status_code=400)
@user_router.post(
    "/registerIdentity",
    response_model=VerifylIdVietNamResponse,
    responses={"404": {"model": ExceptionResponseSchema}},
)    
async def verifyId(request: VerifyIdVietNameRequest):
    if not request.stringidvn or request.stringidvn == "": 
            return {"id": "Not Found"} ## ExceptionTrack
    try:
        # Convert string to Image
        decoded_data = base64.b64decode(request.stringidvn)
        # np_data = np.fromstring(decoded_data, np.uint8)
        # image = cv2.imdecode(np_data, cv2.IMREAD_UNCHANGED)
        # image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        image_io = io.BytesIO(decoded_data)
        image = Image.open(image_io)
        img = np.asarray(image)
        result = await UserService().verify_idImage(imageId=img)
        return {"id": str(result)}
    except Exception as e:
        print(str(e))
        return JSONResponse(content={"error": str(e)}, status_code=400)
    