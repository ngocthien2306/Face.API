from fastapi import APIRouter
from fastapi.responses import FileResponse
from app.file.schemas.file import (
    ExceptionPictureResponse
)
from core.config import IMAGE_NOT_FOUND_PATH
import os
FOLDER_IMAGE = '/data/thinhlv/hung/Capstone/ML.API/public/images/'

file_router = APIRouter()
print("Wellcome to file controller")
@file_router.get(
    "/picture",
    response_model_exclude={"image_path"},
    responses={"400": {"model": ExceptionPictureResponse}},
)
async def get_picture(image_path: str):
    try:
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
        image_files = []
        if image_path == '' or image_path is None:
            return FileResponse(IMAGE_NOT_FOUND_PATH, media_type="image/jpeg")
        for dirpath, _, filenames in os.walk(FOLDER_IMAGE):
            for file in filenames:
                _, extension = os.path.splitext(file)
                if extension.lower() in image_extensions:
                    image_files.append(os.path.join(dirpath, file))

        found_paths = list(filter(lambda path: image_path in path, image_files))
        if len(found_paths) == 0:
            return FileResponse(IMAGE_NOT_FOUND_PATH, media_type="image/jpeg")

        # If multiple images with the same name are found, choose the first one
        return FileResponse(found_paths[0], media_type="image/jpeg")
    except:
        return FileResponse(IMAGE_NOT_FOUND_PATH, media_type="image/jpeg")
