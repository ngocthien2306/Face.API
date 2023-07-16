from fastapi import APIRouter

# from api.user.v1.user import user_router as user_v1_router
# from api.auth.auth import auth_router
# from api.file.fileHandle import file_router
# from api.track.v1.track import track_router
from api.face.face import face_router
router = APIRouter()

router.include_router(face_router, prefix="/api/face", tags=["Face"])
# router.include_router(auth_router, prefix="/auth", tags=["Auth"])
# router.include_router(file_router, prefix="/api/file", tags=["File Handle"])
# router.include_router(track_router, prefix="/api/track", tags=["Tracking"])

__all__ = ["router"]
