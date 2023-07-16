from pydantic import BaseModel

class GetPictureResponse(BaseModel):
    content: bytes
    media_type: str
    
class ExceptionPictureResponse(BaseModel):
    status_code: str
    content: dict
    