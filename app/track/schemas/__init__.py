
from pydantic import BaseModel
class ExceptionTrackResponseSchema(BaseModel):
    error: str
