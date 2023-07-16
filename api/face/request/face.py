from pydantic import BaseModel, Field


class FaceExtractRequest(BaseModel):
    facebase64: str = Field(..., description="License Plate Image")


class ChangeModelRequest(BaseModel):
    network: str = Field(..., description="Model Type")
