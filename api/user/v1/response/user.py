from pydantic import BaseModel, Field


class LoginResponse(BaseModel):
    token: str = Field(..., description="Token")
    refresh_token: str = Field(..., description="Refresh token")
class VerifylicensePlateResponse(BaseModel):
    license: str = Field(..., description="License plate")
class VerifylIdVietNamResponse(BaseModel):
    id: str = Field(..., description="ID Viet Nam")
class ResgisterIdVietNamResponse(BaseModel):
    result: bool = Field(..., description="Result of registration")