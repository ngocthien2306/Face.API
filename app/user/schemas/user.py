from pydantic import BaseModel, Field
from datetime import datetime

class GetUserListResponseSchema(BaseModel):
    Id: int = Field(..., description="ID")
    Email: str = Field(..., description="Email")
    NickName: str = Field(..., description="Nickname")
    CreatedAt: datetime = Field(..., description="Created Day")
    UpdatedAt: datetime = Field(..., description="Updated Day")
    
    class Config:
        orm_mode = True

class CreateUserRequestSchema(BaseModel):
    email: str = Field(..., description="Email")
    password1: str = Field(..., description="Password1")
    password2: str = Field(..., description="Password2")
    nickname: str = Field(..., description="Nickname")


class CreateUserResponseSchema(BaseModel):
    email: str = Field(..., description="Email")
    nickname: str = Field(..., description="Nickname")

    class Config:
        orm_mode = True


class LoginResponseSchema(BaseModel):
    token: str = Field(..., description="Token")
    refresh_token: str = Field(..., description="Refresh token")
