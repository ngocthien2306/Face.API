from pydantic import BaseModel, Field


class LoginRequest(BaseModel):
    email: str = Field(..., description="Email")
    password: str = Field(..., description="Password")

class VerifylicensePlateRequest(BaseModel):
    stringlp : str =  Field(..., description="License Plate Image")
class VerifyIdVietNameRequest(BaseModel):
    stringidvn : str =  Field(..., description="Id VietName Image")
class RegisterIdVietNamRequest(BaseModel):
    Cccd : str = Field(..., description="Number of Id Viet Nam")
    Gender  : str = Field(..., description="Gender")
    FullName : str = Field(..., description="Full Name")
    BirthDay : str = Field(..., description="Birth Day")
    PictureFace : str = Field(..., description="Picture Face")
    Stringidvn : str =  Field(..., description="Id VietName Image")