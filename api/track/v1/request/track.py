from pydantic import BaseModel, Field

class CheckVehicleRequest(BaseModel):
    platenum: str = Field(..., description="License Plate")
    typeTransport: str = Field(..., description="Type Transport")
    typeLicensePlate: str =  Field(..., description="Type License Plate")
    stringFace: str =  Field(..., description="Face Image")
    stringlp : str =  Field(..., description="License Plate Image")
    siteId: str = Field(..., description="Parking ID")
class CheckVehicleReport(BaseModel):
    platenum: str = Field(..., description="License Plate")
    siteId: int = Field(..., description="Parking ID")
    typeTransport: str = Field(..., description="Type Transport")
    typeLicensePlate: str =  Field(..., description="Type License Plate")
    stringFace: str =  Field(..., description="Face Image")
    stringlp : str =  Field(..., description="License Plate Image")
    