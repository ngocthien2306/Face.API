from pydantic import BaseModel, Field


class CheckVehicleResponse(BaseModel):
    status: str = Field(..., description="Stutus Vehicle in Parking")
    fee: str = Field(..., description="Fee")
    
