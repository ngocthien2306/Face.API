from pydantic import BaseModel, Field

class TrackVehicleResposeSchemas(BaseModel):
    status: str = Field(..., description="Stutus Vehicle in Parking")
    fee: str = Field(..., description="Fee")
    
    class Config:
        orm_mode = True
        fields = {
            'fee': ('fee_vnd', lambda x: '{} VND'.format(x) if x else None),
        }
class TrackReportResposeSchemas(BaseModel):
    status: str = Field(..., description="Stutus Report in Parking")

    