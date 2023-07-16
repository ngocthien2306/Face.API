from sqlalchemy import Column, Unicode, BigInteger, Boolean, Integer, String,ForeignKey,Float,DateTime
from sqlalchemy.orm import relationship
from core.db import Base
from core.db.mixins import TimestampMixin


class Track(Base):
    __tablename__ = "tblTrack"
    Id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    trackNumber = Column(Integer)
    vehicleId = Column(Integer, ForeignKey('tblVehicle.id'))
    startTime = Column(DateTime,default=True)
    endTime = Column(DateTime, default=False)
    fee = Column(Float, default=False)
    siteId= Column(Integer, default=True)
    locationX = Column(Float, default=False)
    locationY = Column(Float, default=False)
    userId = Column(Integer)
    detectInFace = Column(String(250), default=True) 
    detectOutFace = Column(String(250), default=False) 
    plateIn = Column(String(250), default=True)
    plateOut = Column(String(250), default=False)
    vehicle = relationship("Vehicle", back_populates="tracks")
    
