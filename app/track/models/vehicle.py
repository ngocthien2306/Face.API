from sqlalchemy import Boolean, Column, ForeignKey, Integer, String, DateTime
from sqlalchemy.orm import relationship

from core.db import Base


class Vehicle(Base):
    __tablename__ = "tblVehicle"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    plateNum =Column(String(20), default=True)
    status =Column(String(10), default=True)
    typeTransport = Column(String(10), default=True)
    typePlate = Column(String(10), default=True)
    tracks = relationship("Track", back_populates="vehicle") # tạo quan hệ