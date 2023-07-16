from sqlalchemy import Boolean, Column, ForeignKey, Integer, String, DateTime
from sqlalchemy.orm import relationship
from core.db import Base
class Guest(Base):
    __tablename__ = "tblGuest"
    driverId = Column(Integer, primary_key=True, index=True)
    originPathFace =Column(String(100), default=True)
    detectPathFace =Column(String(100), default=False)
    status = Column(String(10), default=True)