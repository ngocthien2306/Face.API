from sqlalchemy import Column, Unicode, BigInteger, Boolean

from core.db import Base
from core.db.mixins import TimestampMixin


class User(Base, TimestampMixin):
    __tablename__ = "tblUser"

    Id = Column(BigInteger, primary_key=True, autoincrement=True)
    Password = Column(Unicode(255), nullable=False)
    Email = Column(Unicode(255), nullable=False, unique=True)
    NickName = Column(Unicode(255), nullable=False, unique=True)
    IsAdmin = Column(Boolean, default=False)
