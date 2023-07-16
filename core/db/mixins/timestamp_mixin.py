from sqlalchemy import Column, DateTime, func
from sqlalchemy.ext.declarative import declared_attr


class TimestampMixin:
    @declared_attr
    def CreatedAt(cls):
        return Column(DateTime, default=func.now(), nullable=False)

    @declared_attr
    def UpdatedAt(cls):
        return Column(
            DateTime,
            default=func.now(),
            onupdate=func.now(),
            nullable=False,
        )
