from sqlalchemy import Column, Integer, String, DateTime
from .database import Base
from datetime import datetime, timezone

class CallLog(Base):
    __tablename__ = "call_logs"

    id = Column(Integer, primary_key=True, index=True)
    caller_id = Column(String, index=True)
    event_type = Column(String, index=True)
    event_detail = Column(String, nullable=True)
    inserted_on_utc = Column(DateTime, default=lambda: datetime.now(timezone.utc))
