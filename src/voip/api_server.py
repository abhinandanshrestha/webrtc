import os,sys
# Add the project root directory to sys.path and then import llm_handler from callbot
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dbconn import models
from dbconn.database import SessionLocal, engine
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
import uvicorn
from call_handler import CallHandler
# Create the database tables
models.Base.metadata.create_all(bind=engine)

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

app = FastAPI()

@app.post("/call_logs/")
def create_call_log(caller_id:str, event_type: str, event_detail: str = None, db: Session = Depends(get_db)):
    db_call_log = models.CallLog(caller_id=caller_id, event_type=event_type, event_detail=event_detail)
    db.add(db_call_log)
    db.commit()
    db.refresh(db_call_log)
    return db_call_log

@app.get("/call_logs/")
def read_call_logs(skip: int = 0, limit: int = 10, db: Session = Depends(get_db)):
    return db.query(models.CallLog).offset(skip).limit(limit).all()

uvicorn.run(app, host="0.0.0.0", port=8001)

