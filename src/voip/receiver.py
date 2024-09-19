import uvicorn
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from pyVoIP.VoIP import VoIPPhone
import wave
import uuid
import numpy as np
import threading
from call_handler import CallHandler
import os,sys
# Add the project root directory to sys.path and then import llm_handler from callbot
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from dbconn import models
# from dbconn.database import SessionLocal, engine

app = FastAPI()

# # Create the database tables
# models.Base.metadata.create_all(bind=engine)

# # Dependency
# def get_db():
#     db = SessionLocal()
#     try:
#         yield db
#     finally:
#         db.close()

# Function to handle each call by creating a CallHandler instance
def handle_call_in_thread(call):
    call_handler = CallHandler()
    threading.Thread(target=call_handler.handle_call, daemon=True, args=(call,)).start()

def start_voip():
    phone = VoIPPhone(
        # server="192.168.88.5",
        server="192.168.98.48",

        port=5060,
        username="5001",
        password="iX3TxsD9jWxmZU5",
        myIP="192.168.88.10",
        callCallback=handle_call_in_thread
    )

    phone.start()
    # phone.call('5000')
    print("Phone started. Status:", phone.get_status())

    input('Press anything to exit')
    phone.stop()
    print("Phone stopped")

start_voip()
# def run_fastapi():
#     uvicorn.run(app, host="0.0.0.0", port=8002)

# @app.get("/")
# def index():
#     return {
#         "/": "Default Endpoint",
#         "/call_logs/":"get logs"
#     }

# @app.post("/call_logs/")
# def create_call_log(caller_id:str, event_type: str, event_detail: str = None, db: Session = Depends(get_db)):
#     db_call_log = models.CallLog(caller_id=caller_id, event_type=event_type, event_detail=event_detail)
#     db.add(db_call_log)
#     db.commit()
#     db.refresh(db_call_log)
#     return db_call_log

# @app.get("/call_logs/")
# def read_call_logs(skip: int = 0, limit: int = 10, db: Session = Depends(get_db)):
#     return db.query(models.CallLog).offset(skip).limit(limit).all()

# Start FastAPI server in a separate thread
# threading.Thread(target=start_voip,daemon=False).start()
# threading.Thread(target=run_fastapi).start()

