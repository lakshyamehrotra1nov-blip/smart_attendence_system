import cv2
import threading
import asyncio
from fastapi import FastAPI, Request, Form, File, UploadFile, Depends, HTTPException, status, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
import uvicorn
import numpy as np

import sys
import os
import secrets
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.crud import get_today_attendance, register_student
from database.models import SessionLocal
from attendance.matcher import AttendanceMatcher
from datetime import datetime

app = FastAPI()

app.mount("/static", StaticFiles(directory="web/static"), name="static")
templates = Jinja2Templates(directory="web/templates")

matcher = AttendanceMatcher()

# Configurable Camera Source
cam_src = os.environ.get("CAMERA_SOURCE", "0")
if cam_src.isdigit():
    cam_src = int(cam_src)
camera = cv2.VideoCapture(cam_src)

# Basic Auth
security = HTTPBasic()

def get_current_username(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = secrets.compare_digest(credentials.username, "admin")
    correct_password = secrets.compare_digest(credentials.password, "password")
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

# WebSocket Manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass

manager = ConnectionManager()
loop = None

@app.on_event("startup")
async def startup_event():
    global loop
    loop = asyncio.get_running_loop()

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            regions = matcher.scanner.scan_image(frame)
            for (x, y, w, h) in regions:
                face_roi = frame[y:y+h, x:x+w]
                if face_roi.size > 0:
                    name, conf, just_logged = matcher.match_profile(face_roi)
                    
                    if just_logged and loop:
                        now_time = datetime.now().strftime("%H:%M:%S")
                        msg = {"student_name": name, "time": now_time, "confidence": round(conf, 2)}
                        asyncio.run_coroutine_threadsafe(manager.broadcast(msg), loop)
                    
                    color = (0, 255, 0) if name != "Unknown" and name != "Spoof Detected" else (0, 0, 255)
                    label = f"{name} ({conf:.2f})"
                    
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.get("/", response_class=HTMLResponse)
async def index(request: Request, username: str = Depends(get_current_username)):
    return templates.TemplateResponse(request=request, name="index.html")

@app.get("/video_feed")
def video_feed(username: str = Depends(get_current_username)):
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/api/attendance")
def get_attendance(username: str = Depends(get_current_username)):
    db = SessionLocal()
    records = get_today_attendance(db)
    results = [{"student_name": r.student.name, "time": r.timestamp.strftime("%H:%M:%S"), "confidence": round(r.confidence, 2)} for r in records]
    db.close()
    return {"attendance": results}

@app.post("/api/register")
async def api_register(name: str = Form(...), file: UploadFile = File(...), username: str = Depends(get_current_username)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img_bgr is None:
        return {"error": "Invalid image"}
        
    regions = matcher.scanner.scan_image(img_bgr)
    if len(regions) == 0:
        return {"error": "No face detected in the image"}
        
    x, y, w, h = regions[0]
    face_roi = img_bgr[y:y+h, x:x+w]
    face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
    
    signature = matcher.profiler.generate_signature(face_rgb)
    
    db = SessionLocal()
    register_student(db, name, signature)
    db.close()
    
    matcher.reload_students()
    
    return {"message": f"Student {name} registered successfully!"}

def start_server():
    uvicorn.run(app, host="127.0.0.1", port=8000)

if __name__ == "__main__":
    start_server()
