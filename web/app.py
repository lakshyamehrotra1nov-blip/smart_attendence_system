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

latest_frame = None

def generate_frames():
    global latest_frame
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            latest_frame = frame.copy()
            regions = matcher.scanner.scan_image(frame)
            for region in regions:
                x, y, w, h = int(region[0]), int(region[1]), int(region[2]), int(region[3])
                
                name, conf, just_logged = matcher.match_profile(frame, region)
                
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

import io
import csv
from fastapi.responses import HTMLResponse, StreamingResponse, Response

@app.get("/api/attendance")
def get_attendance(username: str = Depends(get_current_username)):
    db = SessionLocal()
    records = get_today_attendance(db)
    results = [{"student_name": r.student.name, "time": r.timestamp.strftime("%H:%M:%S"), "confidence": round(r.confidence, 2)} for r in records if r.student is not None]
    db.close()
    return {"attendance": results}

@app.get("/api/export/csv")
def export_csv(username: str = Depends(get_current_username)):
    db = SessionLocal()
    records = get_today_attendance(db)
    
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Name", "Time", "Confidence"])
    
    for r in records:
        if r.student is not None:
            writer.writerow([r.student.name, r.timestamp.strftime("%H:%M:%S"), round(r.confidence, 2)])
        
    db.close()
    
    headers = {
        "Content-Disposition": f"attachment; filename=attendance_{datetime.now().strftime('%Y-%m-%d')}.csv"
    }
    return Response(content=output.getvalue(), media_type="text/csv", headers=headers)

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
        
    face_region = regions[0]
    face_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    signature = matcher.profiler.generate_signature(face_rgb, face_region)
    
    db = SessionLocal()
    register_student(db, name, signature)
    db.close()
    
    matcher.reload_students()
    
    return {"message": f"Student {name} registered successfully!"}

@app.post("/api/register_live")
async def api_register_live(name: str = Form(...), username: str = Depends(get_current_username)):
    global latest_frame
    
    if latest_frame is None:
        return {"error": "Camera is not active or hasn't captured a frame yet."}
        
    regions = matcher.scanner.scan_image(latest_frame)
    if len(regions) == 0:
        return {"error": "No face detected in the live camera right now. Please look at the camera."}
        
    face_region = regions[0]
    face_rgb = cv2.cvtColor(latest_frame, cv2.COLOR_BGR2RGB)
    
    signature = matcher.profiler.generate_signature(face_rgb, face_region)
    
    db = SessionLocal()
    register_student(db, name, signature)
    db.close()
    
    matcher.reload_students()
    
    return {"message": f"Student {name} registered successfully from live camera!"}

def start_server():
    uvicorn.run(app, host="127.0.0.1", port=8000)

if __name__ == "__main__":
    start_server()
