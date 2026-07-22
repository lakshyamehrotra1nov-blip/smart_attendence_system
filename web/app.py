import cv2
import threading
from fastapi import FastAPI, Request, Form, File, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn
import numpy as np

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.crud import get_today_attendance, register_student
from database.models import SessionLocal
from attendance.matcher import AttendanceMatcher

app = FastAPI()

app.mount("/static", StaticFiles(directory="web/static"), name="static")
templates = Jinja2Templates(directory="web/templates")

matcher = AttendanceMatcher()
camera = cv2.VideoCapture(0)

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Detect faces and draw boxes (We borrow detector logic here for display)
            regions = matcher.scanner.scan_image(frame)
            for (x, y, w, h) in regions:
                face_roi = frame[y:y+h, x:x+w]
                if face_roi.size > 0:
                    name, conf = matcher.match_profile(face_roi)
                    
                    color = (0, 255, 0) if name != "Unknown" and name != "Spoof Detected" else (0, 0, 255)
                    label = f"{name} ({conf:.2f})"
                    
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/api/attendance")
def get_attendance():
    db = SessionLocal()
    records = get_today_attendance(db)
    results = [{"student_name": r.student.name, "time": r.timestamp.strftime("%H:%M:%S"), "confidence": round(r.confidence, 2)} for r in records]
    db.close()
    return {"attendance": results}

@app.post("/api/register")
async def api_register(name: str = Form(...), file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img_bgr is None:
        return {"error": "Invalid image"}
        
    regions = matcher.scanner.scan_image(img_bgr)
    if len(regions) == 0:
        return {"error": "No face detected in the image"}
        
    # Take first face
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
