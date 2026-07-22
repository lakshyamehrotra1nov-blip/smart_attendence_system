import cv2
import numpy as np
from datetime import datetime

from attendance.scanner import ImageScanner
from attendance.profiler import ProfileGenerator
from attendance.anti_spoof import AntiSpoofer
from database.crud import get_all_students, log_attendance
from database.models import SessionLocal, init_db

class AttendanceMatcher:
    def __init__(self, confidence_threshold=0.65):
        # Ensure DB is initialized
        init_db()
        
        self.confidence_threshold = confidence_threshold
        self.scanner = ImageScanner()
        self.profiler = ProfileGenerator()
        self.anti_spoof = AntiSpoofer()
        
        self.db = SessionLocal()
        self.known_students = []
        self.reload_students()
        print(f"Loaded {len(self.known_students)} known students from DB.")

    def reload_students(self):
        # API calls this to refresh
        self.known_students = get_all_students(self.db)

    def match_profile(self, frame_bgr, face_region):
        # Extract face_roi for anti-spoofing
        x, y, w, h = int(face_region[0]), int(face_region[1]), int(face_region[2]), int(face_region[3])
        face_roi = frame_bgr[y:y+h, x:x+w]
        
        if face_roi.size == 0:
            return "Unknown", 0.0, False

        if not self.anti_spoof.is_real(face_roi):
            return "Spoof Detected", 0.0, False

        image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        live_sig = self.profiler.generate_signature(image_rgb, face_region)
        
        best_name = "Unknown"
        best_score = 0.0
        best_id = None
        
        # Go through everyone in DB and find best match
        for student in self.known_students:
            db_sig = student.get_signature()
            if db_sig:
                score = self.profiler.compare_signatures(live_sig, db_sig)
                if score > best_score:
                    best_score = score
                    best_name = student.name
                    best_id = student.id
                    
        # Check against threshold and log if they passed
        if best_score >= self.confidence_threshold:
            success, _ = log_attendance(self.db, best_id, best_score)
            if success:
                print(f"Logged attendance for {best_name}")
            return best_name, best_score, success
            
        return "Unknown", best_score, False

    def close(self):
        self.db.close()
