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
        """Called by API to refresh the list of students."""
        self.known_students = get_all_students(self.db)

    def match_profile(self, image_bgr):
        """
        Takes a cropped BGR image of a person.
        Returns (name, confidence_score).
        """
        # 1. Liveness check
        if not self.anti_spoof.is_real(image_bgr):
            return "Spoof Detected", 0.0

        # 2. Extract signature
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        live_signature = self.profiler.generate_signature(image_rgb)
        
        best_match_name = "Unknown"
        best_match_score = 0.0
        best_match_id = None
        
        # 3. Compare with known students
        for student in self.known_students:
            db_signature = student.get_signature()
            if db_signature is not None:
                score = self.profiler.compare_signatures(live_signature, db_signature)
                if score > best_match_score:
                    best_match_score = score
                    best_match_name = student.name
                    best_match_id = student.id
                    
        # 4. Threshold check
        if best_match_score >= self.confidence_threshold:
            # Try to log attendance
            success, _ = log_attendance(self.db, best_match_id, best_match_score)
            if success:
                print(f"Attendance marked for {best_match_name}")
            return best_match_name, best_match_score
            
        return "Unknown", best_match_score

    def close(self):
        self.db.close()
