import cv2
import mediapipe as mp
import numpy as np

class AntiSpoofer:
    def __init__(self):
        # We use mediapipe face mesh to track eye movements
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        # Store recent EAR (Eye Aspect Ratio) values to detect blinks
        self.ear_history = []
        self.has_blinked = False
        self.frames_since_blink = 0

    def calculate_ear(self, landmarks, eye_indices):
        """Calculate Eye Aspect Ratio (EAR)."""
        # Horizontal distance
        p1 = np.array([landmarks[eye_indices[0]].x, landmarks[eye_indices[0]].y])
        p4 = np.array([landmarks[eye_indices[3]].x, landmarks[eye_indices[3]].y])
        
        # Vertical distances
        p2 = np.array([landmarks[eye_indices[1]].x, landmarks[eye_indices[1]].y])
        p6 = np.array([landmarks[eye_indices[5]].x, landmarks[eye_indices[5]].y])
        p3 = np.array([landmarks[eye_indices[2]].x, landmarks[eye_indices[2]].y])
        p5 = np.array([landmarks[eye_indices[4]].x, landmarks[eye_indices[4]].y])
        
        hor = np.linalg.norm(p1 - p4)
        ver1 = np.linalg.norm(p2 - p6)
        ver2 = np.linalg.norm(p3 - p5)
        
        if hor == 0:
            return 0.0
            
        ear = (ver1 + ver2) / (2.0 * hor)
        return ear

    def is_real(self, face_image_bgr):
        # Mediapipe works best with RGB
        image_rgb = cv2.cvtColor(face_image_bgr, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)
        
        if not results.multi_face_landmarks:
            # If we can't find a face mesh, fall back to variance of laplacian (old method)
            gray = cv2.cvtColor(face_image_bgr, cv2.COLOR_BGR2GRAY)
            variance = cv2.Laplacian(gray, cv2.CV_64F).var()
            return variance > 10.0
            
        landmarks = results.multi_face_landmarks[0].landmark
        
        # Approximate indices for the left and right eyes (inner/outer corners and top/bottom)
        left_eye = [33, 160, 158, 133, 153, 144]
        right_eye = [362, 385, 387, 263, 373, 380]
        
        left_ear = self.calculate_ear(landmarks, left_eye)
        right_ear = self.calculate_ear(landmarks, right_eye)
        
        ear = (left_ear + right_ear) / 2.0
        
        self.ear_history.append(ear)
        if len(self.ear_history) > 15:
            self.ear_history.pop(0)
            
        # Check for a dip in EAR (a blink)
        # Typical open EAR > 0.2, blink < 0.15
        if len(self.ear_history) == 15:
            min_ear = min(self.ear_history)
            max_ear = max(self.ear_history)
            if min_ear < 0.20 and max_ear > 0.25:
                self.has_blinked = True
                self.frames_since_blink = 0
                
        if self.has_blinked:
            self.frames_since_blink += 1
            # Reset blink status after 100 frames (~3-4 seconds) to require a fresh blink
            if self.frames_since_blink > 100:
                self.has_blinked = False
                self.ear_history = []
            return True
            
        return False
