import cv2
import mediapipe as mp
import numpy as np

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

class AntiSpoofer:
    def __init__(self, model_asset_path='assets/face_landmarker.task'):
        try:
            options = FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_asset_path),
                running_mode=VisionRunningMode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5
            )
            self.landmarker = FaceLandmarker.create_from_options(options)
        except Exception as e:
            print(f"Failed to load face landmarker: {e}")
            self.landmarker = None
            
        self.ear_history = []
        self.has_blinked = False
        self.frames_since_blink = 0

    def calculate_ear(self, landmarks, eye_indices):
        p1 = np.array([landmarks[eye_indices[0]].x, landmarks[eye_indices[0]].y])
        p4 = np.array([landmarks[eye_indices[3]].x, landmarks[eye_indices[3]].y])
        
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
        if self.landmarker is None:
            gray = cv2.cvtColor(face_image_bgr, cv2.COLOR_BGR2GRAY)
            variance = cv2.Laplacian(gray, cv2.CV_64F).var()
            return variance > 10.0
            
        image_rgb = cv2.cvtColor(face_image_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        
        detection_result = self.landmarker.detect(mp_image)
        
        if not detection_result.face_landmarks:
            return False
            
        landmarks = detection_result.face_landmarks[0]
        
        left_eye = [33, 160, 158, 133, 153, 144]
        right_eye = [362, 385, 387, 263, 373, 380]
        
        left_ear = self.calculate_ear(landmarks, left_eye)
        right_ear = self.calculate_ear(landmarks, right_eye)
        
        ear = (left_ear + right_ear) / 2.0
        
        self.ear_history.append(ear)
        if len(self.ear_history) > 15:
            self.ear_history.pop(0)
            
        if len(self.ear_history) == 15:
            min_ear = min(self.ear_history)
            max_ear = max(self.ear_history)
            # Make blink detection more lenient (was <0.20 and >0.25)
            if min_ear < 0.23 and max_ear > 0.23:
                self.has_blinked = True
                self.frames_since_blink = 0
                
        if self.has_blinked:
            self.frames_since_blink += 1
            if self.frames_since_blink > 100:
                self.has_blinked = False
                self.ear_history = []
            return True
            
        return False
