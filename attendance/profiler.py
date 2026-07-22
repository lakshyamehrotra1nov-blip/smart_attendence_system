import cv2
import numpy as np
import os

class ProfileGenerator:
    def __init__(self, asset_path='assets/face_recognition_sface_2021dec.onnx'):
        abs_asset_path = os.path.abspath(asset_path)
        # Generates a 128D signature profile
        try:
            self.generator = cv2.FaceRecognizerSF.create(
                model=abs_asset_path,
                config=""
            )
        except Exception as e:
            print(f"Error loading asset at {abs_asset_path}. Please download it.")
            self.generator = None

    def generate_signature(self, image_rgb, face_region):
        if self.generator is None:
            return [0.0] * 128
            
        img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        # alignCrop uses the 15-dim face array (bbox + 5 landmarks + confidence)
        # to perfectly warp and align the face for the SFace model
        aligned_face = self.generator.alignCrop(img_bgr, face_region)
        feature = self.generator.feature(aligned_face)
        
        signature = feature[0].tolist()
        return signature
        
    def compare_signatures(self, sig1, sig2):
        feature1 = np.array([sig1], dtype=np.float32)
        feature2 = np.array([sig2], dtype=np.float32)
        
        if self.generator is None:
            return 0.0
            
        # invert distance so higher is better
        distance = self.generator.match(feature1, feature2, cv2.FaceRecognizerSF_FR_COSINE)
        return 1.0 - distance

