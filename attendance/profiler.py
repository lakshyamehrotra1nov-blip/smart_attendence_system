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

    def generate_signature(self, image_rgb):
        """
        Takes an aligned/cropped RGB image (numpy array) and returns a 128-D signature list.
        """
        if self.generator is None:
            return [0.0] * 128
            
        # Natively takes BGR
        img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        # Resize the crop to 112x112
        img_resized = cv2.resize(img_bgr, (112, 112))
        
        # We pass a simple bounding box
        dummy_box = np.array([0, 0, 112, 112, 0,0,0,0,0,0,0,0,0,0,1.0], dtype=np.float32)
        
        feature = self.generator.feature(img_resized)
        
        signature = feature[0].tolist()
        return signature
        
    def compare_signatures(self, sig1, sig2):
        """Returns the similarity score between two signatures."""
        feature1 = np.array([sig1], dtype=np.float32)
        feature2 = np.array([sig2], dtype=np.float32)
        
        if self.generator is None:
            return 0.0
            
        # match() returns distance. We invert it so that higher is better.
        distance = self.generator.match(feature1, feature2, cv2.FaceRecognizerSF_FR_COSINE)
        return 1.0 - distance

