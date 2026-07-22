import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1

class ProfileGenerator:
    def __init__(self):
        try:
            self.device = torch.device('cpu')
            self.generator = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        except Exception as e:
            print(f"Error loading PyTorch FaceNet: {e}")
            self.generator = None

    def generate_signature(self, image_rgb, face_region):
        if self.generator is None:
            return [0.0] * 512
            
        x, y, w, h = map(int, face_region[:4])
        
        # Crop the face with a slight margin to match MTCNN output style
        margin = 15
        height, width, _ = image_rgb.shape
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(width, x + w + margin)
        y2 = min(height, y + h + margin)
        
        face_img = image_rgb[y1:y2, x1:x2]
        face_img = cv2.resize(face_img, (160, 160))
        
        # Normalize to [-1, 1] as expected by InceptionResnetV1
        face_tensor = torch.tensor(face_img).permute(2, 0, 1).float()
        face_tensor = (face_tensor - 127.5) / 128.0
        face_tensor = face_tensor.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            embedding = self.generator(face_tensor)
            
        return embedding[0].cpu().numpy().tolist()
        
    def compare_signatures(self, sig1, sig2):
        if self.generator is None:
            return 0.0
            
        # Calculate L2 distance between the two 512D embeddings
        s1 = np.array(sig1, dtype=np.float32)
        s2 = np.array(sig2, dtype=np.float32)
        dist = np.linalg.norm(s1 - s2)
        
        # Convert distance to similarity score (higher is better)
        # FaceNet L2 distance is < 0.9 for matches, > 1.1 for mismatches
        return 2.0 - dist

