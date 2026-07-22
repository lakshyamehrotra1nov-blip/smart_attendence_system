import cv2
import numpy as np
from facenet_pytorch import MTCNN

class ImageScanner:
    def __init__(self):
        try:
            # Initialize MTCNN for CPU
            self.scanner = MTCNN(keep_all=True, device='cpu')
        except Exception as e:
            print(f"Couldn't load MTCNN: {e}")
            self.scanner = None

    def scan_image(self, image):
        """Returns list of (x, y, w, h, score) boxes"""
        if self.scanner is None:
            return []
            
        # MTCNN expects RGB image
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        boxes, probs = self.scanner.detect(img_rgb)
        
        region_boxes = []
        if boxes is not None:
            for i, box in enumerate(boxes):
                if probs[i] < 0.85: # High confidence threshold
                    continue
                    
                x1, y1, x2, y2 = box
                
                # Convert to x, y, w, h for compatibility
                x = int(max(0, x1))
                y = int(max(0, y1))
                w = int(max(0, x2 - x))
                h = int(max(0, y2 - y))
                
                if w > 0 and h > 0:
                    # Return [x, y, w, h, score]
                    # We pad it to 15 elements to maintain backward compatibility with any hardcoded slices
                    region = np.zeros(15, dtype=np.float32)
                    region[0:4] = [x, y, w, h]
                    region[-1] = probs[i]
                    region_boxes.append(region)
                
        return region_boxes

    def draw_regions(self, image, regions, label="Unknown", color=(0, 255, 0)):
        """
        Draws bounding boxes and labels on the image.
        """
        for region in regions:
            x, y, w, h = map(int, region[:4])
            cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
            cv2.putText(image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        return image
