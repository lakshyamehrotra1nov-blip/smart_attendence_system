import cv2
import os

class ImageScanner:
    def __init__(self, asset_path='assets/face_detection_yunet_2023mar.onnx'):
        # We need absolute path for the asset
        abs_asset_path = os.path.abspath(asset_path)
        
        # Initialize internal scanner. Score threshold 0.82 ensures high confidence.
        try:
            self.scanner = cv2.FaceDetectorYN.create(
                model=abs_asset_path,
                config="",
                input_size=(320, 320),
                score_threshold=0.82,
                nms_threshold=0.3,
                top_k=5000
            )
        except Exception as e:
            print(f"Error loading asset at {abs_asset_path}. Please download it.")
            self.scanner = None

    def scan_image(self, image):
        """
        Takes a BGR image.
        Returns a list of regions where each region is (x, y, w, h).
        """
        if self.scanner is None:
            return []
            
        height, width, _ = image.shape
        
        # Scale down large images to work reliably and avoid crashes
        max_size = 640
        scale = 1.0
        if max(width, height) > max_size:
            scale = max_size / max(width, height)
            input_img = cv2.resize(image, (int(width * scale), int(height * scale)))
        else:
            input_img = image
            
        h, w, _ = input_img.shape
        self.scanner.setInputSize((w, h))
        
        # Process image
        _, regions = self.scanner.detect(input_img)
        
        region_boxes = []
        if regions is not None:
            for region in regions:
                box = list(map(int, region[:4]))
                
                # Scale box back up to original image size
                if scale != 1.0:
                    box = [int(v / scale) for v in box]
                    
                # Ensure box is within image bounds
                x, y, w, h = box
                x = max(0, x)
                y = max(0, y)
                w = min(width - x, w)
                h = min(height - y, h)
                
                if w > 0 and h > 0:
                    region_boxes.append((x, y, w, h))
                
        return region_boxes

    def draw_regions(self, image, regions, label="Unknown", color=(0, 255, 0)):
        """
        Draws bounding boxes and labels on the image.
        """
        for (x, y, w, h) in regions:
            cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
            cv2.putText(image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        return image
