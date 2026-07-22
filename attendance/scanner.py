import cv2
import os

class ImageScanner:
    def __init__(self, asset_path='assets/face_detection_yunet_2023mar.onnx'):
        abs_asset_path = os.path.abspath(asset_path)
        
        try:
            # high confidence threshold so it doesn't pick up random noise
            self.scanner = cv2.FaceDetectorYN.create(
                model=abs_asset_path,
                config="",
                input_size=(320, 320),
                score_threshold=0.82,
                nms_threshold=0.3,
                top_k=5000
            )
        except Exception as e:
            print(f"Couldn't load scanner asset from {abs_asset_path}. Make sure the file exists.")
            self.scanner = None

    def scan_image(self, image):
        # returns list of (x, y, w, h) boxes

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
                scaled_region = np.copy(region)
                # Scale everything except the last element (confidence)
                if scale != 1.0:
                    scaled_region[:14] = scaled_region[:14] / scale
                    
                # Ensure box is within image bounds
                x, y, w, h = scaled_region[:4]
                x = max(0, x)
                y = max(0, y)
                w = min(width - x, w)
                h = min(height - y, h)
                
                scaled_region[0] = x
                scaled_region[1] = y
                scaled_region[2] = w
                scaled_region[3] = h
                
                if w > 0 and h > 0:
                    region_boxes.append(scaled_region)
                
        return region_boxes

    def draw_regions(self, image, regions, label="Unknown", color=(0, 255, 0)):
        """
        Draws bounding boxes and labels on the image.
        """
        for (x, y, w, h) in regions:
            cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
            cv2.putText(image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        return image
