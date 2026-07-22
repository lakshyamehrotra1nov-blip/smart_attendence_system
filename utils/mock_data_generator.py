import os
import numpy as np
from PIL import Image

def generate_mock_dataset(base_dir, num_classes=3, images_per_class=20):
    """
    Generates a mock dataset with random noise images for testing the pipeline.
    """
    os.makedirs(base_dir, exist_ok=True)
    
    # We will generate images of size 224x224 (ResNet default)
    img_size = (224, 224, 3)
    
    for i in range(num_classes):
        class_name = f"Student_{i+1}"
        class_dir = os.path.join(base_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        for j in range(images_per_class):
            # Generate a random noise image
            random_image = np.random.randint(0, 256, img_size, dtype=np.uint8)
            img = Image.fromarray(random_image)
            img_path = os.path.join(class_dir, f"{class_name}_{j}.jpg")
            img.save(img_path)
            
    print(f"Mock dataset generated at {base_dir} with {num_classes} classes and {images_per_class} images per class.")

if __name__ == "__main__":
    generate_mock_dataset(os.path.join("dataset", "CASIA_WEB_FACE"))
