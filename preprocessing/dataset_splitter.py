import os
import shutil
import random

def split_dataset(src_dir, dest_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Splits the dataset into train, val, and test sets.
    
    Args:
        src_dir (str): Path to the original dataset containing class folders.
        dest_dir (str): Path to the destination directory for splits.
        train_ratio (float): Proportion for training.
        val_ratio (float): Proportion for validation.
        test_ratio (float): Proportion for testing.
        seed (int): Random seed for reproducibility.
    """
    random.seed(seed)
    
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Ratios must sum to 1"
    
    splits = ['train', 'val', 'test']
    for split in splits:
        os.makedirs(os.path.join(dest_dir, split), exist_ok=True)
        
    if not os.path.exists(src_dir):
        print(f"Source directory {src_dir} does not exist.")
        return
        
    classes = [d for d in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, d))]
    print(f"Found {len(classes)} classes.")
    
    for cls in classes:
        cls_dir = os.path.join(src_dir, cls)
        images = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Filter out hidden or non-image files if any
        if not images:
            continue
            
        random.shuffle(images)
        
        n_total = len(images)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        train_imgs = images[:n_train]
        val_imgs = images[n_train:n_train+n_val]
        test_imgs = images[n_train+n_val:]
        
        # Helper to copy images
        def copy_imgs(img_list, split_name):
            split_cls_dir = os.path.join(dest_dir, split_name, cls)
            os.makedirs(split_cls_dir, exist_ok=True)
            for img in img_list:
                src_path = os.path.join(cls_dir, img)
                dest_path = os.path.join(split_cls_dir, img)
                shutil.copy2(src_path, dest_path)
                
        copy_imgs(train_imgs, 'train')
        copy_imgs(val_imgs, 'val')
        copy_imgs(test_imgs, 'test')
        
    print(f"Dataset split completed successfully into {dest_dir}.")

if __name__ == "__main__":
    src_directory = os.path.join('dataset', 'CASIA_WEB_FACE')
    dest_directory = os.path.join('dataset', 'splits')
    split_dataset(src_directory, dest_directory)
