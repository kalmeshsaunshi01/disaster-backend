import os
import numpy as np
import cv2
from tqdm import tqdm

# Define dataset paths
DATASET_DIR = "dataset"
PREPROCESSED_DIR = "dataset/preprocessed"
DISASTERS = ["deforestationdata", "flooddata", "landslidedata"]

# Image size for resizing
IMG_SIZE = (256, 256)

def preprocess_disaster(disaster):
    """ Preprocess images and masks for a given disaster type """
    img_folder = os.path.join(DATASET_DIR, disaster, "images")
    mask_folder = os.path.join(DATASET_DIR, disaster, "mask")
    save_path = os.path.join(PREPROCESSED_DIR, disaster)

    if not os.path.exists(img_folder) or not os.path.exists(mask_folder):
        print(f"❌ Error: {img_folder} OR {mask_folder} not found!")
        return

    os.makedirs(save_path, exist_ok=True)

    images, masks = [], []
    image_files = sorted(os.listdir(img_folder))
    
    for img_name in tqdm(image_files, desc=f"Processing {disaster}"):
        img_path = os.path.join(img_folder, img_name)
        mask_path = os.path.join(mask_folder, img_name)  # Assuming same names

        if not os.path.exists(mask_path):
            print(f"⚠️ Warning: Mask missing for {img_name}, skipping...")
            continue

        # Read and resize images
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if img is None or mask is None:
            print(f"⚠️ Error reading {img_name}, skipping...")
            continue

        img = cv2.resize(img, IMG_SIZE)
        mask = cv2.resize(mask, IMG_SIZE)

        images.append(img)
        masks.append(mask)

    # Convert lists to NumPy arrays
    images = np.array(images, dtype=np.uint8)
    masks = np.array(masks, dtype=np.uint8)

    # Save as .npy files
    np.save(os.path.join(save_path, "images.npy"), images)
    np.save(os.path.join(save_path, "masks.npy"), masks)

    print(f"✅ {disaster} processed: {images.shape} images, {masks.shape} masks saved!")

# Run preprocessing for each disaster type
if __name__ == "__main__":
    os.makedirs(PREPROCESSED_DIR, exist_ok=True)
    for disaster in DISASTERS:
        preprocess_disaster(disaster)

    print("🎉 All datasets processed successfully!")
