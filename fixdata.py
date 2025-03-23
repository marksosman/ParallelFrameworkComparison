import os
import shutil
import tarfile
from PIL import Image
from torchvision import datasets

# Define dataset paths
dataset_root = "./data/food-101"
image_dir = os.path.join(dataset_root, "images")
meta_dir = os.path.join(dataset_root, "meta")
tar_file = "./data/food-101.tar.gz"

def remove_corrupt_images(directory):
    """Deletes corrupted images that cannot be opened by PIL."""
    corrupt_count = 0
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".jpg"):
                file_path = os.path.join(root, file)
                try:
                    with Image.open(file_path) as img:
                        img.verify()  # Check if image is valid
                except Exception:
                    print(f"Deleting corrupted image: {file_path}")
                    os.remove(file_path)
                    corrupt_count += 1
    print(f"Removed {corrupt_count} corrupted images.")

def delete_metadata():
    """Deletes metadata to force dataset re-download."""
    if os.path.exists(meta_dir):
        shutil.rmtree(meta_dir)
        print("Deleted metadata folder to force re-download.")

def extract_dataset():
    """Extracts food-101.tar.gz if dataset folder is missing."""
    if os.path.exists(tar_file) and not os.path.exists(image_dir):
        print("Extracting dataset...")
        with tarfile.open(tar_file, "r:gz") as tar:
            tar.extractall(path="./data")
        print("Extraction complete.")

def redownload_food101():
    """Redownloads the Food101 dataset after cleaning."""
    print("Redownloading missing Food101 dataset files...")
    datasets.Food101(root=dataset_root, split="train", download=True)
    print("Redownload complete.")

if __name__ == "__main__":
    # Step 1: Extract dataset if needed
    extract_dataset()

    # Step 2: Remove corrupted images
    remove_corrupt_images(image_dir)

    # Step 3: Delete metadata folder to force re-download of missing images
    delete_metadata()

    # Step 4: Redownload the dataset (only missing files)
    redownload_food101()