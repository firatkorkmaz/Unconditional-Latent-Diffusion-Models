import os
import shutil
import argparse
import numpy as np
from utils.zip_utils import zip_folder
from utils.file_utils import input_to_bool
from utils.config import images_path, latents_path
from operations.encoder import encode_images_to_latents


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse image size and batch size for encoding.")
    parser.add_argument("--image_size", type=int, default=256, help="Image size (default: 256)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (default: 32)")
    args = parser.parse_args()
    
    image_size=args.image_size
    batch_size=args.batch_size
    
    assert image_size % 8 == 0, "Image size must be divisible by 8 for autoencoder!"
    
    vae_model = "stabilityai/sd-vae-ft-ema"

    encode_images_to_latents(images_path, latents_path, image_size, batch_size, vae_model)
    zipping = input_to_bool(input("Save latents dataset to .zip file? (y/n): ").strip())
    if zipping:
        zipfile_path = f"{os.path.basename(latents_path)}.zip"
        zip_folder(latents_path, zipfile_path)
    
    cleanup = input_to_bool(input("Remove original image dataset folder? (y/n): ").strip())
    if cleanup:
        shutil.rmtree(os.path.dirname(images_path), ignore_errors=True)
        print("Image dataset folder removed: 'dataset'")
    
    print("Encoding completed.")