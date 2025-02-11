import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from torch.amp import autocast
from torch.utils.data import DataLoader
from diffusers.models import AutoencoderKL
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from utils.config import images_archive
from utils.setup_utils import get_device
from utils.file_utils import input_to_bool, verify_dataset, download_dataset


# Function to encode image files to latent files with autoencoder
def encode_images_to_latents(images_path, latents_path, image_size, batch_size, vae_model):
    # Check if latents dataset already exists
    if os.path.exists(latents_path):
        for subdirectory, _, files in os.walk(latents_path):
            if (filecount := sum(1 for file in files if os.path.splitext(file)[-1] in ('.npy',))) > 0:
                print(f"Latents dataset already exists at: '{latents_path}' with {filecount} files.")
                proceed = input_to_bool(input("Proceed with re-encoding to overwrite the existing files? (y/n): ").strip())
                if not proceed:
                    print("Exiting the encoding process...")
                    sys.exit(0)
    
    os.makedirs(latents_path, exist_ok=True)
    
    # Get the available device
    device = get_device()
    
    # Download dataset if necessary
    download_dataset(images_archive, images_path, filetype="image")
    proceed = verify_dataset(images_path)
    
    if not proceed:
        print("Exiting the encoding process...")
        sys.exit(0)
    
    vae = AutoencoderKL.from_pretrained(vae_model).to(device)
    vae.eval()

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])

    dataset = ImageFolder(os.path.dirname(images_path), transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    img_index = 0
    with torch.no_grad():
        for x, y in tqdm(data_loader, desc=f"Encoding images", leave=True):
            x = x.to(device)
            with autocast('cuda'):
                latent_features = vae.encode(x).latent_dist.sample().mul_(0.18215)
                latent_features = latent_features.detach().cpu()

            for latent in latent_features.split(1, 0):
                np.save(os.path.join(latents_path, f'{img_index:05}.npy'), latent.squeeze(0).numpy())
                img_index += 1
    
    print(f"Latents dataset saved to: '{latents_path}'")