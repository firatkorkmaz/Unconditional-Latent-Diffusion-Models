import os
import math
import torch
import numpy as np
import torchvision.utils as vutils
from samplers.ddim import ddim_diffuse
from samplers.cold import cold_diffuse
from utils.checkpoint import load_model
from utils.setup_utils import get_device
from utils.file_utils import download_model, get_unique_sample_path, save_single_images


# Main function to generate images using diffusion models
def generate_images(model_name, batch_size=32):
    # Get the available device
    device = get_device()
    
    # Download pretrained model if necessary
    download_model(model_name)
    
    # Define parameters for image generation
    in_channels = 4
    latent_size = 32
    noise_sigma = 1.0
    base_save_path = "outputs/GeneratedSamples"
    model_file = f"{model_name}.pt"
    checkpoint_path = f"outputs/CheckpointFiles/{model_file}"

    # Define models
    vae_model = "stabilityai/sd-vae-ft-ema"

    # Initialize models
    net, vae = load_model(checkpoint_path, in_channels, latent_size, vae_model)
    net.eval(); vae.eval()

    # Create a unique directory and file for saving results
    unique_sample_path = get_unique_sample_path(base_save_path, model_name)

    with torch.no_grad():
        if "unet_hugg" in model_name:
            num_steps = 1000
            diff_img = ddim_diffuse(diffusion_model=net,
                                    batch_size=batch_size,
                                    in_channels=in_channels,
                                    input_size=latent_size,
                                    total_steps=num_steps,
                                    noise_sigma=noise_sigma,
                                    device=device)
        elif "unet_cold" in model_name or "dit_cold" in model_name:
            num_steps = 100
            diff_img = cold_diffuse(diffusion_model=net,
                                    batch_size=batch_size,
                                    in_channels=in_channels,
                                    input_size=latent_size,
                                    total_steps=num_steps,
                                    noise_sigma=noise_sigma,
                                    device=device)
        else:
            raise ValueError("Model name must contain 'unet_hugg', 'unet_cold' or 'dit_cold'!")

        # Decode latents to images and save as files
        fake_sample = vae.decode(diff_img / 0.18215).sample
        save_single_images(fake_sample, unique_sample_path)
        vutils.save_image(fake_sample.cpu().float(),
                          f"{unique_sample_path}.png",
                          normalize=True, value_range=(-1, 1))
