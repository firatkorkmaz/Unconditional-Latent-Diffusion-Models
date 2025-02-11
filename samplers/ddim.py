import copy
import torch
from tqdm import tqdm
from diffusers import DDIMScheduler


def ddim_diffuse(diffusion_model, batch_size, in_channels, input_size, total_steps, noise_sigma, device):
    # Initialize the DDIM scheduler
    diffusion_model.eval()
    ddim_scheduler = DDIMScheduler(num_train_timesteps=total_steps)
    ddim_scheduler.set_timesteps(num_inference_steps=total_steps // 20)

    # Convert timesteps to a tensor and send to the device once
    timesteps = ddim_scheduler.timesteps.clone().detach().to(device)

    # Initialize noisy latents
    latents = noise_sigma * torch.randn(batch_size, in_channels, input_size, input_size, device=device)

    # Begin DDIM sampling loop
    with torch.no_grad():
        for t in tqdm(timesteps, disable=True):
            # Predict the noise residual
            noise_pred = diffusion_model(latents, t).sample

            # Update the latents using the DDIM scheduler
            latents = ddim_scheduler.step(noise_pred, t, latents).prev_sample

    return latents