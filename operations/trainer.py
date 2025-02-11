import os
import sys
import copy
import torch
import shutil
import torch.nn as nn
import statistics as st
from decimal import Decimal
import torch.optim as optim
from datetime import datetime
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.utils as vutils
from diffusers import DDPMScheduler
from collections import defaultdict
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.amp import autocast, GradScaler
from diffusers.models import AutoencoderKL
from utils.config import latents_archive
from utils.setup_utils import get_device
from models.dit import DiT
from models.unet import HFUNet, DFUNet
from utils.dataset import LatentDataset
from samplers.ddim import ddim_diffuse
from samplers.cold import cold_diffuse, cosine_alphas_bar
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.file_utils import initialize_directories, download_dataset, get_unique_log_file_path


# Training function for the diffusion model
def train_diffusion_model(dataset_path, vae_model, save_path, in_channels, latent_size, batch_size, num_epochs, learning_rate,
                          weight_decay, eta_min, T_0, T_mult, accum_steps, num_steps, noise_sigma, model_name, use_checkpoint):

    # Get the available device
    device = get_device()
    
    # Download dataset if necessary
    download_dataset(latents_archive, dataset_path, filetype="latent")
    
    # Directory initializations
    save_model_name = f"{model_name}"
    best_model_name = f"{model_name}_best"
    checkpoint_root, savesample_root = initialize_directories(save_path)

    checkpoint_path = os.path.join(checkpoint_root, f"{model_name}.pt")
    save_model_path = os.path.join(checkpoint_root, f"{save_model_name}.pt")
    best_model_path = os.path.join(checkpoint_root, f"{best_model_name}.pt")   

    # Dataset and DataLoader initialization
    dataset = LatentDataset(latent_dir=dataset_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    # Instantiate the diffusion model, vae model, optimizer, and scheduler
    if "unet_hugg" in model_name:
        diffusion_net = HFUNet(in_channels=in_channels, out_channels=in_channels, sample_size=latent_size)().to(device)
    elif "unet_cold" in model_name:
        diffusion_net = DFUNet(channels=in_channels, out_dim=in_channels, img_size=latent_size, dim=64, dim_mults=(1, 2, 4, 8)).to(device)
    elif "dit_cold" in model_name:
        diffusion_net = DiT(in_channels=in_channels, input_size=latent_size, patch_size=2, hidden_size=768, depth=14, num_heads=8).to(device)
    else:
        raise ValueError("Model name must contain 'unet_hugg', 'unet_cold' or 'dit_cold'!")
    
    vae = AutoencoderKL.from_pretrained(vae_model).to(device)
    optimizer = optim.AdamW(diffusion_net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=eta_min)
    #lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, eta_min=eta_min, T_0=T_0, T_mult=T_mult)

    # Gradient scaler for mixed-precision
    scaler = GradScaler()

    # Get start_epoch and data_logger
    start_epoch, data_logger, load_model = load_checkpoint(
                                           model=diffusion_net,
                                           optimizer=optimizer,
                                           lr_scheduler=lr_scheduler,
                                           scaler=scaler,
                                           checkpoint_path=checkpoint_path,
                                           use_checkpoint=use_checkpoint
                                           )
    diffusion_net_best = copy.deepcopy(diffusion_net)
    
    alphas = None
    noise_scheduler = None
    if "unet_hugg" in model_name:
        # Initialize noise scheduler
        noise_scheduler = DDPMScheduler(num_train_timesteps=num_steps)
    
    elif "unet_cold" in model_name or "dit_cold" in model_name:
        # Alpha schedule for diffusion process
        alphas = torch.flip(cosine_alphas_bar(num_steps), dims=(0,)).to(device)

    else:
        raise ValueError("Model name must contain 'unet_hugg', 'unet_cold' or 'dit_cold'!")

    vae.eval()      # Keep autoencoder in evaluation mode
    best_loss = float('inf')

    # Early stopping parameters (inactive)
    #patience = 20    # Number of epochs to wait for improvement before stopping
    #no_improve_count = 0               # Counter for epochs without improvement
    
    # Preparing log file parameters
    log_base_path="logs"
    filename = "train_log"
    resume_training = load_model
    unique_log_file_path = None
    
    # Training loop
    for epoch in range(start_epoch, num_epochs):
        diffusion_net.train()
        mean_loss = 0
        total_loss_epoch = 0
        num_batches = len(dataloader)

        # Initialize progress bar
        progress_bar = tqdm(enumerate(dataloader), total=num_batches, desc=f"Epoch {epoch + 1:04}/{num_epochs:04}")

        for i, latents in progress_bar:
            latents = latents.to(device)
            bs = latents.size(0)

            if bs == 0:
                continue    # Handle case with empty batch
            
            if "unet_hugg" in model_name:
                # Generate random sample and apply noise to latents
                random_sample = torch.randn_like(latents).to(device)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.size(0),), device=device).long()
                noise_latents = noise_scheduler.add_noise(latents, random_sample, timesteps)

                # Autocast for mixed-precision training
                with autocast('cuda'):
                    # Forward pass and loss computation
                    model_output = diffusion_net(noise_latents, timesteps).sample
                    loss = F.l1_loss(model_output, random_sample)
                    total_loss_epoch += loss.item()
                    mean_loss += loss.item()
                    
            elif "unet_cold" in model_name or "dit_cold" in model_name:
                # Sample random timesteps and apply cold diffusion noise
                index = torch.randint(num_steps, (bs,), device=device)
                alpha = alphas[index].reshape(bs, 1, 1, 1)
                random_sample = torch.randn_like(latents).to(device)
                noise_latents = alpha.sqrt() * latents + (1 - alpha).sqrt() * random_sample

                # Autocast for mixed-precision training
                with autocast('cuda'):
                    # Forward pass and loss computation
                    model_output = diffusion_net(noise_latents, index)
                    loss = F.l1_loss(model_output, latents)
                    total_loss_epoch += loss.item()
                    mean_loss += loss.item()
            
            else:
                raise ValueError("Model name must contain 'unet_hugg', 'unet_cold' or 'dit_cold'!")
                
            # Backpropagation with gradient scaling
            scaler.scale(loss / accum_steps).backward()

            # Perform optimizer step after accum_steps or at the last batch
            if (i + 1) % accum_steps == 0 or (i + 1) == num_batches:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                # Log mean loss and reset accumulator
                avg_loss = mean_loss / accum_steps
                data_logger["loss"].append(avg_loss)
                mean_loss = 0

                # Convert lr value to 1.00e-4 type string
                lr = optimizer.param_groups[0]['lr']
                e_lr = f"{Decimal(lr):.2e}"    # Decimal function to formal lr

                # Update progress bar with current average loss and learning rate
                progress_bar.set_postfix_str(f"lr={e_lr}, loss={avg_loss:.5f}")

        # Save checkpoint and generate images periodically
        diffusion_net.eval()

        with torch.no_grad():
            if "unet_hugg" in model_name:
                #num_steps = 1000
                # Generate latents using the cold diffusion process
                diff_img = ddim_diffuse(diffusion_model=diffusion_net,
                                        batch_size=batch_size,
                                        in_channels=in_channels,
                                        input_size=latent_size,
                                        total_steps=num_steps,
                                        noise_sigma=noise_sigma,
                                        device=device)
                                         
            elif "unet_cold" in model_name or "dit_cold" in model_name:
                #num_steps = 100
                # Generate latents using the cold diffusion process
                diff_img = cold_diffuse(diffusion_model=diffusion_net,
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
            vutils.save_image(fake_sample.cpu().float(),
                os.path.join(savesample_root, f"{save_model_name}_{epoch + 1:04}.png"),
                normalize=True, value_range=(-1, 1))

        diffusion_net.train()
        
        # Save checkpoint
        save_checkpoint(
            epoch=epoch, 
            data_logger=data_logger, 
            model=diffusion_net, 
            optimizer=optimizer, 
            lr_scheduler=lr_scheduler, 
            scaler=scaler, 
            save_path=save_model_path
        )
        
        # Calculate average loss for current epoch
        avg_loss_epoch = total_loss_epoch / num_batches

        # Save the best model based on epoch loss
        if avg_loss_epoch < best_loss:
            best_loss = avg_loss_epoch

            # Backup best model for saving lightweight model file later
            diffusion_net_best = copy.deepcopy(diffusion_net)
            
            # Save checkpoint
            save_checkpoint(
                epoch=epoch, 
                data_logger=data_logger, 
                model=diffusion_net_best, 
                optimizer=optimizer, 
                lr_scheduler=lr_scheduler, 
                scaler=scaler, 
                save_path=best_model_path
            )
            
            print(f"Epoch-{epoch + 1} finished with average loss: {avg_loss_epoch:.5f} || New best model saved with average loss: {avg_loss_epoch:.5f}")
            #no_improve_count = 0       # Early stopping (inactive)

        else:
            print(f"Epoch-{epoch + 1} finished with average loss: {avg_loss_epoch:.5f}")
            #no_improve_count += 1      # Early stopping (inactive)
        
        
        # Create unique log file
        if unique_log_file_path is None:
            unique_log_file_path = get_unique_log_file_path(log_base_path, resume_training, filename)
            
        # Write epoch data to log file
        with open(unique_log_file_path, "a") as log_file:
            if epoch == start_epoch and resume_training:
                log_file.write("--- Resuming Training ---\n")
                log_file.flush()
        
            log_message = (f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                f"Epoch: {epoch+1:04}/{num_epochs:04} | "
                f"Loss: {avg_loss:.5f} | LR: {e_lr} | "
                f"Model: {model_name}")
            log_file.write(f"{log_message}\n")
            log_file.flush()
        
        
        '''# Early stopping check (inactive)
        if no_improve_count >= patience:
            print(f"\nEarly stopped after {epoch + 1} epochs due to no improvement.")
            break'''

        '''
        # Optional: Decrease base_lr and eta_min of the CAWR scheduler
        if (epoch + 1) % T_0 == 0 and epoch < num_epochs-1:
            lr_scheduler.base_lrs[0] *= 0.8
            lr_scheduler.eta_min *= 0.7
        '''

        # Update learning rate scheduler
        lr_scheduler.step()
    
    #-ENDED Loop: "for epoch in range(start_epoch, num_epochs)"
    
    
    # Save weights-only checkpoint files with small size at the end of training
    save_checkpoint(model=diffusion_net, save_path=save_model_path, weights_only=True)
    save_checkpoint(model=diffusion_net_best, save_path=best_model_path, weights_only=True)
    
    print("Training completed.")
