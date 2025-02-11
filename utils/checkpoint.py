import os
import sys
import torch
from collections import defaultdict
from diffusers.models import AutoencoderKL
from models.dit import DiT
from models.unet import HFUNet, DFUNet
from utils.file_utils import input_to_bool
from utils.setup_utils import get_device


# Load checkpoint
def load_checkpoint(
    model=None,
    optimizer=None,
    lr_scheduler=None,
    scaler=None,
    checkpoint_path=None,
    use_checkpoint=True
):
    if use_checkpoint and None in [model, checkpoint_path]:
        raise ValueError("Model and checkpoint_path are required to load checkpoint!")

    start_epoch, data_logger = 0, defaultdict(lambda: [])

    load_model = use_checkpoint

    if os.path.isfile(checkpoint_path):
        if not use_checkpoint:
            process = input("use_checkpoint is False but a checkpoint file found!\nContinue training using this checkpoint? (y/n): ").strip()
            load_model = input_to_bool(process)
    
    else:
        if use_checkpoint:
            process = input("use_checkpoint is True but no checkpoint file found!\nBegin new training session from scratch? (y/n): ").strip()
            load_model = not input_to_bool(process)
            if load_model:      # if not new training
                print("Exiting the training process...")
                sys.exit(0)


    if load_model:
        try:
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

            # Load model state
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Load additional states
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
                start_epoch = checkpoint['epoch']
                data_logger.update(checkpoint['data_logger'])
                
            except KeyError:
                print("-Checkpoint contains only model weights!")
                print("-Exiting the training process...")
                sys.exit(0)
            
            print(f"-Checkpoint loaded!")
            print(f"-Starting from Epoch-{start_epoch + 1}")
            
        except Exception as e:
            print(f"-Error loading checkpoint: {e}")
            print("-Exiting the training process...")
            sys.exit(0)
            
    else:
        print("-Starting from scratch")
    
    return start_epoch, data_logger, load_model


# Save checkpoint
def save_checkpoint(
	epoch=None, 
    data_logger=None, 
    model=None, 
    optimizer=None, 
    lr_scheduler=None, 
    scaler=None, 
    save_path=None, 
    weights_only=False
):

    if model is None or save_path is None:
        raise ValueError("Both 'model' and 'save_path' are required.")

    if not weights_only:
        if None in [epoch, data_logger, optimizer, lr_scheduler, scaler]:
            raise ValueError("All arguments are required when weights_only=False.")
        
        torch.save({
            'epoch': epoch + 1,
            'data_logger': dict(data_logger),
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict()
        }, save_path)
        
    else:
        torch.save({
            'model_state_dict': model.state_dict()
        }, save_path)
        
    #print(f"Checkpoint saved successfully at: {save_path}")

#####################################################################

# Function to initialize diffusion model and autoencoder
def load_model(checkpoint_path, in_channels, latent_size, vae_model):
    # Get device
    device = get_device()
    
    # Instantiate the diffusion model
    model_file = checkpoint_path.split("/")[-1]
    model_name = model_file[:model_file.rfind('.')]
    if "unet_hugg" in model_name:
        net = HFUNet(in_channels=in_channels, out_channels=in_channels, sample_size=latent_size)().to(device)
    elif "unet_cold" in model_name:
        net = DFUNet(channels=in_channels, out_dim=in_channels, img_size=latent_size, dim=64, dim_mults=(1, 2, 4, 8)).to(device)
    elif "dit_cold" in model_name:
        net = DiT(in_channels=in_channels, input_size=latent_size, patch_size=2, hidden_size=768, depth=14, num_heads=8).to(device)
    else:
        raise ValueError("Model name must contain 'unet_hugg', 'unet_cold' or 'dit_cold'!")
    
    try:
        # Load checkpoint and vae model
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        net.load_state_dict(checkpoint['model_state_dict'])
        vae = AutoencoderKL.from_pretrained(vae_model).to(device)
        
    except Exception as e:
        print(f"-Error loading checkpoint: {e}")
        print("-Exiting the sampling process...")
        sys.exit(0)
    
    print("-Checkpoint loaded!")
    return net, vae
