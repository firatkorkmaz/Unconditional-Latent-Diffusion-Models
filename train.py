import argparse    
from utils.config import latents_path
from utils.setup_utils import set_seed
from utils.file_utils import input_to_bool
from operations.trainer import train_diffusion_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse hyperparameters for training.")
    parser.add_argument("--model_name", type=str, required=False, help="Name of the model")
    parser.add_argument("--num_epochs", type=int, default=1000, help="Epoch number (default: 1000)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--accum_steps", type=int, default=2, help="Accumulation steps (default: 2)")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument("--use_checkpoint", type=input_to_bool, nargs='?', const=True, default=False, help="Use checkpoint (default: False)")
    args = parser.parse_args()
    
    
    # Check for missing arguments and ask for the model name & number of images
    if args.model_name is None:
        args.model_name = input("Enter the model name to train, including;\n'unet_hugg' or 'unet_cold' or 'dit_cold': ").strip()
        if "unet_hugg" in args.model_name or "unet_cold" in args.model_name or "dit_cold" in args.model_name:
            print("\nHyperparameters...")
            print(f"Epoch number: {args.num_epochs}")
            print(f"Batch size: {args.batch_size}")
            print(f"Accumulation steps: {args.accum_steps}")
            print(f"Learning rate: {args.learning_rate}")
            print(f"Use checkpoint: {args.use_checkpoint}")
            assign_values = input_to_bool(input("\nEnter new values? (y/n): ").strip())
            if assign_values:
                args.num_epochs = int(input("Enter epoch number: ").strip())
                args.batch_size = int(input("Enter batch size: ").strip())
                args.accum_steps = int(input("Enter accumulation steps: ").strip())
                args.learning_rate = float(input("Enter learning rate: ").strip())
                args.use_checkpoint = input_to_bool(input("Use checkpoint (true/false): ").strip())
        else:
            raise ValueError("Model name must contain: 'unet_hugg' or 'unet_cold' or 'dit_cold'!")
    else:
        if "unet_hugg" not in args.model_name and "unet_cold" not in args.model_name and "dit_cold" not in args.model_name:
            raise ValueError("Model name must contain 'unet_hugg', 'unet_cold' or 'dit_cold'!")


    set_seed()
    vae_model = "stabilityai/sd-vae-ft-ema"
    save_path = "."
    in_channels = 4
    latent_size = 32
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    weight_decay = 1e-5
    eta_min = 1e-7
    T_0 = num_epochs//10
    T_mult = 1
    accum_steps = args.accum_steps
    num_steps = 1000 if "unet_hugg" in args.model_name else 100
    noise_sigma = 1.0
    model_name = args.model_name
    use_checkpoint = args.use_checkpoint
    
    
    # Call the training function
    train_diffusion_model(latents_path, vae_model, save_path, in_channels, latent_size, batch_size, num_epochs, learning_rate,
                          weight_decay, eta_min, T_0, T_mult, accum_steps, num_steps, noise_sigma, model_name, use_checkpoint)