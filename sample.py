import argparse
from utils.file_utils import input_to_bool
from operations.sampler import generate_images


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse model name and batch size.")
    parser.add_argument("--model_name", type=str, required=False, help="Name of the model")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (default: 32)")
    args = parser.parse_args()

    # Check for missing arguments and ask for the model name & number of images
    if args.model_name is None:
        args.model_name = input("Enter the model name to sample, including;\n'unet_hugg' or 'unet_cold' or 'dit_cold': ").strip()
        if "unet_hugg" in args.model_name or "unet_cold" in args.model_name or "dit_cold" in args.model_name:
            print("\nHyperparameters...")
            print(f"Batch size: {args.batch_size}")
            assign_value = input_to_bool(input("\nEnter new value? (y/n): ").strip())
            if assign_value:
                args.batch_size = int(input("Enter batch size: ").strip())
        else:
            raise ValueError("Model name must contain: 'unet_hugg' or 'unet_cold' or 'dit_cold'!")
    else:
        if "unet_hugg" not in args.model_name and "unet_cold" not in args.model_name and "dit_cold" not in args.model_name:
            raise ValueError("Model name must contain 'unet_hugg', 'unet_cold' or 'dit_cold'!")
        
    # Generate images from pretrained diffusion models
    generate_images(args.model_name, args.batch_size)
