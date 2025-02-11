import os
import sys
import torch
import random
import numpy as np
from math import ceil
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from utils.zip_utils import unzip
from utils.downloader import download
from utils.config import datasets_root, models_root


# Convert input (bool, int, str) to boolean
def input_to_bool(value: bool | int | str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        if value == 1:
            return True
        elif value == 0:
            return False
    if isinstance(value, str):
        value = value.strip().lower()
        if value in {'true', 't', '1', 'yes', 'y'}:
            return True
        elif value in {'false', 'f', '0', 'no', 'n'}:
            return False
    raise ValueError(f'{value} is not a valid boolean value')


# Detect corrupted image files
def detect_corruption(file_path):
    try:
        with Image.open(file_path) as img:
            img.verify()      # Verify image file integrity
        return False          # Image is not corrupted
    except (IOError, SyntaxError) as e:
        return True           # Image is corrupted


# Verify dataset
def verify_dataset(directory, extensions=('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
    print(f"Starting directory scan at: '{directory}'")
    image_files = []
    corrupted_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(extensions):
                image_files.append(os.path.join(root, file))  # Add image file to list
    
    print(f"Found {len(image_files)} image files, running verification...")
    
    # Process files and track progress with tqdm
    for file_path in tqdm(image_files, desc="Processing Images"):
        if detect_corruption(file_path):
            corrupted_files.append(file_path)

    print(f"{len(corrupted_files)} image files corrupted! {len(image_files)-len(corrupted_files)} image files valid.")
    if len(corrupted_files) > 0:
        print("To proceed with the encoding process, you must approve the removal of corrupted files.")
        proceed = input_to_bool(input("Would you like to proceed? (y/n): ").strip())
        if proceed:
            print("Removing corrupted image files:")
            for file_path in corrupted_files:
                os.remove(file_path)
                print(file_path)
            return True
        else:
            return False
    else:
        return True


# Initialize necessary directories
def initialize_directories(save_path):
    checkpoint_root = os.path.join(save_path, "outputs/CheckpointFiles")
    savesample_root = os.path.join(save_path, "outputs/TrainingSamples")
    
    os.makedirs(checkpoint_root, exist_ok=True)
    os.makedirs(savesample_root, exist_ok=True)
    
    return checkpoint_root, savesample_root


# Define function to download and unzip dataset
def download_dataset(dataset_file, dataset_path, filetype):
    # Define extensions
    if filetype.lower() == "image":
        extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff',)
    elif filetype.lower() == "latent":
        extensions = ('.npy',)
    else:
        raise ValueError("Filetype must be either image or latent!")
    
    # Check if dataset is already extracted
    if os.path.exists(dataset_path):
        for subdirectory, _, files in os.walk(dataset_path):
            if (filecount := sum(1 for file in files if os.path.splitext(file)[-1] in extensions)) > 0:
                print(f"Dataset already exists at: '{dataset_path}' with {filecount} files.")
                return
    
    # Download the zipfile if not exists
    if not os.path.exists(dataset_file):
        dataset_url = os.path.join(datasets_root, dataset_file)
        download(dataset_url, dataset_file)

    # Extract dataset to the 'data' folder
    unzip(dataset_file, dataset_path, filetype)


# Define pretrained model downloader function
def download_model(model_name):
    # Proceed with downloading
    model_folder = "outputs/CheckpointFiles"
    os.makedirs(model_folder, exist_ok=True)
    model_file = f"{model_folder}/{model_name}.pt"

    if not os.path.exists(model_file):
        url = os.path.join(models_root, f"{model_name}.pt")
        download(url, model_file)


def get_unique_log_file_path(log_base_path, resume_training=False, filename="train_log"):
    # Ensure the base path exists
    os.makedirs(log_base_path, exist_ok=True)

    # List all existing log files with '.txt' extension that match the filename
    existing_log_files = [
        f for f in os.listdir(log_base_path)
        if os.path.isfile(os.path.join(log_base_path, f)) and f.startswith(filename) and f.endswith('.txt')
    ]

    # Extract the counter values from existing log files
    existing_counters = [
        int(f.split('_')[-1].split('.')[0])  # Extract numbers from filenames like 'train_log_0001.txt'
        for f in existing_log_files if f[len(filename):].startswith('_') and f.split('_')[-1].split('.')[0].isdigit()
    ]

    # Determine counter value
    if resume_training:
        if (next_counter := max(existing_counters, default=0)) == 0:
            next_counter += 1
    else:
        next_counter = max(existing_counters, default=0) + 1

    # Generate the unique log file path
    unique_log_file_path = os.path.join(log_base_path, f"{filename}_{next_counter:04}.txt")

    return unique_log_file_path


# Function to create a unique directory name
def get_unique_sample_path(base_path, model_name):
    # Ensure the base path exists
    os.makedirs(base_path, exist_ok=True)

    # List all existing directories that match the model_name pattern
    existing_dirs = [
        d for d in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, d)) and d.startswith(model_name)
    ]

    # Extract the counter values from existing directories
    existing_counters = [
        int(d.split("_")[-1]) for d in existing_dirs if d[len(model_name):].startswith("_") and d.split("_")[-1].isdigit()
    ]

    # Determine the next counter value
    next_counter = max(existing_counters, default=0) + 1

    # Generate unique paths for the next directory
    unique_sample_path = os.path.join(base_path, f"{model_name}_{next_counter:04}")
    
    return unique_sample_path


# Function to save generated images
def save_single_images(fake_sample, save_path):
    os.makedirs(save_path, exist_ok=True)
    for idx, img_tensor in enumerate(fake_sample):
        img = (img_tensor / 2 + 0.5).clamp(0, 1).cpu().numpy()
        img = np.transpose(img, (1, 2, 0))
        img = (img * 255).astype(np.uint8)
        img_pil = Image.fromarray(img)
        img_pil.save(f"{save_path}/image_{idx+1:04}.png")
    print(f"-Saved {len(fake_sample)} images.")


# Function to plot generated images in grid
def plot_image_grid(fake_sample, n_images, cols=8):
    # Create a grid of images
    grid = vutils.make_grid(fake_sample, nrow=cols, normalize=True, value_range=(-1, 1))

    # Convert the grid tensor to a numpy array
    grid_np = grid.permute(1, 2, 0).cpu().numpy()

    # Ensure dtype compatibility with matplotlib
    if grid_np.dtype not in ('float32', 'float64'):
        grid_np = grid_np.astype('float32')

    # Calculate grid dimensions
    rows = ceil(n_images / cols)
    fig_width = cols * 3
    fig_height = rows * 3

    # Create a figure for the grid
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=100)
    ax = fig.add_axes((0, 0, 1, 1))  # Full figure axis
    ax.axis("off")  # Disable axes
    ax.imshow(grid_np)
    plt.show()


