import os
from dotenv import load_dotenv


# Load .env file
load_dotenv()

# Root URL of model checkpoint files
models_root = os.getenv("MODELS_ROOT")

# Root url of latents archive
datasets_root = os.getenv("DATASETS_ROOT")

# Filename of images archive
images_archive = os.getenv("IMAGES_ARCHIVE", "archive.zip")

# Path to extract image files
images_path = os.getenv("IMAGES_PATH", "dataset/ffhq256")

# Filename of latents archive
latents_archive = os.getenv("LATENTS_ARCHIVE", "ffhq32.zip")

# Path to extract latent files
latents_path = os.getenv("LATENTS_PATH", "data/ffhq32")
