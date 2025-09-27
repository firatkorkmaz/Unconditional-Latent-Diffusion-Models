# Latent Diffusion Models for Unconditional Image Generation

![Model](https://img.shields.io/badge/Model-Diffusion-brown?logo=pytorch&logoColor=orange)
![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white)
[![License](https://img.shields.io/badge/License-MIT-yellow?logo=opensourceinitiative&logoColor=white)](LICENSE)
![Status](https://img.shields.io/badge/Status-Complete-success?logo=checkmarx&logoColor=white)
![Last Updated](https://img.shields.io/badge/Updated-February%202025-orange?logo=github&logoColor=white)

**Tags:** `Latent Diffusion`, `Unconditional Generation`, `Image Synthesis`, `U-Net`, `DiT`, `Diffusion Transformer`, `DDPM`, `DDIM`, `Cold Diffusion`, `Cosine Scheduling`, `VAE Encoder`, `FFHQ Dataset`, `PyTorch`, `Hugging Face`, `Computer Vision`, `Generative AI`, `Deep Learning`, `Python`

Training three different latent diffusion models unconditionally with two distinct noise scheduling techniques and sampling images from the pretrained models.

## Table of Contents
- [General Information](#general-information)
- [Project Structure](#project-structure)
- [Project Parameters](#project-parameters)
- [Project Results](#project-results)
- [Setup & Run](#setup--run)

## General Information
This project provides a complete pipeline for latent diffusion models, covering image dataset encoding into latents, training three different models with two distinct noise schedules, and sampling images from the pretrained models.

### Encoder
Encodes **256x256 images** of the [Flickr-Faces-HQ Dataset (FFHQ)](https://www.kaggle.com/datasets/denislukovnikov/ffhq256-images-only) into **32x32 .npy latents** using **sd-vae-ft-ema** Autoencoder model with the specified batch size.

### Trainer
Provides three training options for [Latent Diffusion Models](https://arxiv.org/abs/2112.10752):

1. Training UNet2DModel (Hugging Face) with the [DDPM](https://arxiv.org/abs/2006.11239) Noise Scheduler.

2. Training Diffusion U-Net model with the [Cosine Alphas Noise Scheduler](https://arxiv.org/abs/2102.09672).

3. Training [Diffusion Transformer](https://arxiv.org/abs/2212.09748) model with the Cosine Alphas Noise Scheduler.

### Sampler
Generates images from the three pretrained models using:

1. [DDIM](https://arxiv.org/abs/2010.02502) Sampler.

2. [Cold Diffusion](https://arxiv.org/abs/2208.09392) Sampler.

**NOTE**: While the Cold Diffusion implementation here currently uses Gaussian noising, this pipeline can easily be extended to incorporate other image distortion techniques such as blurring, sharpening, or any other custom transformations.

### Dataset

All pretrained models are trained on **FFHQ latents dataset**.

The **~1GB FFHQ latents dataset** is automatically downloaded from the **Release** section when the **Trainer** program is run (see **.env** file). If you would like to run the **Encoder** program on your own, you can download the original **~7GB** [FFHQ](https://www.kaggle.com/datasets/denislukovnikov/ffhq256-images-only) **image dataset**.

The **FFHQ image dataset** is provided as an **archive.zip** file containing **70,000 images** inside an **ffhq256** folder. To use the **Encoder** program, simply place this **archive.zip** file in the project's **root folder**.

The program will extract the **ffhq256** folder into a new **dataset** folder located in the project's root directory. If the **ffhq256** folder already exists in the **dataset** folder, the program will detect it and use the existing images automatically.

The **Encoder** program will generate **70,000 latents** and save them in the **data/ffhq32** folder, which will be used for **training** later.

## Project Structure

```
project_root/				# == Unconditional-Latent-Diffusion-Models ==
├── *dataset/				# Stores the original images dataset for encoding.
│  └── ffhq256/				# Contains the image files with .png extension.
├── *data/				# Stores the encoded latents dataset for training.
│  └── ffhq32/				# Contains the latent files with .npy extension.
├── *outputs/				# Stores the outputs of training and sampling.
│  ├── Checkpoints/			# Saved model checkpoints during training.
│  ├── TrainingSamples/			# Generated image samples during training.
│  └── GeneratedSamples/		# Generated image samples after training.
├── *logs/				# Stores training log files with .txt extension.
├── models/				# Contains diffusion model architecture files.
│  ├── __init__.py			# Makes the models/ folder a Python package.
│  ├── dit.py				# Diffusion Transformer model implementation.
│  └── unet.py				# Diffusion Unet and HuggingFace Unet models.
├── samplers/				# Contains sampling methods for diffusion.
│  ├── __init__.py			# Makes the samplers/ folder a Python package.
│  ├── ddim.py				# Implementation of the DDIM sampling method.
│  └── cold.py				# Implementation of the Cold Diffusion sampler.
├── utils/				# Contains utility scripts for common functions.
│  ├── __init__.py			# Makes the utils/ folder a Python package.
│  ├── config.py			# Loads environment variables for central usage.
│  ├── zipper.py			# Handles zipping the encoded latent dataset files.
│  ├── unzipper.py			# Handles unzipping the downloaded archive files.
│  ├── downloader.py			# Handles dataset/pretrained model downloading.
│  ├── setup_utils.py			# Device configuration and random seed management.
│  ├── file_utils.py			# Directory and file management utilities.
│  ├── checkpoint.py			# Functions for saving/loading checkpoints.
│  └── dataset.py			# Dataset loading and preprocessing utilities.
├── operations/				# Contains both training and sampling scripts.
│  ├── __init__.py			# Makes the operations/ folder a Python package.
│  ├── encoder.py			# Encodes the image dataset into latents for training.
│  ├── trainer.py			# Main training function for latent diffusion models.
│  └── sampler.py			# Main sampling function for latent diffusion models.
├── README.md				# Project description and instructions.
├── requirements.txt			# Dependencies required for the project.
├── setup.sh				# Script for setup before running the program.
├── .env				# Environment variables configuration file.
├── encode.py				# Main program to run the dataset encoder function.
├── train.py				# Main program to run the model trainer function.
└── sample.py				# Main program to run the image sampler function.

[*] Will exist after program run.
```

## Project Parameters

### HuggingFace U-Net Parameters

* **in_channels**: 4
* **out_channels**: 4
* **sample_size**: 32
* **layers_per_block**: 2
* **block_out_channels**: (64, 128, 256, 512)
* **down_block_types**: ("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D")
* **up_block_types**: ("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D")
* **attention_head_dim**: 64

### Diffusion U-Net Parameters

* **channels**: 4
* **out_dim**: 4
* **img_size**: 32
* **dim**: 64
* **dim_mults**: (1, 2, 4, 8)

### Diffusion Transformer Parameters

* **in_channels**: 4
* **input_size**: 32
* **patch_size**: 2
* **hidden_size**: 768
* **num_heads**: 8
* **depth**: 14

### Hyperparameters of the Pretrained Models
* **Number of Epochs**: 200
* **Optimizer**: AdamW -> learning_rate=2e-4 || weight_decay=1e-5
* **LR Scheduler**: CosineAnnealingLR -> T_max=200 || eta_min=1e-6
* **Noise Scheduler**: DDPM: num_steps=1000 || Cold: num_steps=100
* **Precision**: GradScaler() of torch.amp.autocast
* **Early Stopping**: patience=20 (Inactive)
* **Loss Type**: L1 Loss - Mean Absolute Error (MAE)
* **Batch Size**: 32 Samples
* **Accumulation Steps**: 2 (Effective Batch Size: 32×2=64)

## Project Results

The results below are from a 200-epoch training run. They effectively demonstrate the model differences under identical conditions, as all samples were generated using the same seed for consistency.

### Loss Curves from Each Model

<img title="U-Net HuggingFace Diffusion Loss Curve" src="https://github.com/firatkorkmaz/Unconditional-Latent-Diffusion-Models/blob/main/images/unet_hugg_diffusion_loss.jpg">

<img title="U-Net Cold Diffusion Loss Curve" src="https://github.com/firatkorkmaz/Unconditional-Latent-Diffusion-Models/blob/main/images/unet_cold_diffusion_loss.jpg">

<img title="DiT Cold Diffusion Loss Curve" src="https://github.com/firatkorkmaz/Unconditional-Latent-Diffusion-Models/blob/main/images/dit_cold_diffusion_loss.jpg">

### Image Samples from Each Model

<img title="U-Net HuggingFace Diffusion Image Samples" src="https://github.com/firatkorkmaz/Unconditional-Latent-Diffusion-Models/blob/main/images/unet_hugg_diffusion_samples.jpg">

<img title="U-Net Cold Diffusion Image Samples" src="https://github.com/firatkorkmaz/Unconditional-Latent-Diffusion-Models/blob/main/images/unet_cold_diffusion_samples.jpg">

<img title="DiT Cold Diffusion Image Samples" src="https://github.com/firatkorkmaz/Unconditional-Latent-Diffusion-Models/blob/main/images/dit_cold_diffusion_samples.jpg">


## Setup & Run

**NOTE**: This script requires Python 3.x (between version 3.9 and 3.12) to work properly. Please ensure that you have a compatible version installed. If you're unsure of the version, you can check it by running:

```
# Check Python version
python3 --version	(Linux/macOS)
python --version	(Windows)
```

In order to setup the environment, run this command in Bash (Linux/MacOS/Windows-Git):

```
source setup.sh
```

After activating the environment, run each program with such commands:

### Encoder

Running this program requires **archive.zip** file of the [FFHQ](https://www.kaggle.com/datasets/denislukovnikov/ffhq256-images-only) **image dataset**, downloaded to the **project's root path**.

**Command Examples**:

```
python encode.py
python encode.py --batch_size 16
```

If the **batch_size** parameter is not provided as an argument, it is assumed that the user has approved its default value:

```
batch_size: 32
```

There is also an **image_size** parameter, which defaults to **256** for the current dataset of 256x256 face images.

### Trainer

Running this program automatically downloads **ffhq32.zip** file of the **FFHQ latents dataset**, if it is not found in the **project's root path**. 

**Command Examples**:

```
python train.py
python train.py --model_name unet_hugg_diffusion
python train.py --model_name unet_cold_diffusion --batch_size 64
python train.py --model_name dit_cold_diffusion --batch_size 16 --accum_steps 4
python train.py --model_name dit_cold_diffusion --num_epochs 2000 --learning_rate 5e-5 --use_checkpoint true
```

If the **model_name** parameter is not provided as an argument at runtime, the script will enter an interactive mode and the user is first asked to enter a model name. Then, the default values of all other parameters are listed for approval or modification. If only **model_name** is provided at runtime without specifying any other parameters, it is assumed that the user has approved all default values:

```
num_epochs: 1000
batch_size: 32
accum_steps: 2
learning_rate: 1e-4
use_checkpoint: False
```

If the **use_checkpoint** parameter is set to **True**, the program looks for an existing **.pt** checkpoint file in the **outputs/CheckpointFiles** directory. This file must be from a previous **incomplete training** session. If the **use_checkpoint** parameter is set to **False**, but a **.pt** checkpoint file already exists in the **outputs/CheckpointFiles** directory, the program will prompt the user whether to resume training or start a new session from scratch.

Once the training is fully completed, final checkpoint files will contain only model weights. These files cannot be used to resume training, as they lack the necessary information to continue from the last session.

The provided model names for **Training** must contain one of the following phrases for the program to recognize them and determine which training algorithm to use:

```
...unet_hugg...
...unet_cold...
...dit_cold...
```

### Sampler

Running this program automatically downloads checkpoint files of the specified model names at runtime, if they are not found in the **outputs/CheckpointFiles** directory.

The pretrained models of this project that are available for download via the **Sampler** program have the following names:

```
unet_hugg_diffusion
unet_cold_diffusion
dit_cold_diffusion
```

If you trained your own models with different filenames that include one of the required phrases (**unet_hugg**, **unet_cold**, **dit_cold**) and uploaded your checkpoint files to a server, you must update the **.env** file to set the **MODELS_ROOT** variable accordingly. Then, provide one of those names as the **model_name** parameter when running the **Sampler** program.

**Command Examples**:

```
python sample.py
python sample.py --model_name unet_cold_diffusion
python sample.py --model_name dit_cold_diffusion --batch_size 16
```

If the **model_name** parameter is not provided as an argument at runtime, the script will enter an interactive mode and the user is first asked to enter a model name. This pretrained model file must either be placed in the **outputs/CheckpointFiles** folder or uploaded to a server, with its link specified in the **.env** file's **MODELS_ROOT** variable. Then, the default value of **batch_size** parameter is listed for approval or modification. If only **model_name** is provided at runtime without specifying any other parameters, it is assumed that the user has approved the default **batch_size** value:

```
batch_size: 32
```

When the **model_name** is provided by the user, let's assume that it is **dit_cold_diffusion**, the program first checks the **outputs/CheckpointFiles** directory for the pretrained model's checkpoint file named **dit_cold_diffusion.pt**. If this file is not found, then it checks the **MODELS_ROOT** variable in the **.env** file and constructs a download URL by appending the provided model name to this web address to attempt a download.

**NOTE**: If auto-download fails, you can manually download the pretrained models and latents dataset using the links below.

[Models](https://www.mediafire.com/file/3yj954jg3fgzmi3/Models.zip): Download the **Models.zip** file and extract its **CheckpointFiles** folder into the **outputs** directory in the **project's root path**.

[Dataset](https://www.mediafire.com/file/j5tbnsh5a1p2rfv/ffhq32.zip): Download the **ffhq32.zip** file and place it in the **project's root path**.
