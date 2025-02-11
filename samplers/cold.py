import copy
import torch
from tqdm.auto import trange


def noise_from_x0(curr_img, img_pred, alpha):
    return (curr_img - alpha.sqrt() * img_pred)/((1 - alpha).sqrt() + 1e-4)


def cosine_alphas_bar(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, steps, steps)
    alphas_bar = torch.cos(((x / steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_bar = alphas_bar / alphas_bar[0]
    return alphas_bar[:timesteps]


def cold_diffuse(diffusion_model, batch_size, in_channels, input_size, total_steps, noise_sigma, device):
    diffusion_model.eval()
    random_sample = noise_sigma * torch.randn(batch_size, in_channels, input_size, input_size, device=device)
    alphas = torch.flip(cosine_alphas_bar(total_steps), dims=(0,)).to(device)

    with torch.no_grad():
        for i in trange(total_steps - 1, disable=True):
            index = torch.full((batch_size,), i, device=device, dtype=torch.long)

            img_output = diffusion_model(random_sample, index)

            noise = noise_from_x0(random_sample, img_output, alphas[i])
            x0 = img_output

            rep1 = alphas[i].sqrt() * x0 + (1 - alphas[i]).sqrt() * noise
            rep2 = alphas[i + 1].sqrt() * x0 + (1 - alphas[i + 1]).sqrt() * noise

            random_sample += rep2 - rep1

        index = torch.full((batch_size,), (total_steps - 1), device=device, dtype=torch.long)
        img_output = diffusion_model(random_sample, index)

    return img_output