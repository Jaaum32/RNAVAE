import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import random
import os
from vae_model import VAE

# === Inferência ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vae = VAE(latent_dim=256).to(device)

# Carregar modelo treinado
model_path = "vae_trained.pth"
vae.load_state_dict(torch.load(model_path, map_location=device))
vae.eval()

# Escolher arquivo .pt aleatório
data_dir = r"preprocessed_pairs"
files = [f for f in os.listdir(data_dir) if f.endswith('.pt')]
sample_file = os.path.join(data_dir, random.choice(files))

# Carregar input + target
data = torch.load(sample_file)
degraded = data['input'].unsqueeze(0).to(device)
original = data['target'].unsqueeze(0).to(device)

# Reconstrução
with torch.no_grad():
    recon, _, _ = vae(degraded)

# Visualização
def show_images(input_img, recon_img, target_img):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(input_img.cpu().squeeze().permute(1, 2, 0).numpy())
    axs[0].set_title("Imagem degradada")
    axs[1].imshow(recon_img.cpu().squeeze().permute(1, 2, 0).numpy())
    axs[1].set_title("Reconstruída pelo VAE")
    axs[2].imshow(target_img.cpu().squeeze().permute(1, 2, 0).numpy())
    axs[2].set_title("Imagem original")
    for ax in axs:
        ax.axis('off')
    plt.tight_layout()
    plt.show()

show_images(degraded, recon, original)
