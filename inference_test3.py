import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import random
import os

# === Modelo VAE com BatchNorm e LeakyReLU ===
class VAE(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        act = lambda: nn.LeakyReLU(0.01)

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            act(),

            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            act(),

            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            act(),

            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            act(),

            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            act(),

            nn.Flatten()
        )

        self.fc_mu = nn.Linear(512 * 16 * 16, latent_dim)
        self.fc_logvar = nn.Linear(512 * 16 * 16, latent_dim)

        self.decoder_fc = nn.Linear(latent_dim, 512 * 16 * 16)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (512, 16, 16)),

            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            act(),

            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            act(),

            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            act(),

            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            act(),

            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_fc(z)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


# === Inferência ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vae = VAE(latent_dim=256).to(device)

# Carregar modelo treinado
model_path = "vae_final_trained.pth"
vae.load_state_dict(torch.load(model_path, map_location=device))
vae.eval()

# Escolher arquivo .pt aleatório
data_dir = r"C:\Users\USER\PycharmProjects\VAE\preprocessed_pairs_ood"
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
