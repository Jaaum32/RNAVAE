import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, latent_dim=128, input_channels=3, input_size=512):
        super().__init__()

        # Calcula tamanho após encoder conv (3 camadas stride=2 → div por 8)
        conv_out_size = input_size // 8  # 512/8 = 64
        conv_out_channels = 128

        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),  # 512 → 256
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),              # 256 → 128
            nn.ReLU(),
            nn.Conv2d(64, conv_out_channels, kernel_size=4, stride=2, padding=1), # 128 → 64
            nn.ReLU(),
            nn.Flatten()
        )

        self.fc_mu = nn.Linear(conv_out_channels * conv_out_size * conv_out_size, latent_dim)
        self.fc_logvar = nn.Linear(conv_out_channels * conv_out_size * conv_out_size, latent_dim)

        self.decoder_fc = nn.Linear(latent_dim, conv_out_channels * conv_out_size * conv_out_size)

        self.decoder = nn.Sequential(
            nn.Unflatten(1, (conv_out_channels, conv_out_size, conv_out_size)),
            nn.ConvTranspose2d(conv_out_channels, 64, kernel_size=4, stride=2, padding=1),  # 64 → 128
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),                 # 128 → 256
            nn.ReLU(),
            nn.ConvTranspose2d(32, input_channels, kernel_size=4, stride=2, padding=1),    # 256 → 512
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
