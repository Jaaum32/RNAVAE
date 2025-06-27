import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from vae_model import VAE

# ============ Dataset que carrega os arquivos .pt já gerados ============
class PairedPTDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.files = [f for f in os.listdir(data_dir) if f.endswith('.pt')]
        self.files.sort()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(os.path.join(self.data_dir, self.files[idx]))
        return data['input'], data['target']


# ============ Função de perda ============
def loss_function(recon_x, x, mu, logvar):
    batch_size = x.size(0)
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum') / batch_size
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
    return recon_loss + kl_div

# ============ Treinamento ============
def train():
    data_dir = r"preprocessed_pairs"
    latent_dim = 256
    num_epochs = 30
    batch_size = 16
    learning_rate = 1e-4

    full_dataset = PairedPTDataset(data_dir)
    dataloader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vae = VAE(latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(vae.parameters(), lr=learning_rate)

    print(f"==== Treinamento Iniciado ====")
    print(f"Total de amostras: {len(full_dataset)}")
    print(f"Total de batches por época: {len(dataloader)}")
    print(f"Latent Dim: {latent_dim}, Batch Size: {batch_size}, Learning Rate: {learning_rate}")

    for epoch in range(num_epochs):
        vae.train()
        total_loss = 0
        print(f"\n==== Época {epoch + 1}/{num_epochs} ====")

        for batch_idx, (degraded, original) in enumerate(dataloader):
            degraded = degraded.to(device)
            original = original.to(device)

            optimizer.zero_grad()
            recon, mu, logvar = vae(degraded)
            loss = loss_function(recon, original, mu, logvar)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(dataloader):
                print(f"Batch [{batch_idx + 1}/{len(dataloader)}] - Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"===> Fim da Época {epoch + 1}: Loss Médio: {avg_loss:.4f}")

    torch.save(vae.state_dict(), "vae_trained.pth")
    print("\n==== Treinamento Finalizado ====")
    print("Modelo salvo com sucesso!")

if __name__ == "__main__":
    train()
