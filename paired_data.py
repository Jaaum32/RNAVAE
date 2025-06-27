import os
import cv2
import torch
import torchvision.transforms as transforms

# Caminhos das pastas
original_dir = r"C:\Users\USER\.cache\kagglehub\datasets\arnaud58\flickrfaceshq-dataset-ffhq\versions\1"
degraded_dir = r"C:\Users\USER\PycharmProjects\VAE\old_photos_dataset"

# Pasta onde vamos salvar os dados prontos
output_dir = r"C:\Users\USER\PycharmProjects\VAE\preprocessed_pairs"
os.makedirs(output_dir, exist_ok=True)

# Transformação (garante que tudo vira tensor, normalizado)
transform = transforms.Compose([
    transforms.ToTensor()  # converte para [0, 1] e tensor
])

# Coleta os arquivos (assumindo que nomes são iguais nas duas pastas)
files = [f for f in os.listdir(original_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
files.sort()  # garantir ordem estável

for idx, filename in enumerate(files):
    orig_path = os.path.join(original_dir, filename)
    deg_path = os.path.join(degraded_dir, filename)

    if not os.path.exists(deg_path):
        print(f"Arquivo não encontrado no degradado: {filename}")
        continue

    # Carregar as imagens
    orig_img = cv2.imread(orig_path)
    deg_img = cv2.imread(deg_path)

    # Verificar leitura
    if orig_img is None or deg_img is None:
        print(f"Erro ao ler: {filename}")
        continue

    # Converter BGR -> RGB (OpenCV padrão)
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    deg_img = cv2.cvtColor(deg_img, cv2.COLOR_BGR2RGB)

    # Transformar em tensor (shape: C x H x W)
    orig_tensor = transform(orig_img)
    deg_tensor = transform(deg_img)

    # Salvar o par em arquivo .pt
    output_file = os.path.join(output_dir, f"{idx:05d}.pt")
    pair = {'input': deg_tensor, 'target': orig_tensor}
    torch.save(pair, output_file)

    # Mensagem por arquivo
    print(f"[{idx+1}/{len(files)}] Salvo: {output_file}")

print("Pré-processamento finalizado!")
