import cv2
import os
import numpy as np
import random
import math

# Define a função para escolher um filtro de sépia aleatório
def choose_sepia_variant():
    variants = {
        'classic': np.array([
            [0.131, 0.534, 0.272],
            [0.168, 0.686, 0.349],
            [0.189, 0.769, 0.393]
        ]),
        'reddish': np.array([
            [0.100, 0.400, 0.300],
            [0.150, 0.600, 0.400],
            [0.200, 0.800, 0.500]
        ]),
        'yellowish': np.array([
            [0.080, 0.500, 0.400],
            [0.120, 0.700, 0.500],
            [0.180, 0.900, 0.600]
        ]),
        'washed_out': np.array([
            [0.050, 0.300, 0.150],
            [0.070, 0.500, 0.200],
            [0.090, 0.600, 0.250]
        ]),
        'grayish': np.array([
            [0.100, 0.400, 0.150],
            [0.130, 0.500, 0.200],
            [0.160, 0.600, 0.250]
        ])
    }
    variant_name = random.choice(list(variants.keys()))
    return variants[variant_name], variant_name

# Aplica o filtro sépia
def apply_sepia_bgr(image):
    sepia_filter, variant = choose_sepia_variant()
    img_float = image.astype(np.float32)
    sepia_img = cv2.transform(img_float, sepia_filter)
    sepia_img = np.clip(sepia_img, 0, 255).astype(np.uint8)
    print(f"Sépia variante aplicada: {variant}")
    return sepia_img

def add_blur(image, kernel_size=(5, 5)):
    return cv2.GaussianBlur(image, kernel_size, 0)

def add_scratch(image, num_scratches=5, scratch_length_range=(20, 100), scratch_thickness_range=(1, 3)):
    img_copy = image.copy()
    h, w, _ = img_copy.shape
    for _ in range(num_scratches):
        sx, sy = random.randint(0, w), random.randint(0, h)
        length = random.randint(*scratch_length_range)
        thickness = random.randint(*scratch_thickness_range)
        angle = random.uniform(0, 2 * math.pi)
        ex = int(sx + length * math.cos(angle))
        ey = int(sy + length * math.sin(angle))
        color = (random.randint(200, 255),) * 3 if random.random() > 0.5 else (random.randint(0, 50),) * 3
        cv2.line(img_copy, (sx, sy), (ex, ey), color, thickness)
    return img_copy

def add_stains_with_opacity(image, num_stains=3, stain_size_range=(10, 50)):
    img_copy = image.copy()
    h, w, _ = img_copy.shape
    overlay = np.zeros_like(img_copy, dtype=np.uint8)
    for _ in range(num_stains):
        cx, cy = random.randint(0, w), random.randint(0, h)
        radius = random.randint(*stain_size_range)
        color = (
            random.randint(50, 200),
            random.randint(50, 200),
            random.randint(50, 200)
        )
        cv2.circle(overlay, (cx, cy), radius, color, -1)
    alpha = random.uniform(0.3, 0.7)
    cv2.addWeighted(overlay, alpha, img_copy, 1 - alpha, 0, img_copy)
    return img_copy

def add_tear(image, num_tears=1, tear_length_range=(50, 150), tear_thickness_range=(1, 2)):
    img_copy = image.copy()
    h, w, _ = img_copy.shape
    for _ in range(num_tears):
        sx, sy = random.randint(0, w), random.randint(0, h)
        length = random.randint(*tear_length_range)
        thickness = random.randint(*tear_thickness_range)
        angle = random.uniform(0, 2 * math.pi)
        ex = int(sx + length * math.cos(angle))
        ey = int(sy + length * math.sin(angle))
        color = (255, 255, 255)
        cv2.line(img_copy, (sx, sy), (ex, ey), color, thickness)
    return img_copy

def add_grain(image, intensity=0.02):
    img_float = image.astype(np.float32) / 255.0
    noise = np.random.normal(0, intensity, img_float.shape).astype(np.float32)
    noisy_img = img_float + noise
    noisy_img = np.clip(noisy_img, 0, 1)
    return (noisy_img * 255).astype(np.uint8)

# Processamento completo de uma imagem
def process_image(image_path, output_dir):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Erro ao ler imagem: {image_path}")
        return

    img = apply_sepia_bgr(img)
    img = add_blur(img)

    if random.random() < 0.7:
        img = add_scratch(img)
    if random.random() < 0.5:
        img = add_stains_with_opacity(img)
    if random.random() < 0.3:
        img = add_tear(img)

    if random.random() < 0.5:
        intensity = random.uniform(0.01, 0.03)
        img = add_grain(img, intensity=intensity)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, os.path.basename(image_path))
    cv2.imwrite(output_path, img)
    print(f"Imagem salva: {output_path}")

# Processa o dataset inteiro com limite
def process_dataset(original_dataset_path, output_base_dir="old_photos_ood", max_images=4800):
    image_files = [f for f in os.listdir(original_dataset_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    if not image_files:
        print("Nenhuma imagem encontrada.")
        return

    os.makedirs(output_base_dir, exist_ok=True)

    image_files = image_files[:max_images]  # LIMITA A 4800 IMAGENS

    for file in image_files:
        full_image_path = os.path.join(original_dataset_path, file)
        process_image(full_image_path, output_base_dir)

# === RODAR ===
if __name__ == "__main__":
    dataset_path = r"photos_ood"
    process_dataset(dataset_path)
