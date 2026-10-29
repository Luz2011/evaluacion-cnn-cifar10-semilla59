import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, random_split

# ============================================================
# SEMILLA INDIVIDUAL (últimos 2 dígitos del código: 59)
# ============================================================
SEED = 59
torch.manual_seed(SEED)
np.random.seed(SEED)

# ============================================================
# 1. PIPELINE DE TRANSFORMACIONES (Data Augmentation)
# ============================================================
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),       # volteo horizontal
    transforms.RandomCrop(32, padding=4),         # recorte aleatorio con padding
    transforms.ToTensor(),                        # a tensor [0,1]
    transforms.Normalize((0.4914, 0.4822, 0.4465),  # media CIFAR-10
                         (0.2470, 0.2435, 0.2616))  # std CIFAR-10
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2470, 0.2435, 0.2616))
])

CLASSES = ['avión', 'automóvil', 'pájaro', 'gato', 'ciervo',
           'perro', 'rana', 'caballo', 'barco', 'camión']

# ============================================================
# 2. CARGA DEL DATASET CIFAR-10
# ============================================================
def get_datasets():
    train_full = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=train_transform)
    test_set = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=test_transform)
    return train_full, test_set

# ============================================================
# 3. PARTICIÓN ESTRATIFICADA (80% train / 20% val)
# ============================================================
def stratified_split(dataset, val_ratio=0.2):
    targets = np.array(dataset.targets)
    train_idx, val_idx = [], []
    for c in np.unique(targets):
        idx_c = np.where(targets == c)[0]
        np.random.shuffle(idx_c)
        n_val = int(len(idx_c) * val_ratio)
        val_idx.extend(idx_c[:n_val])
        train_idx.extend(idx_c[n_val:])
    train_sub = torch.utils.data.Subset(dataset, train_idx)
    val_sub = torch.utils.data.Subset(dataset, val_idx)
    return train_sub, val_sub

# ============================================================
# 4. VISUALIZACIÓN DE MUESTRAS
# ============================================================
def visualize_samples(dataset, n=16):
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        img, label = dataset[i]
        img = img.permute(1, 2, 0).numpy()
        # desnormalizar para visualizar
        mean = np.array([0.4914, 0.4822, 0.4465])
        std = np.array([0.2470, 0.2435, 0.2616])
        img = img * std + mean
        img = np.clip(img, 0, 1)
        ax.imshow(img)
        ax.set_title(CLASSES[label])
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('outputs/samples_cifar10.png', dpi=100)
    plt.show()

# ============================================================
# 5. EJECUCIÓN DIRECTA
# ============================================================
if __name__ == "__main__":
    import os
    os.makedirs('outputs', exist_ok=True)
    print(f"Semilla utilizada: {SEED}")
    train_full, test_set = get_datasets()
    train_set, val_set = stratified_split(train_full)

    print(f"Train: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_set)}")

    visualize_samples(test_set, n=16)

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

    imgs, labels = next(iter(train_loader))
    print(f"Shape de un batch: {imgs.shape}")  # [64, 3, 32, 32]