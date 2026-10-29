import os, sys
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.dataset import get_datasets, stratified_split, CLASSES, SEED
from src.model import CNN_CIFAR10

# Reproducibilidad
torch.manual_seed(SEED)
np.random.seed(SEED)

EPOCHS = 15
LR = 1e-3
BATCH = 64

def main():
    os.makedirs('outputs', exist_ok=True)

    # Datos
    train_full, test_set = get_datasets()
    train_set, val_set = stratified_split(train_full)
    train_loader = DataLoader(train_set, batch_size=BATCH, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=BATCH, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = CNN_CIFAR10().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    hist = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(EPOCHS):
        # --- TRAIN ---
        model.train()
        tl, tc, tt = 0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            tl += loss.item() * imgs.size(0)
            tc += (out.argmax(1) == labels).sum().item()
            tt += imgs.size(0)
        # --- VALIDATION ---
        model.eval()
        vl, vc, vt = 0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                out = model(imgs)
                loss = criterion(out, labels)
                vl += loss.item() * imgs.size(0)
                vc += (out.argmax(1) == labels).sum().item()
                vt += imgs.size(0)

        hist['train_loss'].append(tl/tt)
        hist['train_acc'].append(tc/tt)
        hist['val_loss'].append(vl/vt)
        hist['val_acc'].append(vc/vt)

        print(f"Epoch {epoch+1:02d}/{EPOCHS} | "
              f"Train Loss {tl/tt:.4f} Acc {tc/tt:.4f} | "
              f"Val Loss {vl/vt:.4f} Acc {vc/vt:.4f}")

    # --- GRÁFICA LOSS / ACCURACY ---
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(hist['train_loss'], label='train')
    ax[0].plot(hist['val_loss'], label='val')
    ax[0].set_title('Loss'); ax[0].set_xlabel('Época'); ax[0].legend()
    ax[1].plot(hist['train_acc'], label='train')
    ax[1].plot(hist['val_acc'], label='val')
    ax[1].set_title('Accuracy'); ax[1].set_xlabel('Época'); ax[1].legend()
    plt.tight_layout()
    plt.savefig('outputs/loss_accuracy.png', dpi=120)
    plt.close()

    # --- MATRIZ DE CONFUSIÓN ---
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            out = model(imgs)
            y_pred.extend(out.argmax(1).cpu().numpy())
            y_true.extend(labels.numpy())

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=CLASSES)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, xticks_rotation=45, cmap='Blues', colorbar=False)
    plt.tight_layout()
    plt.savefig('outputs/confusion_matrix.png', dpi=120)
    plt.close()

    acc = np.mean(np.array(y_true) == np.array(y_pred))
    print(f"\n✔ Precisión final en Test: {acc*100:.2f}%")

if __name__ == "__main__":
    main()