import torch
import torch.nn as nn

class CNN_CIFAR10(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # Bloque convolucional 1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)   # 32x32 -> 16x16
        # Bloque convolucional 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)   # 16x16 -> 8x8
        # Bloque convolucional 3 (opcional pero recomendado)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)   # 8x8 -> 4x4

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        # 128 * 4 * 4 = 2048
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = self.pool3(self.relu(self.conv3(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# ============================================================
# VALIDACIÓN DE SHAPES CON ASERCIÓN (obligatorio)
# ============================================================
if __name__ == "__main__":
    model = CNN_CIFAR10()
    batch = torch.randn(8, 3, 32, 32)   # batch simulado

    print("=== Validación de shapes ===")
    print(f"Input:  {batch.shape}")

    x = model.pool1(model.relu(model.conv1(batch)))
    print(f"Tras conv1+pool1: {x.shape}  (esperado [8, 32, 16, 16])")
    x = model.pool2(model.relu(model.conv2(x)))
    print(f"Tras conv2+pool2: {x.shape}  (esperado [8, 64, 8, 8])")
    x = model.pool3(model.relu(model.conv3(x)))
    print(f"Tras conv3+pool3: {x.shape}  (esperado [8, 128, 4, 4])")
    x = model.flatten(x)
    print(f"Tras Flatten:     {x.shape}  (esperado [8, 2048])")
    out = model(batch)
    print(f"Salida final:     {out.shape}  (esperado [8, 10])")

    assert out.shape == (8, 10), "Shape incorrecto"
    print("✔ Forward pass exitoso")
    print(f"Parámetros totales: {sum(p.numel() for p in model.parameters()):,}")