# Evaluación Segunda Unidad IA — CNN CIFAR-10 (Semilla 59)

## Datos del estudiante
- **Nombre:** Rodriguez Castro Luz Dioneth
- **Código:** 021201359B
- **Semilla utilizada:** `59` (últimos 2 dígitos)

---

## Pregunta 1: Carga, visualización y pipeline

- Se cargó CIFAR-10 (50 000 train / 10 000 test) desde `torchvision.datasets`.
- **Semilla 59** aplicada a `torch.manual_seed` y `np.random.seed` para garantizar reproducibilidad.
- **Visualización**: 16 imágenes del dataset mostradas con su clase (`avión`, `gato`, etc.).
- **Transformaciones**: `RandomHorizontalFlip(p=0.5)`, `RandomCrop(32, padding=4)`, `ToTensor` y `Normalize` con media/std de CIFAR-10.
- **Partición estratificada** 80/20 → 40 000 train / 10 000 val, manteniendo balance de clases.

## Pregunta 2: Justificación CNN vs MLP

Un **MLP** sobre imágenes CIFAR-10 (32×32×3 = 3 072 entradas) requeriría ~3 072 × N conexiones solo en la primera capa, **destruye la estructura espacial** (aplanamiento) y **no reconoce patrones locales** (bordes, texturas) independientemente de la posición.

La **CNN** soluciona esto con:
- **Conexiones locales**: cada neurona observa solo un parche pequeño (campo receptivo).
- **Weight sharing**: el mismo filtro se desliza por toda la imagen → drástica reducción de parámetros.
- **Invariancia traslacional**: detecta un patrón sin importar su ubicación.
- **Jerarquía**: capas tempranas detectan bordes, capas profundas formas y objetos completos.

## Pregunta 3: Manipulación de tensores 4D

La arquitectura procesa tensores `[Batch, Channels, H, W]`:

| Etapa | Shape salida |
|---|---|
| Input | [8, 3, 32, 32] |
| conv1 + pool1 | [8, 32, 16, 16] |
| conv2 + pool2 | [8, 64, 8, 8] |
| conv3 + pool3 | [8, 128, 4, 4] |
| Flatten | [8, 2048] |
| fc1 + fc2 | [8, 10] |

Se incluyó una **aserción** (`assert out.shape == (8,10)`) para garantizar el flujo 4D correcto.

## Pregunta 4: Entrenamiento y análisis de métricas

- **Optimizador:** Adam (lr = 0.001)
- **Loss:** CrossEntropyLoss
- **Épocas:** 15
- **Batch:** 64
- **Data augmentation** aplicada en train (no en val/test).

### Análisis crítico
- Al inicio `train_loss` y `val_loss` descienden juntos → **aprendizaje correcto**.
- Si `val_loss` comienza a subir mientras `train_loss` sigue bajando → **overfitting**; mitigado con `Dropout(0.5)` y data augmentation.
- No se observa **underfitting** porque ambas curvas bajan en las primeras épocas.
- La matriz de confusión muestra las clases más confundidas (típicamente `gato↔perro`, `ciervo↔caballo`), coherente con su similitud visual.
- **Conclusión:** el modelo generaliza razonablemente; la precisión final en test obtenida con semilla 59 se documenta en `outputs/loss_accuracy.png`.

## Estructura del repositorio
