import torch
from facenet_pytorch import InceptionResnetV1
from PIL import Image
from pathlib import Path
import numpy as np
import csv

# ----------------------------
# CONFIGURACIÓN
# ----------------------------
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_NAME = 'vggface2'

DATA_DIR = Path("data/cropped")
OUT_DIR = Path("models")
OUT_DIR.mkdir(exist_ok=True)

# ----------------------------
# CARGA DE MODELO PREENTRENADO
# ----------------------------
model = InceptionResnetV1(pretrained=MODEL_NAME).eval().to(DEVICE)

# ----------------------------
# VARIABLES DE SALIDA
# ----------------------------
embeddings = []
labels = []
filenames = []

# ----------------------------
# PROCESAMIENTO DE IMÁGENES
# ----------------------------
print(f"Extrayendo embeddings con InceptionResnetV1 ({MODEL_NAME})...")

for label_name, label_val in [("me", 1), ("not_me", 0)]:
    img_dir = DATA_DIR / label_name
    if not img_dir.exists():
        print(f"[WARN] Carpeta no encontrada: {img_dir}")
        continue

    # Aceptar jpg, jpeg, png
    img_files = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.jpeg")) + list(img_dir.glob("*.png"))

    for img_path in img_files:
        try:
            img = Image.open(img_path).convert("RGB")
            img_np = np.array(img).astype(np.float32) / 255.0
            img_tensor = torch.tensor(img_np).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                emb = model(img_tensor).squeeze().cpu().numpy()

            embeddings.append(emb)
            labels.append(label_val)
            filenames.append(img_path.name)
        except Exception as e:
            print(f"[ERROR] {img_path.name}: {e}")

# ----------------------------
# VALIDACIÓN Y GUARDADO
# ----------------------------
if len(embeddings) == 0:
    raise ValueError("No se generaron embeddings. Revisa las rutas y formatos de las imágenes.")

embeddings = np.array(embeddings)
np.save(OUT_DIR / "embeddings.npy", embeddings)

# CSV con etiquetas y nombres
csv_path = OUT_DIR / "labels.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "label"])
    for fname, label in zip(filenames, labels):
        writer.writerow([fname, label])

print(f"\n Embeddings generados: {embeddings.shape}")
print(f"Guardado en: {OUT_DIR}")
print(f"CSV de etiquetas: {csv_path}")
