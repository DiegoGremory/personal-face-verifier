import os
import io
import time
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import joblib
from dotenv import load_dotenv
from flask import send_from_directory

# ----------------------------
# CONFIGURACIÓN
# ----------------------------
load_dotenv()
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

MODEL_VERSION = os.getenv("MODEL_VERSION", "me-verifier-v1")
THRESHOLD = float(os.getenv("THRESHOLD", 0.95))

# Rutas de artefactos
MODEL_PATH = "models/model.joblib"
SCALER_PATH = "models/scaler.joblib"

# ----------------------------
# INICIALIZACIÓN DE MODELOS
# ----------------------------
print("🔹 Cargando modelo de verificación facial...")
mtcnn = MTCNN(image_size=160, margin=0, device=DEVICE)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

app = Flask(__name__)

# ----------------------------
# UTILIDADES
# ----------------------------
def allowed_file(filename):
    return filename.lower().endswith(('.jpg', '.jpeg', '.png'))

def preprocess_image(file_bytes):
    """Detecta y recorta el rostro usando MTCNN."""
    img = Image.open(io.BytesIO(file_bytes)).convert('RGB')
    face = mtcnn(img)
    if face is None:
        return None
    return face.unsqueeze(0).to(DEVICE)

# ----------------------------
# ENDPOINTS
# ----------------------------
@app.route("/healthz", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "model_version": MODEL_VERSION})

@app.route("/verify", methods=["POST"])
def verify():
    start_time = time.time()

    # Validar archivo
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type (only jpg/png allowed)"}), 400

    file_bytes = file.read()

    try:
        face_tensor = preprocess_image(file_bytes)
        if face_tensor is None:
            return jsonify({"error": "No face detected"}), 400

        # Embedding facial
        with torch.no_grad():
            emb = resnet(face_tensor).cpu().numpy()

        # Escalar y predecir
        emb_scaled = scaler.transform(emb)
        if hasattr(model, "predict_proba"):
            score = model.predict_proba(emb_scaled)[0, 1]
        else:
            # Para SVM lineal sin probas
            from sklearn.preprocessing import MinMaxScaler
            s = model.decision_function(emb_scaled).reshape(-1, 1)
            score = MinMaxScaler().fit_transform(s)[0, 0]

        # Verificación
        is_me = bool(score >= THRESHOLD)
        elapsed = (time.time() - start_time) * 1000

        return jsonify({
            "model_version": MODEL_VERSION,
            "is_me": is_me,
            "score": round(float(score), 4),
            "threshold": THRESHOLD,
            "timing_ms": round(elapsed, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/')
def serve_frontend():
    # Devuelve el archivo index.html desde la carpeta /static
    return send_from_directory('../static', 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    # Permite servir también script.js, style.css, etc.
    return send_from_directory('../static', filename)
    
# ----------------------------
# MAIN (para debug local)
# ----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
