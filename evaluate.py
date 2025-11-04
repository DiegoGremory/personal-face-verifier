import numpy as np
import json
from pathlib import Path
import joblib
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    roc_auc_score,
    classification_report,
)
import matplotlib.pyplot as plt

# ----------------------------
# CONFIGURACIÓN DE RUTAS
# ----------------------------
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)

# ----------------------------
# CARGA DE ARTEFACTOS
# ----------------------------
print("🔹 Cargando modelo y datos...")

model = joblib.load(MODELS_DIR / "model.joblib")
scaler = joblib.load(MODELS_DIR / "scaler.joblib")

X = np.load(MODELS_DIR / "embeddings.npy")
labels = np.loadtxt(MODELS_DIR / "labels.csv", delimiter=",", skiprows=1, usecols=1)

X_scaled = scaler.transform(X)

# ----------------------------
# PREDICCIONES Y PROBABILIDADES
# ----------------------------
if hasattr(model, "predict_proba"):
    probs = model.predict_proba(X_scaled)[:, 1]
else:
    # Para SVM lineal sin predict_proba
    from sklearn.preprocessing import MinMaxScaler
    scores = model.decision_function(X_scaled)
    probs = MinMaxScaler().fit_transform(scores.reshape(-1, 1)).ravel()

# ----------------------------
# MÉTRICAS Y UMBRAL ÓPTIMO
# ----------------------------
fpr, tpr, thresholds_roc = roc_curve(labels, probs)
precision, recall, thresholds_pr = precision_recall_curve(labels, probs)

# F1 óptimo (búsqueda de mejor threshold)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
best_idx = np.argmax(f1_scores)
best_threshold = thresholds_pr[best_idx]
best_f1 = f1_scores[best_idx]

# Predicciones binarias con umbral óptimo
preds = (probs >= best_threshold).astype(int)
cm = confusion_matrix(labels, preds)
auc = roc_auc_score(labels, probs)

# ----------------------------
# GUARDAR RESULTADOS
# ----------------------------
metrics = {
    "accuracy": float(np.mean(preds == labels)),
    "auc": float(auc),
    "f1_best": float(best_f1),
    "best_threshold": float(best_threshold),
    "confusion_matrix": cm.tolist(),
}

with open(REPORTS_DIR / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("\n✅ Evaluación completada")
print(json.dumps(metrics, indent=4))

# ----------------------------
# VISUALIZACIÓN
# ----------------------------
plt.figure()
plt.imshow(cm, cmap="Blues")
plt.title("Matriz de confusión")
plt.xlabel("Predicción")
plt.ylabel("Etiqueta real")
plt.colorbar()
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
plt.savefig(REPORTS_DIR / "confusion_matrix.png")
plt.close()

plt.figure()
plt.plot(fpr, tpr, label=f"ROC AUC = {auc:.3f}")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("Tasa de Falsos Positivos")
plt.ylabel("Tasa de Verdaderos Positivos")
plt.title("Curva ROC")
plt.legend()
plt.savefig(REPORTS_DIR / "roc_curve.png")
plt.close()

plt.figure()
plt.plot(recall, precision, label=f"F1 óptimo = {best_f1:.3f} @ τ={best_threshold:.2f}")
plt.xlabel("Recall")
plt.ylabel("Precisión")
plt.title("Curva Precision-Recall")
plt.legend()
plt.savefig(REPORTS_DIR / "pr_curve.png")
plt.close()

print("📊 Gráficos guardados en reports/:")
print(" - confusion_matrix.png")
print(" - roc_curve.png")
print(" - pr_curve.png")
