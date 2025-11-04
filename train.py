import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib, json
from pathlib import Path

X = np.load("models/embeddings.npy")
y = pd.read_csv("models/labels.csv")["label"].values

# Split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Scaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Clasificador
clf = LogisticRegression(max_iter=200, class_weight='balanced')
clf.fit(X_train, y_train)

# Evaluación
y_pred = clf.predict(X_val)
y_proba = clf.predict_proba(X_val)[:, 1]
metrics = {
    "accuracy": float(accuracy_score(y_val, y_pred)),
    "roc_auc": float(roc_auc_score(y_val, y_proba))
}

# Guardar artefactos
Path("models").mkdir(exist_ok=True)
Path("reports").mkdir(exist_ok=True)

joblib.dump(clf, "models/model.joblib")
joblib.dump(scaler, "models/scaler.joblib")
with open("reports/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("Entrenamiento completo")
print(json.dumps(metrics, indent=2))
