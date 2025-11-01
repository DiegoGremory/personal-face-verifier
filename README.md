# me-verifier

Sistema de verificación de identidad "¿soy yo?" vía reconocimiento facial. Expone API REST con modelo de embeddings y clasificador sencillo para detectar al usuario en una foto, devolviendo decisión y puntaje de confianza.

## Estructura del Proyecto

```
me-verifier/
├── api/
│   └── app.py                 # Flask API con endpoints de verificación
├── models/
│   ├── model.joblib          # Modelo clasificador entrenado
│   └── scaler.joblib         # Scaler para normalización de embeddings
├── data/
│   ├── me/                   # Imágenes del usuario autenticado
│   ├── not_me/               # Imágenes de otras personas
│   └── cropped/              # Caras recortadas procesadas
├── scripts/
│   ├── crop_faces.py         # Detectar y recortar caras
│   ├── embeddings.py         # Extraer embeddings faciales
│   └── run_gunicorn.sh       # Script para ejecutar con Gunicorn
├── reports/
│   ├── metrics.json          # Métricas de evaluación del modelo
│   └── confusion_matrix.png  # Visualización de matriz de confusión
├── tests/
│   └── test_api.py           # Tests unitarios de la API
├── train.py                  # Script de entrenamiento del modelo
├── evaluate.py               # Script de evaluación del modelo
├── requirements.txt          # Dependencias de Python
├── .env.example              # Ejemplo de variables de entorno
└── README.md                 # Documentación del proyecto
```

## Instalación

1. Clonar el repositorio:
```bash
git clone https://github.com/DiegoGremory/personal-face-verifier.git
cd personal-face-verifier
```

2. Crear un entorno virtual:
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. Instalar dependencias:
```bash
pip install -r requirements.txt
```

4. Configurar variables de entorno:
```bash
cp .env.example .env
# Editar .env según sea necesario
```

## Uso

### 1. Preparar Datos

Colocar imágenes en los directorios correspondientes:
- `data/me/`: Imágenes del usuario autenticado
- `data/not_me/`: Imágenes de otras personas

### 2. Procesar Imágenes

Recortar caras de las imágenes:
```bash
# Procesar imágenes "me"
python scripts/crop_faces.py --input data/me --output data/cropped/me

# Procesar imágenes "not_me"
python scripts/crop_faces.py --input data/not_me --output data/cropped/not_me
```

### 3. Extraer Embeddings

Extraer embeddings faciales de las imágenes procesadas:
```bash
python scripts/embeddings.py \
    --me-dir data/cropped/me \
    --not-me-dir data/cropped/not_me \
    --output data/embeddings.pkl
```

### 4. Entrenar Modelo

Entrenar el clasificador con los embeddings:
```bash
python train.py \
    --embeddings data/embeddings.pkl \
    --model-type svm \
    --model-output models/model.joblib \
    --scaler-output models/scaler.joblib
```

Opciones de modelo:
- `svm`: Support Vector Machine (recomendado)
- `rf`: Random Forest

### 5. Evaluar Modelo

Generar métricas y visualizaciones:
```bash
python evaluate.py \
    --embeddings data/embeddings.pkl \
    --model models/model.joblib \
    --scaler models/scaler.joblib \
    --metrics-output reports/metrics.json \
    --confusion-matrix-output reports/confusion_matrix.png
```

### 6. Ejecutar API

#### Modo Desarrollo

```bash
python api/app.py
```

#### Modo Producción (Gunicorn)

```bash
./scripts/run_gunicorn.sh
```

O con configuración personalizada:
```bash
HOST=0.0.0.0 PORT=8000 WORKERS=4 ./scripts/run_gunicorn.sh
```

## API Endpoints

### Health Check
```bash
GET /health
```

Respuesta:
```json
{
  "status": "ok",
  "model_loaded": true,
  "scaler_loaded": true
}
```

### Verificar Imagen
```bash
POST /verify
Content-Type: multipart/form-data

image: <archivo de imagen>
```

Respuesta:
```json
{
  "is_me": true,
  "confidence": 0.95,
  "message": "Verification successful"
}
```

### Predecir con Embeddings
```bash
POST /predict
Content-Type: application/json

{
  "embeddings": [0.1, 0.2, ..., 0.128]
}
```

Respuesta:
```json
{
  "prediction": 1,
  "is_me": true,
  "probability": {
    "not_me": 0.05,
    "me": 0.95
  }
}
```

## Tests

Ejecutar tests unitarios:
```bash
# Con unittest
python -m unittest discover tests

# Con pytest
pytest tests/ -v
```

## Estructura de Archivos Generados

### models/model.joblib
Modelo clasificador entrenado (SVM o Random Forest) serializado con joblib.

### models/scaler.joblib
StandardScaler de scikit-learn para normalizar embeddings.

### reports/metrics.json
Métricas de evaluación en formato JSON:
```json
{
  "accuracy": 0.95,
  "precision": 0.93,
  "recall": 0.97,
  "f1_score": 0.95,
  "support": {
    "not_me": 100,
    "me": 100
  }
}
```

### reports/confusion_matrix.png
Visualización de la matriz de confusión del modelo.

## Tecnologías Utilizadas

- **Flask**: Framework web para la API REST
- **scikit-learn**: Entrenamiento y evaluación de modelos
- **OpenCV**: Detección y procesamiento de caras
- **Pillow**: Manipulación de imágenes
- **Gunicorn**: Servidor WSGI para producción
- **Matplotlib/Seaborn**: Visualización de métricas

## Licencia

Este proyecto está disponible bajo la licencia MIT.

## Autor

Diego Gremory
