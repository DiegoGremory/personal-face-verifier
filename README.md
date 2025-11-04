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
python scripts/crop_faces.py
```

### 3. Extraer Embeddings

Extraer embeddings faciales de las imágenes procesadas:
```bash
python scripts/embeddings.py 
```

### 4. Entrenar Modelo

Entrenar el clasificador con los embeddings:
```bash
python train.py 
```

### 5. Evaluar Modelo

Generar métricas y visualizaciones:
```bash
python evaluate.py 
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

## API Endpoints

### Health Check
```bash
GET /health
```

Respuesta:
```json
{
"model_version": "me-verifier-v1",
   "status": "ok"
}
```

### Verificar Imagen
```bash
POST /verify
```
Respuesta no yo:
```json
{
"is_me": false,
    "model_version": "me-verifier-v1",
    "score": 0.0,
    "threshold": 0.95,
    "timing_ms": 318.25
}
```

Respuesta yo:
```json
{
    "is_me": true,
    "model_version": "me-verifier-v1",
    "score": 0.9996,
    "threshold": 0.95,
    "timing_ms": 1058.52
}
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
    "accuracy": 0.9978947368421053,
    "auc": 0.9999333333333333,
    "f1_best": 0.993288585604252,
    "best_threshold": 0.9936454977148538,
    "confusion_matrix": [
        [
            400,
            0
        ],
        [
            1,
            74
        ]
    ]
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

Diego Lopez
