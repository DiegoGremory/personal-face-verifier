"""
Flask API for personal face verification.

This module provides REST endpoints for facial verification,
determining if a person in an image is the authenticated user.
"""

from flask import Flask, request, jsonify
import os
import joblib
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Configuración
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'model.joblib')
SCALER_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'scaler.joblib')

# Cargar modelo y scaler (inicializar como None si no existen aún)
model = None
scaler = None

try:
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)
except Exception as e:
    print(f"Warning: Could not load models: {e}")


@app.route('/health', methods=['GET'])
def health():
    """
    Health check endpoint.
    
    Returns:
        JSON response with service status
    """
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None
    })


@app.route('/verify', methods=['POST'])
def verify():
    """
    Verify if the person in the uploaded image is the authenticated user.
    
    Expected request:
        - multipart/form-data with 'image' file
    
    Returns:
        JSON response with verification result and confidence score
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    if model is None or scaler is None:
        return jsonify({'error': 'Model not loaded'}), 503
    
    try:
        # Leer imagen
        image_file = request.files['image']
        image = Image.open(io.BytesIO(image_file.read()))
        
        # Aquí iría el procesamiento de la imagen y extracción de embeddings
        # Por ahora, devolver respuesta de ejemplo
        
        return jsonify({
            'is_me': True,
            'confidence': 0.95,
            'message': 'Verification successful (placeholder)'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict', methods=['POST'])
def predict():
    """
    Alternative endpoint for prediction with raw embeddings.
    
    Expected request:
        - JSON with 'embeddings' array
    
    Returns:
        JSON response with prediction and probability
    """
    if model is None or scaler is None:
        return jsonify({'error': 'Model not loaded'}), 503
    
    try:
        data = request.get_json()
        
        if 'embeddings' not in data:
            return jsonify({'error': 'No embeddings provided'}), 400
        
        embeddings = np.array(data['embeddings']).reshape(1, -1)
        
        # Escalar embeddings
        embeddings_scaled = scaler.transform(embeddings)
        
        # Predecir
        prediction = model.predict(embeddings_scaled)[0]
        probability = model.predict_proba(embeddings_scaled)[0]
        
        return jsonify({
            'prediction': int(prediction),
            'is_me': bool(prediction == 1),
            'probability': {
                'not_me': float(probability[0]),
                'me': float(probability[1])
            }
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Ejecutar en modo desarrollo
    app.run(debug=True, host='0.0.0.0', port=5000)
