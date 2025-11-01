"""
Tests para la API de verificación facial.

Este módulo contiene tests unitarios y de integración para
verificar el correcto funcionamiento de los endpoints de la API.
"""

import unittest
import json
import io
import os
import sys
from PIL import Image

# Añadir el directorio raíz al path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from api.app import app


class TestAPIEndpoints(unittest.TestCase):
    """Tests para los endpoints de la API."""
    
    def setUp(self):
        """Configurar el cliente de prueba."""
        self.app = app
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()
    
    def test_health_endpoint(self):
        """Test del endpoint /health."""
        response = self.client.get('/health')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertIn('status', data)
        self.assertEqual(data['status'], 'ok')
        self.assertIn('model_loaded', data)
        self.assertIn('scaler_loaded', data)
    
    def test_verify_endpoint_no_image(self):
        """Test del endpoint /verify sin imagen."""
        response = self.client.post('/verify')
        self.assertEqual(response.status_code, 400)
        
        data = json.loads(response.data)
        self.assertIn('error', data)
    
    def test_verify_endpoint_with_image(self):
        """Test del endpoint /verify con imagen."""
        # Crear imagen de prueba
        img = Image.new('RGB', (100, 100), color='red')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        # Enviar imagen
        response = self.client.post(
            '/verify',
            data={'image': (img_bytes, 'test.jpg')},
            content_type='multipart/form-data'
        )
        
        # Puede ser 503 si el modelo no está cargado, o 200 si está cargado
        self.assertIn(response.status_code, [200, 503])
    
    def test_predict_endpoint_no_data(self):
        """Test del endpoint /predict sin datos."""
        response = self.client.post(
            '/predict',
            data=json.dumps({}),
            content_type='application/json'
        )
        
        # Puede ser 400 (no embeddings) o 503 (modelo no cargado)
        self.assertIn(response.status_code, [400, 503])
    
    def test_predict_endpoint_with_embeddings(self):
        """Test del endpoint /predict con embeddings."""
        # Crear embeddings de prueba
        embeddings = [0.1] * 128
        
        response = self.client.post(
            '/predict',
            data=json.dumps({'embeddings': embeddings}),
            content_type='application/json'
        )
        
        # Puede ser 200 si el modelo está cargado, o 503 si no
        self.assertIn(response.status_code, [200, 503])
        
        if response.status_code == 200:
            data = json.loads(response.data)
            self.assertIn('prediction', data)
            self.assertIn('is_me', data)
            self.assertIn('probability', data)


class TestAPIResponses(unittest.TestCase):
    """Tests para las respuestas de la API."""
    
    def setUp(self):
        """Configurar el cliente de prueba."""
        self.app = app
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()
    
    def test_health_response_structure(self):
        """Test de la estructura de respuesta de /health."""
        response = self.client.get('/health')
        data = json.loads(response.data)
        
        # Verificar campos requeridos
        self.assertIsInstance(data['status'], str)
        self.assertIsInstance(data['model_loaded'], bool)
        self.assertIsInstance(data['scaler_loaded'], bool)
    
    def test_invalid_endpoint(self):
        """Test de endpoint inválido."""
        response = self.client.get('/invalid')
        self.assertEqual(response.status_code, 404)


if __name__ == '__main__':
    unittest.main()
