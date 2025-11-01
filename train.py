"""
Script de entrenamiento para el modelo de verificación facial.

Este script carga los embeddings extraídos, divide los datos en
conjuntos de entrenamiento y validación, entrena un clasificador,
y guarda el modelo y el scaler entrenados.
"""

import os
import pickle
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import argparse


def load_embeddings(embeddings_path):
    """
    Carga los embeddings desde un archivo pickle.
    
    Args:
        embeddings_path (str): Ruta al archivo de embeddings
    
    Returns:
        tuple: (X, y) arrays de embeddings y etiquetas
    """
    with open(embeddings_path, 'rb') as f:
        data = pickle.load(f)
    
    return data['embeddings'], data['labels']


def train_model(X_train, y_train, model_type='svm'):
    """
    Entrena un modelo de clasificación.
    
    Args:
        X_train (np.array): Datos de entrenamiento
        y_train (np.array): Etiquetas de entrenamiento
        model_type (str): Tipo de modelo ('svm' o 'rf')
    
    Returns:
        Modelo entrenado
    """
    if model_type == 'svm':
        # Support Vector Machine con probabilidades
        model = SVC(kernel='linear', probability=True, random_state=42)
    elif model_type == 'rf':
        # Random Forest
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    print(f"Training {model_type.upper()} model...")
    model.fit(X_train, y_train)
    
    return model


def main():
    """Función principal del script de entrenamiento."""
    parser = argparse.ArgumentParser(description='Train face verification model')
    parser.add_argument(
        '--embeddings',
        type=str,
        default='data/embeddings.pkl',
        help='Path to embeddings file'
    )
    parser.add_argument(
        '--model-type',
        type=str,
        default='svm',
        choices=['svm', 'rf'],
        help='Type of model to train'
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Proportion of data for testing'
    )
    parser.add_argument(
        '--model-output',
        type=str,
        default='models/model.joblib',
        help='Path to save trained model'
    )
    parser.add_argument(
        '--scaler-output',
        type=str,
        default='models/scaler.joblib',
        help='Path to save scaler'
    )
    
    args = parser.parse_args()
    
    # Cargar embeddings
    print(f"Loading embeddings from {args.embeddings}...")
    X, y = load_embeddings(args.embeddings)
    print(f"Loaded {len(X)} samples")
    print(f"  - Class 0 (not me): {np.sum(y == 0)}")
    print(f"  - Class 1 (me): {np.sum(y == 1)}")
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )
    print(f"\nTrain samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Escalar datos
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Entrenar modelo
    model = train_model(X_train_scaled, y_train, args.model_type)
    
    # Evaluar en conjunto de entrenamiento
    y_train_pred = model.predict(X_train_scaled)
    train_acc = accuracy_score(y_train, y_train_pred)
    print(f"\nTraining accuracy: {train_acc:.4f}")
    
    # Evaluar en conjunto de prueba
    y_test_pred = model.predict(X_test_scaled)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    
    print(f"\nTest metrics:")
    print(f"  Accuracy:  {test_acc:.4f}")
    print(f"  Precision: {test_precision:.4f}")
    print(f"  Recall:    {test_recall:.4f}")
    print(f"  F1-score:  {test_f1:.4f}")
    
    # Guardar modelo y scaler
    os.makedirs(os.path.dirname(args.model_output), exist_ok=True)
    os.makedirs(os.path.dirname(args.scaler_output), exist_ok=True)
    
    joblib.dump(model, args.model_output)
    joblib.dump(scaler, args.scaler_output)
    
    print(f"\nModel saved to: {args.model_output}")
    print(f"Scaler saved to: {args.scaler_output}")


if __name__ == '__main__':
    main()
