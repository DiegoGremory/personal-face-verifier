"""
Script de evaluación para el modelo de verificación facial.

Este script evalúa el modelo entrenado en un conjunto de prueba,
genera métricas detalladas y visualizaciones como matriz de confusión,
y guarda los resultados en el directorio de reportes.
"""

import os
import pickle
import joblib
import numpy as np
import json
import argparse
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import matplotlib
matplotlib.use('Agg')  # Backend sin GUI
import matplotlib.pyplot as plt
import seaborn as sns


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


def evaluate_model(model, scaler, X, y):
    """
    Evalúa el modelo y retorna métricas.
    
    Args:
        model: Modelo entrenado
        scaler: Scaler para normalizar datos
        X (np.array): Datos de entrada
        y (np.array): Etiquetas verdaderas
    
    Returns:
        dict: Diccionario con métricas
    """
    # Escalar datos
    X_scaled = scaler.transform(X)
    
    # Predecir
    y_pred = model.predict(X_scaled)
    y_proba = model.predict_proba(X_scaled)
    
    # Calcular métricas
    metrics = {
        'accuracy': float(accuracy_score(y, y_pred)),
        'precision': float(precision_score(y, y_pred)),
        'recall': float(recall_score(y, y_pred)),
        'f1_score': float(f1_score(y, y_pred)),
        'support': {
            'not_me': int(np.sum(y == 0)),
            'me': int(np.sum(y == 1))
        }
    }
    
    return metrics, y_pred, y_proba


def plot_confusion_matrix(y_true, y_pred, output_path):
    """
    Genera y guarda la matriz de confusión.
    
    Args:
        y_true (np.array): Etiquetas verdaderas
        y_pred (np.array): Predicciones
        output_path (str): Ruta para guardar la imagen
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Not Me', 'Me'],
        yticklabels=['Not Me', 'Me']
    )
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix saved to: {output_path}")


def main():
    """Función principal del script de evaluación."""
    parser = argparse.ArgumentParser(description='Evaluate face verification model')
    parser.add_argument(
        '--embeddings',
        type=str,
        default='data/embeddings.pkl',
        help='Path to embeddings file'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='models/model.joblib',
        help='Path to trained model'
    )
    parser.add_argument(
        '--scaler',
        type=str,
        default='models/scaler.joblib',
        help='Path to scaler'
    )
    parser.add_argument(
        '--metrics-output',
        type=str,
        default='reports/metrics.json',
        help='Path to save metrics JSON'
    )
    parser.add_argument(
        '--confusion-matrix-output',
        type=str,
        default='reports/confusion_matrix.png',
        help='Path to save confusion matrix'
    )
    
    args = parser.parse_args()
    
    # Cargar modelo y scaler
    print(f"Loading model from {args.model}...")
    model = joblib.load(args.model)
    
    print(f"Loading scaler from {args.scaler}...")
    scaler = joblib.load(args.scaler)
    
    # Cargar embeddings
    print(f"Loading embeddings from {args.embeddings}...")
    X, y = load_embeddings(args.embeddings)
    print(f"Loaded {len(X)} samples")
    
    # Evaluar modelo
    print("\nEvaluating model...")
    metrics, y_pred, y_proba = evaluate_model(model, scaler, X, y)
    
    # Mostrar métricas
    print("\nMetrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-score:  {metrics['f1_score']:.4f}")
    print(f"\nSupport:")
    print(f"  Not me: {metrics['support']['not_me']}")
    print(f"  Me:     {metrics['support']['me']}")
    
    # Classification report detallado
    print("\nClassification Report:")
    print(classification_report(
        y, y_pred,
        target_names=['Not Me', 'Me']
    ))
    
    # Guardar métricas
    os.makedirs(os.path.dirname(args.metrics_output), exist_ok=True)
    with open(args.metrics_output, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nMetrics saved to: {args.metrics_output}")
    
    # Generar y guardar matriz de confusión
    os.makedirs(os.path.dirname(args.confusion_matrix_output), exist_ok=True)
    plot_confusion_matrix(y, y_pred, args.confusion_matrix_output)


if __name__ == '__main__':
    main()
