"""
Script para extraer embeddings faciales de imágenes.

Este script utiliza un modelo pre-entrenado (como FaceNet o similar)
para convertir imágenes de caras en vectores de embeddings que pueden
ser utilizados para entrenar el clasificador de verificación.
"""

import os
import numpy as np
import argparse
from pathlib import Path
import pickle


def extract_embeddings(image_path, model=None):
    """
    Extrae embeddings de una imagen de cara.
    
    Args:
        image_path (str): Ruta a la imagen
        model: Modelo pre-entrenado para extraer embeddings
    
    Returns:
        np.array: Vector de embeddings
    """
    # Aquí se implementaría la extracción real de embeddings
    # usando un modelo como FaceNet, VGGFace, etc.
    
    # Por ahora, devolver embeddings de ejemplo (128 dimensiones)
    embeddings = np.random.randn(128)
    return embeddings


def process_directory(input_dir, label, model=None):
    """
    Procesa todas las imágenes de un directorio y extrae embeddings.
    
    Args:
        input_dir (str): Directorio con imágenes
        label (int): Etiqueta para las imágenes (0 o 1)
        model: Modelo para extraer embeddings
    
    Returns:
        tuple: (embeddings_array, labels_array)
    """
    embeddings_list = []
    labels_list = []
    
    for img_file in Path(input_dir).glob('*'):
        if img_file.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
            continue
        
        try:
            # Extraer embeddings
            emb = extract_embeddings(str(img_file), model)
            embeddings_list.append(emb)
            labels_list.append(label)
            
        except Exception as e:
            print(f"Error processing {img_file}: {e}")
    
    return np.array(embeddings_list), np.array(labels_list)


def main():
    """Función principal del script."""
    parser = argparse.ArgumentParser(description='Extract face embeddings')
    parser.add_argument(
        '--me-dir',
        type=str,
        default='data/cropped/me',
        help='Directory with "me" images'
    )
    parser.add_argument(
        '--not-me-dir',
        type=str,
        default='data/cropped/not_me',
        help='Directory with "not me" images'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/embeddings.pkl',
        help='Output file for embeddings'
    )
    
    args = parser.parse_args()
    
    print("Extracting embeddings...")
    
    # Procesar imágenes "me" (label=1)
    me_embeddings, me_labels = process_directory(args.me_dir, label=1)
    print(f"Processed {len(me_embeddings)} 'me' images")
    
    # Procesar imágenes "not_me" (label=0)
    not_me_embeddings, not_me_labels = process_directory(args.not_me_dir, label=0)
    print(f"Processed {len(not_me_embeddings)} 'not me' images")
    
    # Combinar todos los embeddings
    all_embeddings = np.vstack([me_embeddings, not_me_embeddings])
    all_labels = np.concatenate([me_labels, not_me_labels])
    
    # Guardar embeddings
    data = {
        'embeddings': all_embeddings,
        'labels': all_labels
    }
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"\nSaved {len(all_embeddings)} embeddings to {args.output}")
    print(f"  - 'me': {len(me_embeddings)}")
    print(f"  - 'not me': {len(not_me_embeddings)}")


if __name__ == '__main__':
    main()
