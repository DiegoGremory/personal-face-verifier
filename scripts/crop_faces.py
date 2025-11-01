"""
Script para detectar y recortar caras de imágenes.

Este script procesa imágenes de los directorios 'me' y 'not_me',
detecta caras usando un detector facial, y guarda las caras recortadas
en el directorio 'cropped' para su posterior uso en entrenamiento.
"""

import os
import cv2
import argparse
from pathlib import Path


def detect_and_crop_faces(input_dir, output_dir, cascade_path=None):
    """
    Detecta y recorta caras de las imágenes en input_dir.
    
    Args:
        input_dir (str): Directorio con imágenes de entrada
        output_dir (str): Directorio donde guardar caras recortadas
        cascade_path (str): Ruta al archivo cascade de OpenCV
    """
    # Usar el detector de caras de OpenCV
    if cascade_path is None:
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    # Crear directorio de salida si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Procesar cada imagen
    processed = 0
    failed = 0
    
    for img_file in Path(input_dir).glob('*'):
        if img_file.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
            continue
        
        # Leer imagen
        img = cv2.imread(str(img_file))
        if img is None:
            print(f"Error reading {img_file}")
            failed += 1
            continue
        
        # Convertir a escala de grises
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detectar caras
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # Guardar cada cara detectada
        for i, (x, y, w, h) in enumerate(faces):
            face = img[y:y+h, x:x+w]
            output_path = os.path.join(
                output_dir,
                f"{img_file.stem}_face_{i}{img_file.suffix}"
            )
            cv2.imwrite(output_path, face)
            processed += 1
        
        if len(faces) == 0:
            print(f"No faces detected in {img_file}")
            failed += 1
    
    print(f"\nProcessed: {processed} faces")
    print(f"Failed/No faces: {failed} images")


def main():
    """Función principal del script."""
    parser = argparse.ArgumentParser(description='Detect and crop faces from images')
    parser.add_argument(
        '--input',
        type=str,
        default='data/me',
        help='Input directory with images'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/cropped/me',
        help='Output directory for cropped faces'
    )
    parser.add_argument(
        '--cascade',
        type=str,
        default=None,
        help='Path to Haar cascade file'
    )
    
    args = parser.parse_args()
    
    print(f"Processing images from: {args.input}")
    print(f"Saving cropped faces to: {args.output}")
    
    detect_and_crop_faces(args.input, args.output, args.cascade)


if __name__ == '__main__':
    main()
