#!/bin/bash
#
# Script para ejecutar la API Flask con Gunicorn en producción.
#
# Gunicorn es un servidor WSGI de Python para producción que permite
# manejar múltiples workers y conexiones concurrentes.

# Configuración
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-5000}
WORKERS=${WORKERS:-4}
TIMEOUT=${TIMEOUT:-120}

# Directorio de la aplicación
APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Starting Gunicorn server..."
echo "  Host: $HOST"
echo "  Port: $PORT"
echo "  Workers: $WORKERS"
echo "  Timeout: $TIMEOUT"

cd "$APP_DIR"

# Ejecutar Gunicorn
gunicorn \
    --bind "$HOST:$PORT" \
    --workers "$WORKERS" \
    --timeout "$TIMEOUT" \
    --access-logfile - \
    --error-logfile - \
    api.app:app
