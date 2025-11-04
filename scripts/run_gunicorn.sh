#!/bin/bash
# ==============================================================
# Run Gunicorn for Flask API
# ==============================================================

APP_MODULE="api.app:app"
HOST="0.0.0.0"
PORT="5000"
WORKERS=2
LOG_DIR="logs"
ACCESS_LOG="$LOG_DIR/gunicorn_access.log"
ERROR_LOG="$LOG_DIR/gunicorn_error.log"

# Crear carpeta de logs si no existe
mkdir -p $LOG_DIR

echo "🚀 Starting Gunicorn server..."
echo "→ Module: $APP_MODULE"
echo "→ Listening on: http://$HOST:$PORT"
echo "→ Workers: $WORKERS"

# Ejecutar Gunicorn
exec gunicorn -w $WORKERS -b $HOST:$PORT $APP_MODULE \
  --access-logfile $ACCESS_LOG \
  --error-logfile $ERROR_LOG \
  --timeout 60 \
  --log-level info
