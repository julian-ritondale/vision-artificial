#!/usr/bin/env bash

# Si el script se ejecuta con 'source activate.sh', define la función 'activate' en la terminal
if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
    activate() {
        bash "${BASH_SOURCE[0]}" "$@"
    }
    echo "Función 'activate' disponible en la terminal. Uso: activate tp1"
    return 0 2>/dev/null || true
fi

if [ -z "$1" ]; then
    echo "Uso: ./activate.sh <tpX>  (ejemplo: ./activate.sh tp1 o ./activate.sh 1)"
    exit 1
fi

TARGET=$(echo "$1" | tr '[:upper:]' '[:lower:]')

if [[ "$TARGET" =~ ^[0-9]+$ ]]; then
    TP_DIR="tp$TARGET"
else
    TP_DIR="$TARGET"
fi

MAIN_FILE="${TP_DIR}/main.py"

if [ ! -f "$MAIN_FILE" ]; then
    echo "Error: No se encontró el archivo $MAIN_FILE"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"

if [ -d "$VENV_DIR" ]; then
    echo "Activando entorno virtual (.venv)..."
    source "${VENV_DIR}/bin/activate"
fi

echo "Ejecutando ${MAIN_FILE}..."
python3 "${MAIN_FILE}"
