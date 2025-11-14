#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$ROOT_DIR/env"
BACKEND="$ROOT_DIR/web/scout_web_backend/app.py"
FRONTEND_DIR="$ROOT_DIR/web/scout-ui"

if [ ! -d "$VENV_DIR" ]; then
  echo "Python virtual environment not found at $VENV_DIR"
  exit 1
fi

# Ensure backend dependencies are available
"$VENV_DIR/bin/pip" install --quiet Flask Flask-Cors

pushd "$FRONTEND_DIR" > /dev/null
if [ ! -d node_modules ]; then
  npm install
fi
npm run build
popd > /dev/null

STATIC_DIR="$FRONTEND_DIR/dist"
exec "$VENV_DIR/bin/python" "$BACKEND" --static-dir "$STATIC_DIR" "$@"
