#!/usr/bin/env bash
set -euo pipefail

PYODIDE_VERSION="${PYODIDE_VERSION:-0.26.4}"
BASE_URL="${PYODIDE_BASE_URL:-https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full}"
DEST_DIR="${1:-site/vendor/pyodide/v${PYODIDE_VERSION}/full}"

download() {
  local file="$1"
  local target="${DEST_DIR}/${file}"
  if [[ -s "${target}" ]]; then
    printf 'exists %s\n' "${target}"
    return
  fi
  mkdir -p "$(dirname "${target}")"
  printf 'fetch %s\n' "${file}"
  curl -fL --retry 3 --retry-delay 2 -o "${target}.tmp" "${BASE_URL}/${file}"
  mv "${target}.tmp" "${target}"
}

mkdir -p "${DEST_DIR}"

download pyodide-lock.json

python3 - "${DEST_DIR}/pyodide-lock.json" <<'PY' | while IFS= read -r file; do
import json
import sys
from pathlib import Path

lock = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
packages = lock.get("packages", {})
wanted = {"micropip", "numpy", "packaging"}
seen = set()
files = [
    "pyodide.js",
    "pyodide.asm.js",
    "pyodide.asm.wasm",
    "python_stdlib.zip",
]

def add_package(name):
    if name in seen:
        return
    seen.add(name)
    meta = packages.get(name)
    if not isinstance(meta, dict):
        return
    file_name = meta.get("file_name") or meta.get("filename") or meta.get("file")
    if file_name:
        files.append(file_name)
    for dep in meta.get("depends", []):
        dep_name = str(dep).split()[0].split("[")[0]
        if dep_name:
            add_package(dep_name)

for name in sorted(wanted):
    add_package(name)

for file_name in dict.fromkeys(files):
    print(file_name)
PY
  download "${file}"
done

cat <<EOF
Pyodide v${PYODIDE_VERSION} is available at ${DEST_DIR}

The site runner defaults to:
  vendor/pyodide/v${PYODIDE_VERSION}/full/

Set window.AIFSPyodideRunnerConfig.pyodideBaseURL before runner-pyodide.js to
point at a different self-hosted directory.
EOF
