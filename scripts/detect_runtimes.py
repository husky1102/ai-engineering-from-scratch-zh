#!/usr/bin/env python3
"""Detect lesson runtime capabilities for the local learning site.

Usage:
    python3 scripts/detect_runtimes.py
    python3 scripts/detect_runtimes.py --stdout

The detector is intentionally conservative. Browser execution is enabled only
for light Python lessons without network, subprocess, GPU, API key, or heavy
framework markers. Manual overrides can be placed in
`site/runtime-overrides.json`.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent.parent
PHASES_DIR = ROOT / "phases"
DEFAULT_OUT = ROOT / "site" / "runtime-manifest.json"
DEFAULT_OVERRIDES = ROOT / "site" / "runtime-overrides.json"

CODE_SUFFIXES = {
    ".py": "python",
    ".ts": "typescript",
    ".js": "javascript",
    ".mjs": "javascript",
    ".rs": "rust",
    ".jl": "julia",
    ".sh": "shell",
    ".yml": "yaml",
    ".yaml": "yaml",
    ".json": "json",
    "Dockerfile": "docker",
}

PYODIDE_PACKAGES = {"numpy"}
STDLIB_LIKE = {
    "__future__",
    "abc",
    "argparse",
    "ast",
    "base64",
    "collections",
    "contextlib",
    "csv",
    "dataclasses",
    "datetime",
    "decimal",
    "enum",
    "functools",
    "hashlib",
    "heapq",
    "hmac",
    "importlib",
    "itertools",
    "json",
    "math",
    "operator",
    "pathlib",
    "random",
    "re",
    "shutil",
    "statistics",
    "string",
    "sys",
    "textwrap",
    "time",
    "typing",
    "unittest",
}

HEAVY_IMPORTS = {
    "datasets",
    "cv2",
    "h5py",
    "jax",
    "matplotlib",
    "openai",
    "PIL",
    "requests",
    "safetensors",
    "sklearn",
    "tensorflow",
    "torch",
    "torchvision",
    "torchaudio",
    "transformers",
    "urllib",
    "websocket",
    "zstandard",
}

NETWORK_OR_PROCESS_IMPORTS = {"http", "socket", "subprocess", "urllib"}
HEAVY_TEXT_PATTERNS = [
    (re.compile(r"\b[A-Z0-9_]*(?:API_KEY|SECRET|TOKEN)\b"), "uses API key or secret"),
    (re.compile(r"\bos\.environ\b"), "reads environment variables"),
    (re.compile(r"\bsubprocess\b"), "uses subprocess"),
    (re.compile(r"\bsocket\b|\burllib\.request\b|\bhttp\.server\b"), "uses network or server APIs"),
    (re.compile(r"\bcuda\b|\bgpu\b", re.IGNORECASE), "mentions GPU/CUDA"),
]


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def iter_lesson_dirs() -> Iterable[Path]:
    for phase in sorted(PHASES_DIR.iterdir()):
        if not phase.is_dir() or not phase.name[:2].isdigit():
            continue
        for lesson in sorted(phase.iterdir()):
            if lesson.is_dir() and lesson.name[:2].isdigit():
                yield lesson


def code_language(path: Path) -> str | None:
    if path.name == "Dockerfile":
        return "docker"
    return CODE_SUFFIXES.get(path.suffix)


def list_code_files(lesson: Path) -> list[Path]:
    code_dir = lesson / "code"
    if not code_dir.is_dir():
        return []
    files: list[Path] = []
    for path in code_dir.rglob("*"):
        if not path.is_file() or not code_language(path):
            continue
        rel_parts = path.relative_to(code_dir).parts
        if rel_parts and rel_parts[0] == "tests":
            continue
        files.append(path)
    return sorted(files)


def python_imports(path: Path) -> set[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (SyntaxError, UnicodeDecodeError):
        return set()
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".", 1)[0])
    return imports


def local_python_modules(files: list[Path]) -> set[str]:
    return {
        path.stem
        for path in files
        if code_language(path) == "python" and path.stem != "__init__"
    }


def choose_entry(files: list[Path], language: str) -> str | None:
    candidates = [path for path in files if code_language(path) == language]
    if not candidates:
        return None
    preferred = {
        "python": ["main.py", "linear_regression.py", "verify.py"],
        "typescript": ["main.ts", "verify.ts"],
        "javascript": ["main.js", "server.mjs"],
        "rust": ["main.rs"],
        "julia": ["main.jl"],
    }.get(language, [])
    for name in preferred:
        for path in candidates:
            if path.name == name:
                return rel(path)
    return rel(candidates[0])


def load_overrides(path: Path) -> dict[str, dict[str, object]]:
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"{rel(path)} must be a JSON object")
    overrides = data.get("lessons", data)
    if not isinstance(overrides, dict):
        raise SystemExit(f"{rel(path)} lessons must be a JSON object")
    return {str(k): v for k, v in overrides.items() if isinstance(v, dict)}


def detect_lesson(lesson: Path) -> dict[str, object]:
    files = list_code_files(lesson)
    languages = sorted({code_language(path) for path in files if code_language(path)})
    py_files = [path for path in files if code_language(path) == "python"]
    imports: set[str] = set()
    heavy_reasons: list[str] = []

    for path in py_files:
        imports.update(python_imports(path))
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            heavy_reasons.append(f"{rel(path)} is not valid UTF-8")
            continue
        for pattern, reason in HEAVY_TEXT_PATTERNS:
            if pattern.search(text) and reason not in heavy_reasons:
                heavy_reasons.append(reason)

    external_imports = imports - local_python_modules(files)

    heavy_import_hits = sorted(external_imports & HEAVY_IMPORTS)
    for name in heavy_import_hits:
        heavy_reasons.append(f"imports {name}")
    if external_imports & NETWORK_OR_PROCESS_IMPORTS:
        for name in sorted(external_imports & NETWORK_OR_PROCESS_IMPORTS):
            reason = f"imports {name}"
            if reason not in heavy_reasons:
                heavy_reasons.append(reason)

    packages = sorted(name for name in external_imports if name not in STDLIB_LIKE)

    if not files:
        runtime = "static-only"
        language = None
        notes = ["no code files detected"]
    elif py_files and not heavy_reasons and all(pkg in PYODIDE_PACKAGES for pkg in packages):
        runtime = "browser-pyodide"
        language = "python"
        notes = ["light Python candidate"]
    elif py_files:
        runtime = "local-kernel"
        language = "python"
        notes = heavy_reasons or ["Python uses packages outside the browser allowlist"]
    else:
        runtime = "static-only"
        language = languages[0] if languages else None
        notes = ["non-Python lesson; browser execution not enabled yet"]

    entry = choose_entry(files, language) if language else None
    return {
        "runtime": runtime,
        "language": language,
        "languages": languages,
        "packages": packages,
        "entry": entry,
        "notebook": None,
        "lab": None,
        "notes": notes,
    }


def apply_override(record: dict[str, object], override: dict[str, object]) -> dict[str, object]:
    merged = dict(record)
    for key, value in override.items():
        if key == "notes" and isinstance(value, list):
            merged[key] = value
        elif key in {"runtime", "language", "languages", "packages", "entry", "notebook", "lab", "notes"}:
            merged[key] = value
    return merged


def build_manifest(overrides_path: Path) -> dict[str, object]:
    overrides = load_overrides(overrides_path)
    lessons: dict[str, dict[str, object]] = {}
    for lesson in iter_lesson_dirs():
        lesson_path = rel(lesson)
        record = detect_lesson(lesson)
        if lesson_path in overrides:
            record = apply_override(record, overrides[lesson_path])
        lessons[lesson_path] = record

    totals: dict[str, int] = {}
    for record in lessons.values():
        runtime = str(record.get("runtime") or "unknown")
        totals[runtime] = totals.get(runtime, 0) + 1

    return {
        "schema_version": 1,
        "generated_by": "scripts/detect_runtimes.py",
        "totals": totals,
        "lessons": lessons,
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output JSON path")
    parser.add_argument("--overrides", type=Path, default=DEFAULT_OVERRIDES, help="runtime override JSON path")
    parser.add_argument("--stdout", action="store_true", help="print JSON instead of writing")
    args = parser.parse_args(argv)

    out_path = args.out if args.out.is_absolute() else ROOT / args.out
    overrides_path = args.overrides if args.overrides.is_absolute() else ROOT / args.overrides
    manifest = build_manifest(overrides_path)
    payload = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"

    if args.stdout:
        sys.stdout.write(payload)
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload, encoding="utf-8")

    totals = manifest["totals"]
    summary = ", ".join(f"{key}={totals[key]}" for key in sorted(totals))
    print(f"detect_runtimes.py — {len(manifest['lessons'])} lesson(s); {summary}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
