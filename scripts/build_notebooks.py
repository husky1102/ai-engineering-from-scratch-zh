#!/usr/bin/env python3
"""Build lesson notebooks from Markdown docs.

Usage:
    python3 scripts/build_notebooks.py
    python3 scripts/build_notebooks.py --limit 20
    python3 scripts/build_notebooks.py --all

The default generates 20 lightweight Python notebooks, enough for the first
JupyterLite milestone without creating a huge generated diff. Use `--all` for
every browser-pyodide lesson.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNTIME_MANIFEST = ROOT / "site" / "runtime-manifest.json"
NOTEBOOK_ROOT = ROOT / "site" / "notebooks"
FENCE_RE = re.compile(r"^```([^\s`]*)\s*$")


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def lab_url_for(notebook_path: str) -> str:
    """Return the site-local Lab entry URL for a generated notebook path."""

    notebook = notebook_path
    if notebook.startswith("site/"):
        notebook = notebook[len("site/") :]
    return "lab/index.html?path=" + notebook


def notebook_url_for(notebook_path: str) -> str:
    """Return the site-local notebook file URL for a generated notebook path."""

    if notebook_path.startswith("site/"):
        return notebook_path[len("site/") :]
    return notebook_path


def read_runtime_manifest() -> dict[str, object]:
    if not RUNTIME_MANIFEST.is_file():
        raise SystemExit("site/runtime-manifest.json missing; run scripts/detect_runtimes.py first")
    return json.loads(RUNTIME_MANIFEST.read_text(encoding="utf-8"))


def choose_doc(lesson_path: str) -> tuple[Path, str]:
    lesson = ROOT / lesson_path
    zh = lesson / "docs" / "zh-CN.md"
    if zh.is_file():
        return zh, "zh-CN"
    en = lesson / "docs" / "en.md"
    if en.is_file():
        return en, "en"
    raise FileNotFoundError(f"no docs found for {lesson_path}")


def markdown_cell(lines: list[str]) -> dict[str, object] | None:
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        return None
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in lines],
    }


def code_cell(lines: list[str]) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in lines],
    }


def markdown_to_cells(text: str) -> list[dict[str, object]]:
    cells: list[dict[str, object]] = []
    md_buffer: list[str] = []
    code_buffer: list[str] = []
    code_lang = ""
    in_fence = False

    def flush_markdown() -> None:
        cell = markdown_cell(md_buffer)
        if cell:
            cells.append(cell)
        md_buffer.clear()

    for line in text.splitlines():
        match = FENCE_RE.match(line)
        if match:
            if not in_fence:
                in_fence = True
                code_lang = match.group(1).strip() or "text"
                code_buffer = []
                if code_lang == "python":
                    flush_markdown()
                else:
                    md_buffer.append("```" + code_lang)
                continue
            in_fence = False
            if code_lang == "python":
                cells.append(code_cell(code_buffer))
            else:
                md_buffer.extend(code_buffer)
                md_buffer.append("```")
            code_buffer = []
            code_lang = ""
            continue

        if in_fence:
            code_buffer.append(line)
        else:
            md_buffer.append(line)

    if in_fence:
        if code_lang == "python":
            cells.append(code_cell(code_buffer))
        else:
            md_buffer.append("```" + code_lang)
            md_buffer.extend(code_buffer)
    flush_markdown()
    return cells


def notebook_for(lesson_path: str, record: dict[str, object]) -> tuple[dict[str, object], str]:
    doc, lang = choose_doc(lesson_path)
    text = doc.read_text(encoding="utf-8")
    cells = markdown_to_cells(text)
    notebook = {
        "cells": cells,
        "metadata": {
            "aifs": {
                "lesson_path": lesson_path,
                "lang": lang,
                "runtime": record.get("runtime"),
                "packages": record.get("packages", []),
                "source_doc": rel(doc),
            },
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "pygments_lexer": "ipython3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return notebook, lang


def candidate_lessons(manifest: dict[str, object], include_all: bool, limit: int) -> list[tuple[str, dict[str, object]]]:
    lessons = manifest.get("lessons")
    if not isinstance(lessons, dict):
        raise SystemExit("runtime manifest lessons must be an object")
    candidates: list[tuple[str, dict[str, object]]] = []
    for lesson_path, raw_record in sorted(lessons.items()):
        if not isinstance(raw_record, dict):
            continue
        if raw_record.get("runtime") != "browser-pyodide":
            continue
        if not (ROOT / lesson_path / "docs" / "en.md").is_file():
            continue
        candidates.append((lesson_path, raw_record))
    return candidates if include_all else candidates[:limit]


def write_notebooks(include_all: bool, limit: int) -> list[str]:
    manifest = read_runtime_manifest()
    written: list[str] = []
    lessons = manifest.get("lessons")
    assert isinstance(lessons, dict)

    for lesson_path, record in candidate_lessons(manifest, include_all, limit):
        notebook, _lang = notebook_for(lesson_path, record)
        out = NOTEBOOK_ROOT / lesson_path / "lesson.ipynb"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        notebook_rel = rel(out)
        record["notebook"] = notebook_url_for(notebook_rel)
        record["lab"] = lab_url_for(notebook_rel)
        written.append(notebook_rel)

    RUNTIME_MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return written


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=20, help="number of notebooks to generate unless --all is set")
    parser.add_argument("--all", action="store_true", help="generate all browser-pyodide notebooks")
    args = parser.parse_args(argv)
    if args.limit <= 0:
        raise SystemExit("--limit must be positive")

    written = write_notebooks(args.all, args.limit)
    print(f"build_notebooks.py — wrote {len(written)} notebook(s)")
    for path in written[:10]:
        print(f"  {path}")
    if len(written) > 10:
        print(f"  ... {len(written) - 10} more")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
