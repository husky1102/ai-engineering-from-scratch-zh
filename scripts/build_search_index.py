#!/usr/bin/env python3
"""Build the Chinese-first lesson search index.

Usage:
    python3 scripts/build_search_index.py

The index prefers docs/zh-CN.md, falls back to docs/en.md, and includes runtime
labels from site/runtime-manifest.json when present.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent.parent
PHASES_DIR = ROOT / "phases"
DEFAULT_OUT = ROOT / "site" / "search-index.zh-CN.json"
RUNTIME_MANIFEST = ROOT / "site" / "runtime-manifest.json"

HEADING_RE = re.compile(r"^(#{1,3})\s+(.+?)\s*$")
QUOTE_RE = re.compile(r"^>\s*(.+?)\s*$")
META_RE = re.compile(r"^\*\*[^*]+\*\*:\s*")
FENCE_RE = re.compile(r"^```")


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def iter_lesson_dirs() -> Iterable[Path]:
    for phase in sorted(PHASES_DIR.iterdir()):
        if not phase.is_dir() or not phase.name[:2].isdigit():
            continue
        for lesson in sorted(phase.iterdir()):
            if lesson.is_dir() and lesson.name[:2].isdigit():
                yield lesson


def choose_doc(lesson: Path) -> tuple[Path, str] | tuple[None, None]:
    zh = lesson / "docs" / "zh-CN.md"
    if zh.is_file():
        return zh, "zh-CN"
    en = lesson / "docs" / "en.md"
    if en.is_file():
        return en, "en"
    return None, None


def strip_inline(text: str) -> str:
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"[*_]+", "", text)
    return text.strip()


def extract_doc_fields(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8")
    title = ""
    summary = ""
    headings: list[str] = []
    key_terms: list[str] = []
    in_fence = False
    in_key_terms = False

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue

        heading = HEADING_RE.match(line)
        if heading:
            level = len(heading.group(1))
            value = strip_inline(heading.group(2))
            if level == 1 and not title:
                title = value
            elif level in (2, 3):
                headings.append(value)
            in_key_terms = value.lower() in {"key terms", "关键术语"}
            continue

        if not summary:
            quote = QUOTE_RE.match(line)
            if quote:
                summary = strip_inline(quote.group(1))
                continue
            stripped = line.strip()
            if stripped and not stripped.startswith("|") and not META_RE.match(stripped):
                summary = strip_inline(stripped)

        if in_key_terms and "|" in line and not re.match(r"^\s*\|?\s*-+", line):
            cells = [strip_inline(c) for c in line.strip("|").split("|")]
            for cell in cells:
                if cell and cell.lower() not in {"term", "meaning", "术语", "含义"}:
                    key_terms.append(cell)

    return {
        "title": title or path.parent.parent.name,
        "summary": summary,
        "headings": headings,
        "key_terms": key_terms[:20],
    }


def load_runtime() -> dict[str, dict[str, object]]:
    if not RUNTIME_MANIFEST.is_file():
        return {}
    data = json.loads(RUNTIME_MANIFEST.read_text(encoding="utf-8"))
    lessons = data.get("lessons", {})
    return lessons if isinstance(lessons, dict) else {}


def build_index() -> dict[str, object]:
    runtime = load_runtime()
    lessons: list[dict[str, object]] = []
    for lesson in iter_lesson_dirs():
        doc, lang = choose_doc(lesson)
        if doc is None or lang is None:
            continue
        lesson_path = rel(lesson)
        fields = extract_doc_fields(doc)
        phase = lesson.parent
        phase_num = int(phase.name.split("-", 1)[0])
        record = runtime.get(lesson_path, {})
        lessons.append(
            {
                "path": lesson_path,
                "lang": lang,
                "phase": phase_num,
                "phase_slug": phase.name,
                "lesson_slug": lesson.name,
                "title": fields["title"],
                "summary": fields["summary"],
                "headings": fields["headings"],
                "key_terms": fields["key_terms"],
                "runtime": record.get("runtime"),
                "language": record.get("language"),
                "packages": record.get("packages", []),
                "notebook": record.get("notebook"),
                "lab": record.get("lab"),
            }
        )
    return {"schema_version": 1, "locale": "zh-CN", "lessons": lessons}


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output JSON path")
    args = parser.parse_args(argv)
    out = args.out if args.out.is_absolute() else ROOT / args.out
    index = build_index()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(index, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    zh_count = sum(1 for row in index["lessons"] if row["lang"] == "zh-CN")
    print(f"build_search_index.py — {len(index['lessons'])} lesson(s), zh-CN={zh_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
