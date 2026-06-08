#!/usr/bin/env python3
"""Build the Chinese localization manifest for lesson docs and quizzes.

Usage:
    python3 scripts/i18n_inventory.py
    python3 scripts/i18n_inventory.py --stdout
    python3 scripts/i18n_inventory.py --out i18n/manifest.jsonl

The manifest is JSON Lines so it can be diffed, filtered, and resumed without
loading the whole curriculum into memory. Missing zh-CN files are pending, not
errors. Existing targets are marked stale only when a previous manifest recorded
a different source hash for the same target path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent.parent
PHASES_DIR = ROOT / "phases"
DEFAULT_OUT = ROOT / "i18n" / "manifest.jsonl"


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def iter_lesson_dirs() -> Iterable[Path]:
    if not PHASES_DIR.is_dir():
        return
    for phase in sorted(PHASES_DIR.iterdir()):
        if not phase.is_dir() or not phase.name[:2].isdigit():
            continue
        for lesson in sorted(phase.iterdir()):
            if lesson.is_dir() and lesson.name[:2].isdigit():
                yield lesson


def load_previous(path: Path) -> dict[str, dict[str, object]]:
    previous: dict[str, dict[str, object]] = {}
    if not path.is_file():
        return previous
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{rel(path)}:{line_no}: invalid JSON: {exc}") from exc
        target = row.get("target_path")
        if isinstance(target, str):
            previous[target] = row
    return previous


def record_for(
    *,
    lesson: Path,
    kind: str,
    source: Path,
    target: Path,
    previous: dict[str, dict[str, object]],
) -> dict[str, object]:
    source_hash = sha256_file(source)
    target_rel = rel(target)
    target_exists = target.is_file()
    prior = previous.get(target_rel, {})
    if not target_exists:
        status = "pending"
    elif prior.get("source_hash") and prior.get("source_hash") != source_hash:
        status = "stale"
    else:
        status = "current"

    return {
        "lesson_path": rel(lesson),
        "kind": kind,
        "source_path": rel(source),
        "target_path": target_rel,
        "source_hash": source_hash,
        "target_hash": sha256_file(target) if target_exists else None,
        "status": status,
    }


def build_manifest(previous_path: Path) -> list[dict[str, object]]:
    previous = load_previous(previous_path)
    rows: list[dict[str, object]] = []
    for lesson in iter_lesson_dirs():
        doc = lesson / "docs" / "en.md"
        if doc.is_file():
            rows.append(
                record_for(
                    lesson=lesson,
                    kind="docs",
                    source=doc,
                    target=lesson / "docs" / "zh-CN.md",
                    previous=previous,
                )
            )
        quiz = lesson / "quiz.json"
        if quiz.is_file():
            rows.append(
                record_for(
                    lesson=lesson,
                    kind="quiz",
                    source=quiz,
                    target=lesson / "quiz.zh-CN.json",
                    previous=previous,
                )
            )
    return rows


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="manifest path")
    parser.add_argument("--stdout", action="store_true", help="print JSONL instead of writing")
    args = parser.parse_args(argv)

    out_path = args.out if args.out.is_absolute() else ROOT / args.out
    rows = build_manifest(out_path)
    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)

    if args.stdout:
        sys.stdout.write(payload)
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload, encoding="utf-8")

    counts: dict[str, int] = {}
    for row in rows:
        key = f"{row['kind']}:{row['status']}"
        counts[key] = counts.get(key, 0) + 1
    summary = ", ".join(f"{k}={counts[k]}" for k in sorted(counts))
    print(f"i18n_inventory.py — {len(rows)} item(s); {summary}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
