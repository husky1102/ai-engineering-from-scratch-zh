#!/usr/bin/env python3
"""Build local code/output asset metadata for lesson pages.

Usage:
    python3 scripts/build_lesson_assets.py
    python3 scripts/build_lesson_assets.py --stdout

The static lesson page cannot list local directories at runtime. This manifest
keeps the "what this lesson ships" and code panels local-first without using
the GitHub Contents API.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import CODE_SUFFIXES, iter_lesson_dirs, rel_path  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "site" / "lesson-assets.json"

COMMAND_BY_SUFFIX = {
    ".py": "python",
    ".ts": "npx tsx",
    ".js": "node",
    ".mjs": "node",
    ".rs": "rustc --edition 2021",
    ".jl": "julia",
    ".sh": "bash",
}


def rel(path: Path) -> str:
    return rel_path(path, ROOT)


def local_url(path: Path) -> str:
    return "content/" + rel(path)


def language_for(path: Path) -> str:
    if path.name == "Dockerfile":
        return "docker"
    return CODE_SUFFIXES.get(path.suffix, "text")


def command_for(path: Path) -> str:
    base = COMMAND_BY_SUFFIX.get(path.suffix)
    if not base:
        return ""
    return f"{base} {path.name}"


def output_kind(path: Path) -> str:
    lower = path.name.lower()
    if "prompt" in lower:
        return "prompt"
    if "skill" in lower:
        return "skill"
    return "artifact"


def frontmatter_description(text: str) -> str:
    match = re.search(r"^---\s*\n(.*?)\n---\s*$", text, flags=re.DOTALL | re.MULTILINE)
    if match:
        for line in match.group(1).splitlines():
            if line.strip().startswith("description:"):
                value = line.split(":", 1)[1].strip()
                return value.strip("\"'")
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith(("#", "---")):
            return stripped[:160]
    return ""


def file_size(path: Path) -> int:
    return path.stat().st_size


def code_record(path: Path, lesson: Path) -> dict[str, object]:
    return {
        "name": path.name,
        "path": rel(path),
        "url": local_url(path),
        "size": file_size(path),
        "language": language_for(path),
        "command": command_for(path),
        "is_entry_candidate": path.name in {"main.py", "main.ts", "main.js", "main.rs", "main.jl", "verify.py"},
    }


def output_record(path: Path) -> dict[str, object]:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = ""
    return {
        "name": path.name,
        "path": rel(path),
        "url": local_url(path),
        "size": file_size(path),
        "kind": output_kind(path),
        "description": frontmatter_description(text) or path.name,
    }


def lesson_record(lesson: Path) -> dict[str, object]:
    code_dir = lesson / "code"
    output_dir = lesson / "outputs"
    code_files = sorted(
        path
        for path in code_dir.rglob("*")
        if path.is_file()
        and not path.name.startswith(".")
        and (not path.relative_to(code_dir).parts or path.relative_to(code_dir).parts[0] != "tests")
    ) if code_dir.is_dir() else []
    output_files = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file() and not path.name.startswith(".")
    ) if output_dir.is_dir() else []
    return {
        "code": [code_record(path, lesson) for path in code_files],
        "outputs": [output_record(path) for path in output_files],
    }


def build_manifest() -> dict[str, object]:
    lessons = {rel(lesson): lesson_record(lesson) for lesson in iter_lesson_dirs()}
    totals = {
        "lessons": len(lessons),
        "code_files": sum(len(record["code"]) for record in lessons.values()),
        "output_files": sum(len(record["outputs"]) for record in lessons.values()),
    }
    return {
        "schema_version": 1,
        "generated_by": "scripts/build_lesson_assets.py",
        "totals": totals,
        "lessons": lessons,
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output JSON path")
    parser.add_argument("--stdout", action="store_true", help="print JSON instead of writing")
    args = parser.parse_args(argv)

    manifest = build_manifest()
    payload = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"

    if args.stdout:
        sys.stdout.write(payload)
    else:
        out = args.out if args.out.is_absolute() else ROOT / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(payload, encoding="utf-8")

    totals = manifest["totals"]
    print(
        "build_lesson_assets.py — "
        f"{totals['lessons']} lesson(s), "
        f"{totals['code_files']} code file(s), "
        f"{totals['output_files']} output file(s)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
