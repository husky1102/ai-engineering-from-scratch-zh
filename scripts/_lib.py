"""Shared helpers for scripts/ tools.

No external dependencies. Python 3.10+ (PEP 604 unions in type hints).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent.parent
PHASES_DIR = ROOT / "phases"

PHASE_DIR_RE = re.compile(r"^([0-9]{2})-([a-z0-9][a-z0-9-]*)$")
LESSON_DIR_RE = re.compile(r"^([0-9]{2})-([a-z0-9][a-z0-9-]*)$")

CODE_SUFFIXES = {
    ".py": "python",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "javascript",
    ".mjs": "javascript",
    ".rs": "rust",
    ".jl": "julia",
    ".sh": "shell",
    ".yml": "yaml",
    ".yaml": "yaml",
    ".json": "json",
    ".go": "go",
    ".swift": "swift",
    ".ipynb": "jupyter",
}

MAIN_LANGUAGE_BY_SUFFIX = {
    ".py": "Python",
    ".ts": "TypeScript",
    ".rs": "Rust",
    ".jl": "Julia",
}


def rel_path(path: Path, root: Path = ROOT) -> str:
    """Return a POSIX path relative to the repository root."""
    return path.relative_to(root).as_posix()


def iter_phase_dirs(phases_dir: Path = PHASES_DIR) -> Iterable[Path]:
    """Yield valid phase directories in stable order."""
    if not phases_dir.is_dir():
        return
    for path in sorted(phases_dir.iterdir()):
        if path.is_dir() and PHASE_DIR_RE.match(path.name):
            yield path


def iter_lesson_dirs(
    phase_filter: int | None = None,
    phases_dir: Path = PHASES_DIR,
) -> Iterable[Path]:
    """Yield valid lesson directories in stable order, optionally for one phase."""
    for phase in iter_phase_dirs(phases_dir):
        if phase_filter is not None:
            try:
                phase_num = int(phase.name.split("-", 1)[0])
            except ValueError:
                continue
            if phase_num != phase_filter:
                continue
        for lesson in sorted(phase.iterdir()):
            if lesson.is_dir() and LESSON_DIR_RE.match(lesson.name):
                yield lesson


def main_languages(lesson: Path) -> set[str]:
    """Return AGENTS.md language names represented by code/main.* files."""
    code_dir = lesson / "code"
    if not code_dir.is_dir():
        return set()
    return {
        MAIN_LANGUAGE_BY_SUFFIX[path.suffix]
        for path in code_dir.glob("main.*")
        if path.suffix in MAIN_LANGUAGE_BY_SUFFIX
    }


def parse_frontmatter(text: str) -> dict[str, object] | None:
    """Parse a YAML-subset frontmatter block at the top of a markdown string.

    Returns the parsed key/value mapping, or None when no frontmatter is present
    or the closing `---` is missing.

    Supports:
    - bare strings: `key: value`
    - single-quoted: `key: 'value'`
    - double-quoted: `key: "value"`
    - lists: `key: [a, b, "c"]`
    - inline comment lines beginning with `#`
    """
    if not text.startswith("---\n"):
        return None
    # Closing delimiter: "\n---\n" inside the file, or "\n---" at EOF.
    end = text.find("\n---\n", 4)
    if end == -1 and text.endswith("\n---"):
        end = len(text) - 4
    if end == -1:
        return None
    block = text[4:end].strip("\n")
    result: dict[str, object] = {}
    for raw in block.splitlines():
        # Anchor at column 0: skip comments + indented lines.
        if not raw or raw.startswith("#") or raw[0] in (" ", "\t"):
            continue
        if ":" not in raw:
            continue
        key, _, value = raw.partition(":")
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if value.startswith("[") and value.endswith("]"):
            inner = value[1:-1].strip()
            result[key] = (
                [item.strip().strip("'\"") for item in inner.split(",") if item.strip()]
                if inner
                else []
            )
        elif (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            result[key] = value[1:-1]
        else:
            result[key] = value
    return result
