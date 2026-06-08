#!/usr/bin/env python3
"""Validate zh-CN lesson files against their English sources.

Usage:
    python3 scripts/i18n_validate.py
    python3 scripts/i18n_validate.py --lesson phases/00-setup-and-tooling/01-dev-environment
    python3 scripts/i18n_validate.py --json

Missing Chinese files are reported as coverage gaps and do not fail validation.
Existing Chinese docs and quizzes must preserve Markdown code-fence structure
and quiz schema invariants.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent.parent
PHASES_DIR = ROOT / "phases"
FENCE_RE = re.compile(r"^```([^\s`]*)\s*$")


@dataclass
class Issue:
    rule: str
    file: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"rule": self.rule, "file": self.file, "message": self.message}


@dataclass
class Report:
    docs_total: int = 0
    docs_translated: int = 0
    quiz_total: int = 0
    quiz_translated: int = 0
    issues: list[Issue] = field(default_factory=list)

    def add(self, rule: str, path: Path, message: str) -> None:
        self.issues.append(Issue(rule, rel(path), message))


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def iter_lesson_dirs() -> Iterable[Path]:
    if not PHASES_DIR.is_dir():
        return
    for phase in sorted(PHASES_DIR.iterdir()):
        if not phase.is_dir() or not phase.name[:2].isdigit():
            continue
        for lesson in sorted(phase.iterdir()):
            if lesson.is_dir() and lesson.name[:2].isdigit():
                yield lesson


def resolve_lesson(path: str) -> Path:
    lesson = Path(path)
    if not lesson.is_absolute():
        lesson = ROOT / lesson
    resolved = lesson.resolve()
    try:
        resolved.relative_to(ROOT)
    except ValueError as exc:
        raise SystemExit(f"lesson path escapes repository root: {path}") from exc
    if not resolved.is_dir():
        raise SystemExit(f"lesson path is not a directory: {path}")
    return resolved


def fence_langs(text: str) -> list[str]:
    """Return opening-fence language tags in order.

    Closing fences are not counted. If an English source fence has no language
    tag, zh-CN files may use `text` to satisfy the repository rule that newly
    added code fences are tagged.
    """

    langs: list[str] = []
    in_fence = False
    for line in text.splitlines():
        match = FENCE_RE.match(line)
        if not match:
            continue
        if in_fence:
            in_fence = False
            continue
        langs.append(match.group(1).strip())
        in_fence = True
    return langs


def expected_target_lang(source_lang: str) -> str:
    return source_lang or "text"


def validate_docs(report: Report, source: Path, target: Path) -> None:
    report.docs_total += 1
    if not target.is_file():
        return
    report.docs_translated += 1
    try:
        source_text = source.read_text(encoding="utf-8")
        target_text = target.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        report.add("I001", target, f"Markdown is not valid UTF-8: {exc}")
        return

    source_fences = fence_langs(source_text)
    target_fences = fence_langs(target_text)
    if len(source_fences) != len(target_fences):
        report.add(
            "I010",
            target,
            f"code fence count differs from English source ({len(target_fences)} != {len(source_fences)})",
        )
        return
    for idx, (src_lang, dst_lang) in enumerate(zip(source_fences, target_fences), 1):
        expected_lang = expected_target_lang(src_lang)
        if expected_lang != dst_lang:
            report.add(
                "I011",
                target,
                f"code fence {idx} language differs from expected tag ({dst_lang!r} != {expected_lang!r})",
            )
        if not dst_lang:
            report.add("I012", target, f"code fence {idx} has no language tag")


def load_quiz(path: Path) -> tuple[list[dict[str, object]] | None, str | None]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return None, str(exc)
    if isinstance(data, dict) and isinstance(data.get("questions"), list):
        return data["questions"], None
    if isinstance(data, list):
        return data, None
    return None, "quiz must be a list or an object with questions[]"


def validate_quiz(report: Report, source: Path, target: Path) -> None:
    report.quiz_total += 1
    if not target.is_file():
        return
    report.quiz_translated += 1
    source_questions, source_error = load_quiz(source)
    target_questions, target_error = load_quiz(target)
    if source_error or source_questions is None:
        report.add("I020", source, f"English quiz is invalid: {source_error}")
        return
    if target_error or target_questions is None:
        report.add("I021", target, f"Chinese quiz is invalid: {target_error}")
        return

    if len(source_questions) != len(target_questions):
        report.add(
            "I022",
            target,
            f"question count differs from English source ({len(target_questions)} != {len(source_questions)})",
        )
        return

    for idx, (src, dst) in enumerate(zip(source_questions, target_questions)):
        if not isinstance(src, dict) or not isinstance(dst, dict):
            report.add("I023", target, f"question[{idx}] must be an object")
            continue
        for key in ("stage", "correct"):
            if src.get(key) != dst.get(key):
                report.add(
                    "I024",
                    target,
                    f"question[{idx}].{key} differs from English source ({dst.get(key)!r} != {src.get(key)!r})",
                )
        src_options = src.get("options")
        dst_options = dst.get("options")
        if not isinstance(src_options, list) or not isinstance(dst_options, list):
            report.add("I025", target, f"question[{idx}].options must be a list")
            continue
        if len(src_options) != len(dst_options):
            report.add(
                "I026",
                target,
                f"question[{idx}] options length differs from English source ({len(dst_options)} != {len(src_options)})",
            )
        correct = dst.get("correct")
        if not isinstance(correct, int) or not (0 <= correct < len(dst_options)):
            report.add("I027", target, f"question[{idx}].correct is not a valid option index")


def validate_lesson(report: Report, lesson: Path) -> None:
    source_doc = lesson / "docs" / "en.md"
    if source_doc.is_file():
        validate_docs(report, source_doc, lesson / "docs" / "zh-CN.md")
    source_quiz = lesson / "quiz.json"
    if source_quiz.is_file():
        validate_quiz(report, source_quiz, lesson / "quiz.zh-CN.json")


def build_report(lessons: list[Path] | None = None) -> Report:
    report = Report()
    for lesson in lessons if lessons is not None else iter_lesson_dirs():
        validate_lesson(report, lesson)
    return report


def render_report(report: Report) -> str:
    lines = [
        "i18n_validate.py — "
        f"docs {report.docs_translated}/{report.docs_total}, "
        f"quiz {report.quiz_translated}/{report.quiz_total}, "
        f"{len(report.issues)} issue(s)",
    ]
    if report.issues:
        lines.append("")
        for issue in report.issues:
            lines.append(f"  [{issue.rule}] {issue.file}: {issue.message}")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lesson",
        action="append",
        default=[],
        help="restrict validation to a lesson directory; may be repeated",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON report")
    args = parser.parse_args(argv)

    lessons = [resolve_lesson(path) for path in args.lesson] if args.lesson else None
    report = build_report(lessons)
    if args.json:
        json.dump(
            {
                "docs_total": report.docs_total,
                "docs_translated": report.docs_translated,
                "quiz_total": report.quiz_total,
                "quiz_translated": report.quiz_translated,
                "issues": [issue.to_dict() for issue in report.issues],
            },
            sys.stdout,
            ensure_ascii=False,
            indent=2,
        )
        sys.stdout.write("\n")
    else:
        print(render_report(report))
    return 1 if report.issues else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
