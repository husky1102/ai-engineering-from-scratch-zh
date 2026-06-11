#!/usr/bin/env python3
"""Invariant checks across every lesson directory.

Usage:
    python scripts/audit_lessons.py [--phase N] [--json] [--strict]

Exit codes:
    0 — no blocking issues
    1 — blocking issues found, or advisory warnings found with --strict
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import iter_lesson_dirs, main_languages, rel_path  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

LESSON_DIR_RE = re.compile(r"^[0-9]{2}-[a-z0-9][a-z0-9-]*[a-z0-9]$")
MD_LINK_RE = re.compile(r"\[[^\]]*\]\(([^)\s#]+)(?:#[^)]*)?\)")
H1_RE = re.compile(r"^#\s+\S", re.MULTILINE)

CANONICAL_QUIZ_KEYS = {"stage", "question", "options", "correct", "explanation"}
LEGACY_QUIZ_KEYS = {"q", "choices", "answer"}
CODE_IGNORED_NAMES = {"README.md", "AGENTS.md", ".gitkeep", ".DS_Store"}
MIN_DOC_BYTES = 200
MAX_OPTIONS = 6
MIN_OPTIONS = 2
DEFAULT_REPORT_LIMIT = 25
REQUIRED_QUIZ_STAGE_COUNTS = {"pre": 1, "check": 3, "post": 2}
REQUIRED_QUIZ_QUESTION_COUNT = 6
MIN_TEST_CASES = 5
LANGUAGE_ALIASES = {
    "python": "Python",
    "typescript": "TypeScript",
    "rust": "Rust",
    "julia": "Julia",
}
LANGUAGE_LINE_RE = re.compile(r"^\*\*Languages:\*\*\s*(.+)$", re.MULTILINE)
LEARNING_OBJECTIVES_RE = re.compile(r"^##\s+Learning Objectives\s*$", re.MULTILINE)
JS_TEST_RE = re.compile(r"\b(?:test|it)\s*\(")


@dataclass
class Issue:
    rule: str
    lesson: str
    file: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {
            "rule": self.rule,
            "lesson": self.lesson,
            "file": self.file,
            "message": self.message,
        }


@dataclass
class Audit:
    lessons_checked: int = 0
    issues: list[Issue] = field(default_factory=list)
    warnings: list[Issue] = field(default_factory=list)

    def add(self, rule: str, lesson: Path, file: Path | None, message: str) -> None:
        rel_lesson = rel_path(lesson, ROOT)
        rel_file = rel_path(file, ROOT) if file else rel_lesson
        self.issues.append(Issue(rule, rel_lesson, rel_file, message))

    def warn(self, rule: str, lesson: Path, file: Path | None, message: str) -> None:
        rel_lesson = rel_path(lesson, ROOT)
        rel_file = rel_path(file, ROOT) if file else rel_lesson
        self.warnings.append(Issue(rule, rel_lesson, rel_file, message))


def check_lesson_dir_pattern(audit: Audit, lesson: Path) -> bool:
    if not LESSON_DIR_RE.match(lesson.name):
        audit.add(
            "L001",
            lesson,
            None,
            f"lesson dir name does not match NN-slug pattern: {lesson.name!r}",
        )
        return False
    return True


def check_docs_en_md(audit: Audit, lesson: Path) -> str | None:
    doc = lesson / "docs" / "en.md"
    if not doc.is_file():
        audit.add("L002", lesson, doc, "missing docs/en.md")
        return None
    try:
        text = doc.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        audit.add("L002", lesson, doc, "docs/en.md is not valid UTF-8")
        return None
    if len(text.encode("utf-8")) < MIN_DOC_BYTES:
        audit.add(
            "L003",
            lesson,
            doc,
            f"docs/en.md shorter than {MIN_DOC_BYTES} bytes (got {len(text)})",
        )
    if not H1_RE.search(text):
        audit.add("L004", lesson, doc, "docs/en.md missing top-level H1")
    return text


def declared_languages(text: str) -> set[str] | None:
    match = LANGUAGE_LINE_RE.search(text)
    if not match:
        return None
    value = match.group(1)
    found = set()
    for raw, canonical in LANGUAGE_ALIASES.items():
        if re.search(rf"\b{re.escape(raw)}\b", value, re.IGNORECASE):
            found.add(canonical)
    return found


def check_frontmatter_contract(audit: Audit, lesson: Path, text: str) -> None:
    doc = lesson / "docs" / "en.md"
    languages = declared_languages(text)
    if languages is None:
        audit.warn("A001", lesson, doc, "docs/en.md missing **Languages:** field")
    else:
        expected = main_languages(lesson)
        if languages != expected:
            audit.warn(
                "A002",
                lesson,
                doc,
                "**Languages:** must match code/main.* languages "
                f"(declared {sorted(languages)}, expected {sorted(expected)})",
            )
    if not LEARNING_OBJECTIVES_RE.search(text):
        audit.warn("A003", lesson, doc, "docs/en.md missing ## Learning Objectives")


def check_code_main(audit: Audit, lesson: Path) -> None:
    code_dir = lesson / "code"
    if not code_dir.is_dir():
        return
    for path in code_dir.rglob("*"):
        if path.is_file() and path.name not in CODE_IGNORED_NAMES:
            return
    audit.add("L005", lesson, code_dir, "code/ is empty (no source or config files)")


def check_quiz(audit: Audit, lesson: Path) -> None:
    quiz = lesson / "quiz.json"
    if not quiz.is_file():
        return
    try:
        raw = quiz.read_text(encoding="utf-8")
        data = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        audit.add("L006", lesson, quiz, f"quiz.json not valid JSON: {exc}")
        return
    if isinstance(data, list):
        questions = data
    elif isinstance(data, dict):
        questions = data.get("questions")
    else:
        questions = None
    if not isinstance(questions, list) or not questions:
        audit.add(
            "L006",
            lesson,
            quiz,
            "quiz.json must be a non-empty array or a dict with non-empty questions[]",
        )
        return
    for idx, q in enumerate(questions):
        if not isinstance(q, dict):
            audit.add("L006", lesson, quiz, f"question[{idx}] is not an object")
            continue
        legacy = LEGACY_QUIZ_KEYS & q.keys()
        if legacy:
            audit.add(
                "L007",
                lesson,
                quiz,
                f"question[{idx}] uses legacy schema keys {sorted(legacy)} "
                f"(canonical: {sorted(CANONICAL_QUIZ_KEYS)})",
            )
            continue
        missing = CANONICAL_QUIZ_KEYS - q.keys()
        if missing:
            audit.add(
                "L006",
                lesson,
                quiz,
                f"question[{idx}] missing keys {sorted(missing)}",
            )
            continue
        options = q.get("options")
        if not isinstance(options, list) or not (MIN_OPTIONS <= len(options) <= MAX_OPTIONS):
            audit.add(
                "L008",
                lesson,
                quiz,
                f"question[{idx}] options length must be {MIN_OPTIONS}..{MAX_OPTIONS} "
                f"(got {len(options) if isinstance(options, list) else type(options).__name__})",
            )
            continue
        correct = q.get("correct")
        if not isinstance(correct, int) or not (0 <= correct < len(options)):
            audit.add(
                "L009",
                lesson,
                quiz,
                f"question[{idx}] correct={correct!r} not a valid index in options[0..{len(options) - 1}]",
            )


def load_quiz(audit: Audit, lesson: Path) -> object | None:
    quiz = lesson / "quiz.json"
    if not quiz.is_file():
        audit.warn("A004", lesson, quiz, "missing quiz.json")
        return None
    try:
        return json.loads(quiz.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None


def check_quiz_contract(audit: Audit, lesson: Path) -> None:
    quiz = lesson / "quiz.json"
    data = load_quiz(audit, lesson)
    if data is None:
        return
    if isinstance(data, list):
        questions = data
        audit.warn(
            "A005",
            lesson,
            quiz,
            "quiz.json should be an object with lesson, title, and questions[]",
        )
    elif isinstance(data, dict):
        questions = data.get("questions")
        missing = [key for key in ("lesson", "title", "questions") if key not in data]
        if missing:
            audit.warn(
                "A005",
                lesson,
                quiz,
                f"quiz.json missing top-level key(s) {missing}",
            )
    else:
        return
    if not isinstance(questions, list):
        return
    if len(questions) != REQUIRED_QUIZ_QUESTION_COUNT:
        audit.warn(
            "A006",
            lesson,
            quiz,
            f"quiz.json should contain exactly {REQUIRED_QUIZ_QUESTION_COUNT} questions "
            f"(got {len(questions)})",
        )
    stage_counts: dict[str, int] = {}
    for question in questions:
        if isinstance(question, dict):
            stage = question.get("stage")
            if isinstance(stage, str):
                stage_counts[stage] = stage_counts.get(stage, 0) + 1
    if stage_counts != REQUIRED_QUIZ_STAGE_COUNTS:
        audit.warn(
            "A007",
            lesson,
            quiz,
            "quiz stage distribution should be 1 pre, 3 check, 2 post "
            f"(got {stage_counts})",
        )


def count_python_tests(path: Path) -> int:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return 0
    return sum(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test")
        for node in ast.walk(tree)
    )


def count_test_cases(path: Path) -> int:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return 0
    if path.suffix == ".py":
        return count_python_tests(path)
    if path.suffix in {".js", ".mjs", ".cjs", ".ts", ".tsx"}:
        return len(JS_TEST_RE.findall(text))
    if path.suffix == ".rs":
        return text.count("#[test]")
    if path.suffix == ".jl":
        return text.count("@test")
    return 0


def check_tests_contract(audit: Audit, lesson: Path) -> None:
    if not main_languages(lesson):
        return
    tests_dir = lesson / "code" / "tests"
    if not tests_dir.is_dir():
        audit.warn(
            "A008",
            lesson,
            tests_dir,
            f"code/tests/ missing for lesson with code/main.*; expected {MIN_TEST_CASES}+ tests",
        )
        return
    test_count = sum(count_test_cases(path) for path in tests_dir.rglob("*") if path.is_file())
    if test_count < MIN_TEST_CASES:
        audit.warn(
            "A008",
            lesson,
            tests_dir,
            f"code/tests/ should contain {MIN_TEST_CASES}+ unit tests (found {test_count})",
        )


def check_internal_links(audit: Audit, lesson: Path, text: str) -> None:
    doc = lesson / "docs" / "en.md"
    seen: set[str] = set()
    for match in MD_LINK_RE.finditer(text):
        href = match.group(1).strip()
        if href in seen:
            continue
        seen.add(href)
        if href.startswith(("http://", "https://", "mailto:", "data:")):
            continue
        if href.startswith("/"):
            target = ROOT / href.lstrip("/")
        else:
            target = (doc.parent / href).resolve()
        if not target.exists():
            audit.add("L010", lesson, doc, f"internal link does not resolve: {href!r}")


def audit_lesson(audit: Audit, lesson: Path) -> None:
    audit.lessons_checked += 1
    if not check_lesson_dir_pattern(audit, lesson):
        return
    text = check_docs_en_md(audit, lesson)
    check_code_main(audit, lesson)
    check_quiz(audit, lesson)
    if text is not None:
        check_frontmatter_contract(audit, lesson, text)
        check_internal_links(audit, lesson, text)
    check_quiz_contract(audit, lesson)
    check_tests_contract(audit, lesson)


def render_report(audit: Audit, limit: int = DEFAULT_REPORT_LIMIT) -> str:
    by_rule: dict[str, int] = {}
    for issue in audit.issues:
        by_rule[issue.rule] = by_rule.get(issue.rule, 0) + 1
    by_warning_rule: dict[str, int] = {}
    for warning in audit.warnings:
        by_warning_rule[warning.rule] = by_warning_rule.get(warning.rule, 0) + 1
    lines = [
        f"audit_lessons.py — {audit.lessons_checked} lesson(s) checked, "
        f"{len(audit.issues)} issue(s), {len(audit.warnings)} advisory warning(s)",
    ]
    if audit.issues:
        lines.append("")
        lines.append("Blocking issues:")
        for issue in audit.issues[:limit]:
            lines.append(f"  [{issue.rule}] {issue.file}: {issue.message}")
        if len(audit.issues) > limit:
            lines.append(f"  ... {len(audit.issues) - limit} more blocking issue(s); use --json for full detail")
        lines.append("")
        lines.append("Blocking summary by rule:")
        for rule in sorted(by_rule):
            lines.append(f"  {rule}: {by_rule[rule]}")
    if audit.warnings:
        lines.append("")
        lines.append("Advisory warnings:")
        for warning in audit.warnings[:limit]:
            lines.append(f"  [{warning.rule}] {warning.file}: {warning.message}")
        if len(audit.warnings) > limit:
            lines.append(f"  ... {len(audit.warnings) - limit} more advisory warning(s); use --json for full detail")
        lines.append("")
        lines.append("Advisory summary by rule:")
        for rule in sorted(by_warning_rule):
            lines.append(f"  {rule}: {by_warning_rule[rule]}")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", type=int, default=None, help="restrict to a single phase number")
    parser.add_argument("--json", action="store_true", help="emit JSON report on stdout")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="treat advisory contract warnings as errors",
    )
    args = parser.parse_args(argv)

    audit = Audit()
    for lesson in iter_lesson_dirs(args.phase):
        audit_lesson(audit, lesson)

    if args.json:
        json.dump(
            {
                "lessons_checked": audit.lessons_checked,
                "issues": [issue.to_dict() for issue in audit.issues],
                "warnings": [warning.to_dict() for warning in audit.warnings],
                "strict": args.strict,
            },
            sys.stdout,
            indent=2,
        )
        sys.stdout.write("\n")
    else:
        sys.stdout.write(render_report(audit) + "\n")

    return 1 if audit.issues or (args.strict and audit.warnings) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
