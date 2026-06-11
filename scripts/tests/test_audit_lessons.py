from __future__ import annotations

import io
import json
import re
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from scripts import audit_lessons


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def lesson_doc(languages: str | None, learning_objectives: bool) -> str:
    lines = [
        "# Fixture Lesson",
        "",
        "> A compact fixture that is long enough for the audit length check.",
        "",
        "**Type:** Build",
    ]
    if languages is not None:
        lines.append(f"**Languages:** {languages}")
    lines.extend(
        [
            "**Prerequisites:** None",
            "**Time:** ~5 minutes",
            "",
        ]
    )
    if learning_objectives:
        lines.extend(
            [
                "## Learning Objectives",
                "- Explain the fixture contract",
                "- Build a tiny example",
                "- Verify the result",
                "- Compare expected and actual behavior",
                "",
            ]
        )
    lines.extend(
        [
            "## The Problem",
            "",
            "This local fixture exists only to exercise audit behavior without relying on "
            "the repository's current debt counts. It keeps all mandatory blocking "
            "checks satisfied while varying advisory contract fields.",
            "",
            "## Build It",
            "",
            "The code path is intentionally tiny.",
        ]
    )
    return "\n".join(lines) + "\n"


class AuditLessonsTest(unittest.TestCase):
    def run_with_root(self, root: Path, argv: list[str]) -> tuple[int, str]:
        def fixture_lessons(phase_filter: int | None = None):
            phases_dir = root / "phases"
            if not phases_dir.is_dir():
                return
            for phase in sorted(phases_dir.iterdir()):
                if phase_filter is not None and int(phase.name.split("-", 1)[0]) != phase_filter:
                    continue
                for lesson in sorted(phase.iterdir()):
                    if lesson.is_dir():
                        yield lesson

        with mock.patch.object(audit_lessons, "ROOT", root), mock.patch.object(
            audit_lessons, "iter_lesson_dirs", fixture_lessons
        ):
            out = io.StringIO()
            with redirect_stdout(out):
                code = audit_lessons.main(argv)
        return code, out.getvalue()

    def test_advisory_contract_warnings_do_not_fail_default_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lesson = root / "phases" / "01-fixtures" / "01-warning-lesson"
            write(lesson / "docs" / "en.md", lesson_doc(None, False))
            write(lesson / "code" / "main.py", "print('fixture')\n")

            code, output = self.run_with_root(root, ["--phase", "1"])

        self.assertEqual(code, 0)
        self.assertIn("0 issue(s)", output)
        self.assertIn("advisory warning(s)", output)
        self.assertIn("[A001]", output)
        self.assertIn("[A003]", output)
        self.assertIn("[A004]", output)
        self.assertIn("[A008]", output)

    def test_strict_mode_fails_on_advisory_contract_warnings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lesson = root / "phases" / "01-fixtures" / "01-warning-lesson"
            write(lesson / "docs" / "en.md", lesson_doc(None, False))
            write(lesson / "code" / "main.py", "print('fixture')\n")

            code, _output = self.run_with_root(root, ["--phase", "1", "--strict"])

        self.assertEqual(code, 1)

    def test_clean_lesson_contract_has_no_warnings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lesson = root / "phases" / "01-fixtures" / "01-clean-lesson"
            write(lesson / "docs" / "en.md", lesson_doc("Python, TypeScript, Rust, Julia", True))
            write(lesson / "code" / "main.py", "print('fixture')\n")
            write(lesson / "code" / "main.ts", "console.log('fixture');\n")
            write(lesson / "code" / "main.rs", "fn main() { println!(\"fixture\"); }\n")
            write(lesson / "code" / "main.jl", 'println("fixture")\n')
            write(
                lesson / "code" / "tests" / "test_main.py",
                "\n".join(
                    [
                        "import unittest",
                        "",
                        "class FixtureTest(unittest.TestCase):",
                        "    def test_one(self): self.assertTrue(True)",
                        "    def test_two(self): self.assertTrue(True)",
                        "    def test_three(self): self.assertTrue(True)",
                        "    def test_four(self): self.assertTrue(True)",
                        "    def test_five(self): self.assertTrue(True)",
                        "",
                    ]
                ),
            )
            quiz = {
                "lesson": "01-clean-lesson",
                "title": "Fixture Lesson",
                "questions": [
                    {
                        "stage": stage,
                        "question": f"Question {idx}?",
                        "options": ["A", "B", "C", "D"],
                        "correct": 0,
                        "explanation": "Because the fixture says so.",
                    }
                    for idx, stage in enumerate(
                        ["pre", "check", "check", "check", "post", "post"], start=1
                    )
                ],
            }
            write(lesson / "quiz.json", json.dumps(quiz))

            code, output = self.run_with_root(root, ["--phase", "1"])

        self.assertEqual(code, 0)
        self.assertIn("0 issue(s), 0 advisory warning(s)", output)

    def test_quiz_contract_reports_legacy_and_wrong_stage_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lesson = root / "phases" / "01-fixtures" / "01-legacy-quiz"
            write(lesson / "docs" / "en.md", lesson_doc("Python", True))
            write(lesson / "code" / "main.py", "print('fixture')\n")
            write(
                lesson / "code" / "tests" / "test_main.py",
                "\n".join(f"def test_{i}(): assert True" for i in range(5)),
            )
            legacy_quiz = [
                {
                    "stage": "pre",
                    "question": "Legacy list question?",
                    "options": ["A", "B"],
                    "correct": 0,
                    "explanation": "Still renderable, not contract-complete.",
                }
            ]
            write(lesson / "quiz.json", json.dumps(legacy_quiz))

            code, output = self.run_with_root(root, ["--phase", "1"])

        self.assertEqual(code, 0)
        self.assertIn("[A005]", output)
        self.assertIn("[A006]", output)
        self.assertIn("[A007]", output)

    def test_human_report_truncates_long_warning_lists(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for idx in range(30):
                lesson = root / "phases" / "01-fixtures" / f"{idx + 1:02d}-warning-lesson"
                write(lesson / "docs" / "en.md", lesson_doc(None, False))

            code, output = self.run_with_root(root, ["--phase", "1"])

        self.assertEqual(code, 0)
        self.assertIn("... 65 more advisory warning(s); use --json for full detail", output)
        displayed = re.findall(r"^  \[A\d{3}\]", output, flags=re.MULTILINE)
        self.assertEqual(len(displayed), 25)


if __name__ == "__main__":
    unittest.main()
