"""zh-fork review-23 contract smoke tests.

These tests were added by the Chinese fork to check curriculum structure.
They are not original upstream lesson content and do not replace semantic
lesson-specific tests.
"""

from __future__ import annotations

import json
import re
import unittest
from pathlib import Path


LESSON = Path(__file__).resolve().parents[2]
REQUIRED_STAGES = ["pre", "check", "check", "check", "post", "post"]
LANGUAGE_BY_SUFFIX = {
    ".py": "Python",
    ".ts": "TypeScript",
    ".rs": "Rust",
    ".jl": "Julia",
}


class LessonContractTest(unittest.TestCase):
    def test_main_entrypoint_exists_and_matches_declared_languages(self):
        mains = sorted(
            path for path in (LESSON / "code").glob("main.*") if path.suffix in LANGUAGE_BY_SUFFIX
        )
        self.assertTrue(mains, "lesson with tests should ship at least one code/main.* entrypoint")
        doc = (LESSON / "docs" / "en.md").read_text(encoding="utf-8")
        match = re.search(r"^\*\*Languages:\*\*\s*(.+)$", doc, re.MULTILINE)
        self.assertIsNotNone(match)
        declared_line = match.group(1)
        expected = {LANGUAGE_BY_SUFFIX[path.suffix] for path in mains}
        for language in expected:
            self.assertRegex(declared_line, rf"\b{re.escape(language)}\b")

    def test_document_frontmatter_has_required_contract_fields(self):
        doc = (LESSON / "docs" / "en.md").read_text(encoding="utf-8")
        self.assertRegex(doc, r"(?m)^#\s+\S")
        for field in ("Type", "Languages", "Prerequisites", "Time"):
            self.assertRegex(doc, rf"(?m)^\*\*{field}:\*\*\s*\S", msg=f"missing {field} field")
        self.assertRegex(doc, r"(?m)^##\s+Learning Objectives\s*$")

    def test_learning_objectives_are_actionable(self):
        doc = (LESSON / "docs" / "en.md").read_text(encoding="utf-8")
        block = doc.split("## Learning Objectives", 1)[1]
        next_heading = re.search(r"^##\s+", block, re.MULTILINE)
        if next_heading:
            block = block[: next_heading.start()]
        bullets = [line[2:].strip() for line in block.splitlines() if line.startswith("- ")]
        self.assertGreaterEqual(len(bullets), 4)
        for bullet in bullets[:4]:
            first = bullet.split(None, 1)[0]
            self.assertTrue(first[:1].isupper(), bullet)

    def _quiz_questions(self, path):
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            questions = data.get("questions")
        else:
            questions = data
        self.assertIsInstance(questions, list)
        self.assertGreater(len(questions), 0)
        for question in questions:
            self.assertIsInstance(question, dict)
        return questions

    def test_quiz_is_valid_json_when_present(self):
        quiz_path = LESSON / "quiz.json"
        if not quiz_path.is_file():
            self.skipTest("lesson does not ship quiz.json")
        self._quiz_questions(quiz_path)

    def test_chinese_quiz_preserves_source_quiz_shape_when_present(self):
        source_path = LESSON / "quiz.json"
        target_path = LESSON / "quiz.zh-CN.json"
        if not source_path.is_file():
            self.skipTest("lesson does not ship quiz.json")
        if not target_path.is_file():
            self.skipTest("lesson does not ship quiz.zh-CN.json")
        source_questions = self._quiz_questions(source_path)
        target_questions = self._quiz_questions(target_path)
        self.assertEqual(len(target_questions), len(source_questions))
        for src, dst in zip(source_questions, target_questions):
            if "stage" in src or "stage" in dst:
                self.assertEqual(dst.get("stage"), src.get("stage"))
            if "correct" in src or "correct" in dst:
                self.assertEqual(dst.get("correct"), src.get("correct"))
            if "options" in src or "options" in dst:
                self.assertEqual(len(dst.get("options", [])), len(src.get("options", [])))


if __name__ == "__main__":
    unittest.main()
