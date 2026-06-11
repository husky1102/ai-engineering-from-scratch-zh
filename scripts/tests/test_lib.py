from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts import _lib


class SharedLibTest(unittest.TestCase):
    def test_rel_path_uses_posix_separators(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "phases" / "01-alpha" / "02-beta"

            self.assertEqual(_lib.rel_path(path, root), "phases/01-alpha/02-beta")

    def test_iter_lesson_dirs_filters_valid_numbered_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            phases = Path(tmp) / "phases"
            (phases / "01-alpha" / "01-first").mkdir(parents=True)
            (phases / "01-alpha" / "draft").mkdir()
            (phases / "not-a-phase" / "01-hidden").mkdir(parents=True)
            (phases / "02-beta" / "03-third").mkdir(parents=True)

            lessons = [p.as_posix().split("/phases/", 1)[1] for p in _lib.iter_lesson_dirs(phases_dir=phases)]
            phase_one = [
                p.as_posix().split("/phases/", 1)[1]
                for p in _lib.iter_lesson_dirs(phase_filter=1, phases_dir=phases)
            ]

        self.assertEqual(lessons, ["01-alpha/01-first", "02-beta/03-third"])
        self.assertEqual(phase_one, ["01-alpha/01-first"])

    def test_main_languages_uses_only_main_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            lesson = Path(tmp) / "01-lesson"
            code = lesson / "code"
            code.mkdir(parents=True)
            (code / "main.py").write_text("", encoding="utf-8")
            (code / "main.ts").write_text("", encoding="utf-8")
            (code / "helper.rs").write_text("", encoding="utf-8")

            self.assertEqual(_lib.main_languages(lesson), {"Python", "TypeScript"})

    def test_parse_frontmatter_handles_inline_lists_and_quotes(self):
        text = """---
name: demo
description: "quoted value"
tags: [alpha, 'beta', "gamma"]
---

# Body
"""

        self.assertEqual(
            _lib.parse_frontmatter(text),
            {"name": "demo", "description": "quoted value", "tags": ["alpha", "beta", "gamma"]},
        )


if __name__ == "__main__":
    unittest.main()
