"""Tests that fork-authored curriculum changes are visibly marked."""

from __future__ import annotations

import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def changed_phase_paths() -> list[Path]:
    tracked = subprocess.run(
        ["git", "diff", "--name-only", "--", "phases"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        check=True,
    ).stdout.splitlines()
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "--", "phases"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        check=True,
    ).stdout.splitlines()
    return [ROOT / path for path in tracked + untracked]


class ForkVisibleMarkersTest(unittest.TestCase):
    def test_changed_docs_use_visible_fork_note(self):
        missing = []
        for path in changed_phase_paths():
            if not path.as_posix().endswith("/docs/en.md"):
                continue
            text = path.read_text(encoding="utf-8")
            if "::: fork-note " not in text:
                missing.append(path.relative_to(ROOT).as_posix())
        self.assertEqual([], missing)

    def test_added_contract_tests_use_source_header(self):
        missing = []
        for path in changed_phase_paths():
            if path.name != "test_contract.py":
                continue
            text = path.read_text(encoding="utf-8")
            if "zh-fork" not in text or "review-23" not in text:
                missing.append(path.relative_to(ROOT).as_posix())
        self.assertEqual([], missing)


if __name__ == "__main__":
    unittest.main()
