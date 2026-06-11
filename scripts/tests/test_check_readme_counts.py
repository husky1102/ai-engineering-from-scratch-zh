from __future__ import annotations

import unittest

from scripts import check_readme_counts


def sample_readme(lessons: int = 503, phases: int = 20, skills: int = 10, prompts: int = 4) -> str:
    return "\n".join(
        [
            f"![lessons](https://img.shields.io/badge/lessons-{lessons}-3553ff)",
            f'<img alt="{lessons} lessons" src="x">',
            f"> {lessons} lessons. {phases} phases.",
            f"This curriculum is the spine. {phases} phases, {lessons} lessons, built carefully.",
            f"![phases](https://img.shields.io/badge/phases-{phases}-3553ff)",
            f'<img alt="{phases} phases" src="x">',
            f"A portfolio of {lessons} artifacts for daily practice.",
            f"The repo ships {skills} skills and {prompts} prompts for reuse.",
            f"MIT-licensed, {lessons} lessons.",
            "",
        ]
    )


class CheckReadmeCountsTest(unittest.TestCase):
    def test_find_mismatches_reports_line_and_expected_value(self):
        totals = {"lessons": 504, "phases": 20, "skills": 10, "prompts": 4}

        mismatches = check_readme_counts.find_mismatches(sample_readme(), totals)

        self.assertGreaterEqual(len(mismatches), 1)
        self.assertTrue(all(m.pattern.field == "lessons" for m in mismatches))
        self.assertEqual(mismatches[0].expected, 504)
        self.assertEqual(mismatches[0].found, 503)
        self.assertGreaterEqual(mismatches[0].line, 1)

    def test_apply_fixes_rewrites_all_hardcoded_counts(self):
        totals = {"lessons": 504, "phases": 21, "skills": 11, "prompts": 5}
        readme = sample_readme()

        fixed = check_readme_counts.apply_fixes(readme, totals)
        mismatches = check_readme_counts.find_mismatches(fixed, totals)

        self.assertEqual(mismatches, [])
        self.assertIn("lessons-504-3553ff", fixed)
        self.assertIn("> 504 lessons. 21 phases.", fixed)
        self.assertIn("The repo ships 11 skills and 5 prompts", fixed)

    def test_missing_readme_shape_raises_system_exit(self):
        totals = {"lessons": 504, "phases": 21, "skills": 11, "prompts": 5}

        with self.assertRaises(SystemExit):
            check_readme_counts.find_mismatches("no count-bearing README structure", totals)


if __name__ == "__main__":
    unittest.main()
