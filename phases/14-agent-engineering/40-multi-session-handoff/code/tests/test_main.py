import json
import tempfile
import unittest
from pathlib import Path

import main


class HandoffGeneratorTests(unittest.TestCase):
    def snapshot(self, **overrides):
        data = {
            "task_id": "T-42",
            "state": {
                "blockers": ["needs owner decision"],
                "next_action": "rerun verification and open the PR",
            },
            "verdict": {
                "passed": True,
                "findings": [
                    {"severity": "info", "detail": "style nit"},
                    {"severity": "warn", "detail": "missing edge case"},
                    {"severity": "block", "detail": "wrong target"},
                ],
            },
            "review": {"verdict": "soft_fail", "total": 6},
            "feedback": [
                {"command": "pytest test_a.py", "exit_code": 0},
                {"command": "pytest test_b.py", "exit_code": 1},
                {"command": "ruff check .", "exit_code": 0},
            ],
            "diff_summary": {"touched": ["app.py", "tests/test_app.py"]},
        }
        data.update(overrides)
        return main.WorkbenchSnapshot(**data)

    def test_trim_feedback_keeps_tail_and_nonzero_failures(self):
        records = [{"command": f"cmd-{i}", "exit_code": 0} for i in range(8)]
        records[1]["exit_code"] = 2

        trimmed = main.trim_feedback(records)

        self.assertEqual([row["command"] for row in trimmed], ["cmd-3", "cmd-4", "cmd-5", "cmd-6", "cmd-7", "cmd-1"])

    def test_trim_feedback_deduplicates_failures_already_in_tail(self):
        records = [{"command": f"cmd-{i}", "exit_code": 0} for i in range(6)]
        records[5]["exit_code"] = 1

        trimmed = main.trim_feedback(records)

        self.assertEqual([row["command"] for row in trimmed].count("cmd-5"), 1)

    def test_derive_risks_collects_warn_blockers_and_low_review_total(self):
        risks = main.derive_risks(self.snapshot())

        details = [risk["detail"] for risk in risks]
        self.assertIn("missing edge case", details)
        self.assertIn("wrong target", details)
        self.assertIn("open blocker: needs owner decision", details)
        self.assertIn("review total 6 below 7", details)
        self.assertNotIn("style nit", details)

    def test_generate_handoff_shapes_markdown_and_payload(self):
        markdown, payload = main.generate_handoff(self.snapshot())

        self.assertIn("# Handoff: T-42", markdown)
        self.assertIn("`app.py`", markdown)
        self.assertIn("pytest test_b.py -> exit 1", markdown)
        self.assertEqual(payload.changed_files, ["app.py", "tests/test_app.py"])
        self.assertEqual(payload.next_action, "rerun verification and open the PR")
        self.assertEqual(payload.verdict_pointer["review"], "outputs/review/T-42.json")

    def test_generate_handoff_falls_back_when_next_action_is_missing(self):
        snapshot = self.snapshot(state={"blockers": [], "next_action": ""})

        _, payload = main.generate_handoff(snapshot)

        self.assertEqual(payload.next_action, "no next_action recorded; needs human")

    def test_main_writes_markdown_and_json_packet(self):
        original_here = main.HERE
        with tempfile.TemporaryDirectory() as tmp:
            main.HERE = Path(tmp)
            try:
                main.main()
                markdown = (Path(tmp) / "handoff.md").read_text()
                payload = json.loads((Path(tmp) / "handoff.json").read_text())
            finally:
                main.HERE = original_here

        self.assertIn("# Handoff: T-001", markdown)
        self.assertEqual(payload["task_id"], "T-001")
        self.assertIn("feedback_tail", payload)


if __name__ == "__main__":
    unittest.main()
