import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import main


class VerificationGateTests(unittest.TestCase):
    def artifact(self, **overrides):
        data = {
            "task_id": "T-test",
            "acceptance_commands": ["python3 -m unittest"],
            "feedback": [{"command": "python3 -m unittest", "exit_code": 0}],
            "scope_report": {"forbidden_writes": [], "off_scope_writes": []},
            "rule_report": [{"slug": "done/tests-pass", "passed": True}],
            "coverage_report": {"current": 0.84, "previous": 0.85},
            "head_commit": "abc1234",
        }
        data.update(overrides)
        return main.Artifacts(**data)

    def finding_codes(self, report):
        return {finding.code for finding in report.findings}

    def severities(self, report):
        return {finding.code: finding.severity for finding in report.findings}

    def test_exact_one_point_coverage_drop_does_not_block(self):
        report = main.verify(self.artifact())

        self.assertTrue(report.passed)
        self.assertNotIn("coverage.regression", self.finding_codes(report))
        self.assertEqual("warn", self.severities(report)["coverage.minor_regression"])

    def test_missing_acceptance_command_blocks(self):
        report = main.verify(self.artifact(feedback=[]))

        self.assertFalse(report.passed)
        self.assertIn("acceptance.missing", self.finding_codes(report))

    def test_failed_acceptance_exit_blocks(self):
        report = main.verify(self.artifact(feedback=[{"command": "python3 -m unittest", "exit_code": 1}]))

        self.assertFalse(report.passed)
        self.assertIn("acceptance.failed", self.finding_codes(report))

    def test_forbidden_and_off_scope_writes_are_distinguished(self):
        report = main.verify(
            self.artifact(scope_report={"forbidden_writes": ["scripts/release.sh"], "off_scope_writes": ["README.md"]})
        )

        self.assertFalse(report.passed)
        self.assertEqual("block", self.severities(report)["scope.forbidden"])
        self.assertEqual("warn", self.severities(report)["scope.off_scope"])

    def test_strict_mode_promotes_warnings_to_blocks(self):
        art = self.artifact(
            scope_report={"forbidden_writes": [], "off_scope_writes": ["README.md"]},
            coverage_report={"current": 0.85, "previous": 0.85},
        )

        report = main.verify(art, strict=True)

        self.assertFalse(report.passed)
        self.assertEqual("block", self.severities(report)["scope.off_scope"])

    def test_coverage_below_floor_blocks(self):
        report = main.verify(self.artifact(coverage_report={"current": 0.62, "previous": 0.80}))

        self.assertFalse(report.passed)
        self.assertIn("coverage.below_floor", self.finding_codes(report))

    def test_signed_override_round_trips_to_jsonl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            override_path = Path(tmpdir) / "overrides.jsonl"
            with patch.object(main, "OVERRIDES_PATH", override_path), patch.dict(
                os.environ, {"VERIFY_OVERRIDE_SECRET": "test-secret"}, clear=False
            ):
                entry = main.record_override(
                    task_id="T-002",
                    finding_code="scope.off_scope",
                    reason="reviewed scope expansion",
                    user_id="alice",
                    head_commit="def5678",
                )

                self.assertTrue(main.verify_signature(entry))
                self.assertEqual(1, len(override_path.read_text().splitlines()))


if __name__ == "__main__":
    unittest.main()
