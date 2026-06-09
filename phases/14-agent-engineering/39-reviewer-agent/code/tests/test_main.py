import json
import unittest
from dataclasses import asdict

import main


class ReviewerAgentTests(unittest.TestCase):
    def inputs(self, **overrides):
        data = {
            "task_id": "T-test",
            "goal": "add input validation to signup",
            "diff_summary": {"touched": ["app/signup.py", "tests/test_signup.py"]},
            "state": {
                "active_task_id": None,
                "assumptions": ["users sign up with email + password only"],
                "next_action": "pick next task from board",
            },
            "feedback": [{"command": "pytest", "exit_code": 0}],
            "verdict": {"passed": True, "findings": []},
        }
        data.update(overrides)
        return main.ReviewerInputs(**data)

    def dimension_map(self, report):
        return {dim.name: dim for dim in report.dimensions}

    def test_clean_change_passes_with_five_dimensions(self):
        report = main.review(self.inputs())

        self.assertEqual("pass", report.verdict)
        self.assertEqual(5, len(report.dimensions))
        self.assertEqual(9, report.total)

    def test_wrong_problem_is_hard_fail_even_with_passing_feedback(self):
        report = main.review(self.inputs(diff_summary={"touched": ["docs/api.md"]}))

        self.assertEqual("hard_fail", report.verdict)
        self.assertEqual(0, self.dimension_map(report)["problem_fit"].score)

    def test_scope_forbidden_write_scores_zero(self):
        report = main.review(
            self.inputs(verdict={"passed": False, "findings": [{"code": "scope.forbidden", "severity": "block"}]})
        )

        self.assertEqual("hard_fail", report.verdict)
        self.assertEqual(0, self.dimension_map(report)["scope_discipline"].score)

    def test_missing_feedback_exit_scores_zero_verification(self):
        report = main.review(self.inputs(feedback=[{"command": "pytest", "exit_code": None}]))

        self.assertEqual("hard_fail", report.verdict)
        self.assertEqual(0, self.dimension_map(report)["verification_quality"].score)

    def test_soft_fail_requires_low_total_without_zero_dimension(self):
        report = main.review(
            self.inputs(
                state={"active_task_id": "T-test", "assumptions": [], "next_action": ""},
                verdict={"passed": True, "findings": [{"code": "scope.off_scope", "severity": "warn"}]},
            )
        )

        self.assertEqual("soft_fail", report.verdict)
        self.assertEqual(6, report.total)

    def test_handoff_without_next_action_scores_zero(self):
        score = main.score_handoff(self.inputs(state={"active_task_id": None, "assumptions": [], "next_action": ""}))

        self.assertEqual("handoff_readiness", score.name)
        self.assertEqual(0, score.score)

    def test_review_report_serializes_to_json_shape(self):
        report = main.review(self.inputs())
        payload = {
            "task_id": report.task_id,
            "total": report.total,
            "verdict": report.verdict,
            "dimensions": [asdict(dim) for dim in report.dimensions],
        }

        encoded = json.dumps(payload)
        decoded = json.loads(encoded)
        self.assertEqual("T-test", decoded["task_id"])
        self.assertEqual("pass", decoded["verdict"])
        self.assertEqual("problem_fit", decoded["dimensions"][0]["name"])


if __name__ == "__main__":
    unittest.main()
