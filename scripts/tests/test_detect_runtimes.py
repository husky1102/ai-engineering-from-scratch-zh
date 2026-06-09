import unittest

from scripts import detect_runtimes


class DetectRuntimesTest(unittest.TestCase):
    def test_runtime_record_has_notebook_and_lab_slots(self):
        lesson = detect_runtimes.ROOT / "phases/01-math-foundations/01-linear-algebra-intuition"

        record = detect_runtimes.detect_lesson(lesson)

        self.assertIn("notebook", record)
        self.assertIn("lab", record)

    def test_tests_and_hmac_do_not_pollute_runtime_packages(self):
        lesson = detect_runtimes.ROOT / "phases/14-agent-engineering/38-verification-gates"

        record = detect_runtimes.detect_lesson(lesson)

        self.assertEqual("local-kernel", record["runtime"])
        self.assertNotIn("hmac", record["packages"])
        self.assertNotIn("main", record["packages"])
        self.assertNotIn("tempfile", record["packages"])
        self.assertIn("uses API key or secret", record["notes"])

    def test_local_python_modules_do_not_pollute_runtime_packages(self):
        lesson = detect_runtimes.ROOT / "phases/00-setup-and-tooling/01-dev-environment"

        record = detect_runtimes.detect_lesson(lesson)

        self.assertEqual("phases/00-setup-and-tooling/01-dev-environment/code/main.py", record["entry"])
        self.assertNotIn("verify", record["packages"])


if __name__ == "__main__":
    unittest.main()
