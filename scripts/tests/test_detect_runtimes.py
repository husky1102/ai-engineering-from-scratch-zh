import unittest

from scripts import detect_runtimes


class DetectRuntimesTest(unittest.TestCase):
    def test_runtime_record_has_notebook_and_lab_slots(self):
        lesson = detect_runtimes.ROOT / "phases/01-math-foundations/01-linear-algebra-intuition"

        record = detect_runtimes.detect_lesson(lesson)

        self.assertIn("notebook", record)
        self.assertIn("lab", record)


if __name__ == "__main__":
    unittest.main()
