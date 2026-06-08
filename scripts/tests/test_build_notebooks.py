import unittest

from scripts import build_notebooks


class BuildNotebooksTest(unittest.TestCase):
    def test_lab_url_points_to_local_lab_entry(self):
        notebook = "site/notebooks/phases/01-math-foundations/01-linear-algebra-intuition/lesson.ipynb"

        lab_url = build_notebooks.lab_url_for(notebook)

        self.assertEqual(
            lab_url,
            "lab/index.html?path=notebooks/phases/01-math-foundations/01-linear-algebra-intuition/lesson.ipynb",
        )


if __name__ == "__main__":
    unittest.main()
