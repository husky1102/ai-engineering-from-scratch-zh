import unittest

from scripts import build_search_index


class BuildSearchIndexTest(unittest.TestCase):
    def test_index_carries_lab_metadata_from_runtime_manifest(self):
        index = build_search_index.build_index()
        lesson = next(
            row
            for row in index["lessons"]
            if row["path"] == "phases/01-math-foundations/01-linear-algebra-intuition"
        )

        self.assertIn("lab", lesson)
        self.assertTrue(str(lesson["lab"]).startswith("lab/index.html?path=notebooks/"))


if __name__ == "__main__":
    unittest.main()
