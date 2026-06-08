import unittest

from scripts import build_lesson_assets


class BuildLessonAssetsTest(unittest.TestCase):
    def test_manifest_uses_local_urls_for_outputs(self):
        manifest = build_lesson_assets.build_manifest()
        lesson_path = "phases/00-setup-and-tooling/01-dev-environment"

        lesson = manifest["lessons"][lesson_path]
        self.assertGreater(len(lesson["outputs"]), 0)

        first_output = lesson["outputs"][0]
        self.assertTrue(first_output["url"].startswith("../" + lesson_path + "/outputs/"))
        self.assertNotIn("download_url", first_output)
        self.assertNotIn("html_url", first_output)
        self.assertIn(first_output["kind"], {"prompt", "skill", "artifact"})

    def test_manifest_includes_code_file_metadata(self):
        manifest = build_lesson_assets.build_manifest()
        lesson_path = "phases/00-setup-and-tooling/01-dev-environment"

        lesson = manifest["lessons"][lesson_path]
        code_paths = {record["path"] for record in lesson["code"]}

        self.assertIn(lesson_path + "/code/verify.py", code_paths)
        verify_record = next(record for record in lesson["code"] if record["path"].endswith("/verify.py"))
        self.assertEqual(verify_record["language"], "python")
        self.assertEqual(verify_record["command"], "python verify.py")


if __name__ == "__main__":
    unittest.main()
