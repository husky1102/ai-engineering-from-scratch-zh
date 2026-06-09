import contextlib
import io
import unittest

import main as main_entry
import verify


class VerifyEnvironmentTests(unittest.TestCase):
    def capture(self, func, *args, **kwargs):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            result = func(*args, **kwargs)
        return result, out.getvalue()

    def test_run_check_passes_with_static_detail(self):
        ok, output = self.capture(verify.run_check, "Tool", lambda: True, "v1")

        self.assertTrue(ok)
        self.assertIn("[PASS] Tool (v1)", output)

    def test_run_check_fails_on_false_result(self):
        ok, output = self.capture(verify.run_check, "Tool", lambda: False)

        self.assertFalse(ok)
        self.assertIn("[FAIL] Tool", output)

    def test_default_mode_reports_required_failures_but_exits_zero(self):
        original_required = verify.REQUIRED_CHECKS
        original_optional = verify.OPTIONAL_CHECKS
        verify.REQUIRED_CHECKS = [("Missing required", lambda: False, None)]
        verify.OPTIONAL_CHECKS = []
        try:
            code, output = self.capture(verify.main)
        finally:
            verify.REQUIRED_CHECKS = original_required
            verify.OPTIONAL_CHECKS = original_optional

        self.assertEqual(code, 0)
        self.assertIn("rerun with --strict", output)

    def test_strict_mode_fails_required_failures(self):
        original_required = verify.REQUIRED_CHECKS
        original_optional = verify.OPTIONAL_CHECKS
        verify.REQUIRED_CHECKS = [("Missing required", lambda: False, None)]
        verify.OPTIONAL_CHECKS = []
        try:
            code, _ = self.capture(verify.main, strict=True)
        finally:
            verify.REQUIRED_CHECKS = original_required
            verify.OPTIONAL_CHECKS = original_optional

        self.assertEqual(code, 1)

    def test_optional_failures_do_not_fail_strict_mode(self):
        original_required = verify.REQUIRED_CHECKS
        original_optional = verify.OPTIONAL_CHECKS
        verify.REQUIRED_CHECKS = [("Required", lambda: True, None)]
        verify.OPTIONAL_CHECKS = [("Optional", lambda: False, None)]
        try:
            code, _ = self.capture(verify.main, strict=True)
        finally:
            verify.REQUIRED_CHECKS = original_required
            verify.OPTIONAL_CHECKS = original_optional

        self.assertEqual(code, 0)

    def test_main_entrypoint_reuses_verify_parser(self):
        args = main_entry.parse_args(["--strict"])

        self.assertTrue(args.strict)


if __name__ == "__main__":
    unittest.main()
