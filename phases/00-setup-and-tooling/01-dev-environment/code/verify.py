"""Dev environment verifier for the setup lesson.

Lesson docs: phases/00-setup-and-tooling/01-dev-environment/docs/en.md
Checks required runtimes and optional GPU access without hanging on missing tools.
Default mode reports diagnostics and exits 0; --strict turns missing required tools into exit 1.
Refs: Python sys, shutil, argparse, and importlib stdlib documentation.
"""

from __future__ import annotations

import argparse
import importlib
import shutil
import sys
from typing import Callable, List, Optional, Sequence, Tuple

Check = Tuple[str, Callable[[], object], Optional[object]]

REQUIRED_CHECKS: List[Check] = [
    ("Python 3.10+", lambda: sys.version_info >= (3, 10), f"Python {sys.version.split()[0]}"),
    ("NumPy", lambda: importlib.import_module("numpy"), None),
    ("Git", lambda: shutil.which("git") is not None, None),
    ("Node.js", lambda: shutil.which("node") is not None, None),
    ("Rust (cargo)", lambda: shutil.which("cargo") is not None, None),
]

OPTIONAL_CHECKS: List[Check] = [
    ("PyTorch", lambda: importlib.import_module("torch"), None),
    (
        "CUDA",
        lambda: importlib.import_module("torch").cuda.is_available(),
        lambda: (
            importlib.import_module("torch").cuda.get_device_name(0)
            if importlib.import_module("torch").cuda.is_available()
            else "Not available"
        ),
    ),
]


def run_check(name: str, check_fn: Callable[[], object], detail_fn: Optional[object] = None) -> bool:
    try:
        result = check_fn()
        if result is False or result is None:
            raise RuntimeError("check returned no truthy result")
        detail = ""
        if detail_fn:
            detail_value = detail_fn() if callable(detail_fn) else detail_fn
            detail = f" ({detail_value})"
        print(f"  [PASS] {name}{detail}")
        return True
    except Exception:
        print(f"  [FAIL] {name}")
        return False


def run_group(title: str, checks: Sequence[Check]) -> tuple[int, int]:
    print(f"\n{title}:")
    passed = sum(run_check(name, fn, detail) for name, fn, detail in checks)
    return passed, len(checks)


def main(strict: bool = False) -> int:
    print("\n=== AI Engineering from Scratch - Environment Check ===")

    required_passed, required_total = run_group("Required", REQUIRED_CHECKS)
    optional_passed, optional_total = run_group("Optional GPU", OPTIONAL_CHECKS)

    print(f"\nResult: {required_passed}/{required_total} required checks passed", end="")
    if optional_passed:
        print(f", {optional_passed}/{optional_total} optional GPU checks passed")
    else:
        print(" (GPU is optional; most lessons work on CPU)")

    if required_passed == required_total:
        print("\nYou're ready. Start with Phase 1.\n")
        return 0

    print("\nFix the failed required checks above, then run this script again.")
    print("Default mode exits 0 for teaching; rerun with --strict to fail a shell or CI gate.\n")
    return 1 if strict else 0


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check the setup lesson development environment.")
    parser.add_argument("--strict", action="store_true", help="exit 1 when required checks fail")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    sys.exit(main(strict=args.strict))
