"""Python entrypoint for the dev environment verifier.

Lesson docs: phases/00-setup-and-tooling/01-dev-environment/docs/en.md
Delegates to verify.py so the original script path remains compatible.
Default mode reports diagnostics and exits 0; --strict enables failure gates.
Refs: Python sys and argparse stdlib documentation.
"""

from __future__ import annotations

import sys

from verify import main, parse_args


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    sys.exit(main(strict=args.strict))
