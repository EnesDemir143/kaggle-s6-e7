#!/usr/bin/env python3
"""Run every mandatory local quality gate in a deterministic order."""

import subprocess
import sys

COMMANDS = (
    (sys.executable, "-m", "pytest"),
    (sys.executable, "-m", "ruff", "check", "src", "tests", "scripts"),
    (sys.executable, "-m", "mypy", "src", "scripts"),
    (sys.executable, "-m", "compileall", "-q", "src", "tests", "scripts"),
)


def main() -> int:
    for command in COMMANDS:
        print(f"\n> {' '.join(command)}", flush=True)
        result = subprocess.run(command, check=False)
        if result.returncode:
            return result.returncode
    print("\nAll quality gates passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
