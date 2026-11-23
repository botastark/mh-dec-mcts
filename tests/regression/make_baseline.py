"""Write the regression baseline: every case of cases.py through the current code.

    python tests/regression/make_baseline.py            # all cases (a few minutes)
    python tests/regression/make_baseline.py tune       # only some cases

Run once on the code before a refactor and commit tests/regression/baseline/; test_baseline.py
then checks that the refactored code reproduces every number exactly.
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cases import collect  # noqa: E402

BASELINE = Path(__file__).resolve().parent / "baseline"


def main():
    only = sys.argv[1:] or None
    BASELINE.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        for name, data in collect(Path(tmp), only).items():
            (BASELINE / f"{name}.json").write_text(json.dumps(data))
            print(f"-> {BASELINE / name}.json", flush=True)


if __name__ == "__main__":
    main()
