"""Every regression case must reproduce the stored baseline exactly (make_baseline.py).

    pytest -m regression tests/regression            # a few minutes
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cases import MYOPIA_CASES, SINGLE_CASES, collect, diff  # noqa: E402

BASELINE = Path(__file__).resolve().parent / "baseline"
CASES = [*SINGLE_CASES, "multi_fig5", *MYOPIA_CASES, "tune", "horizon_sweep"]


@pytest.mark.regression
@pytest.mark.parametrize("case", CASES)
def test_case_matches_baseline(case, tmp_path):
    expected = json.loads((BASELINE / f"{case}.json").read_text())
    got = json.loads(json.dumps(collect(tmp_path, [case])[case]))  # same normalisation as stored
    bad = diff(expected, got)
    assert not bad, f"{len(bad)} differences, first: " + "; ".join(bad[:5])
