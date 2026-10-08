"""Check on made-up data: the fitted ratings should track the true strengths. Run: python tests/test_synthetic.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from aapr.config import Params
from aapr.pipeline import run_week
from aapr.synthetic import SyntheticProvider

prov = SyntheticProvider()
r = run_week(prov, 2025, 12, Params(), prior="none")["rankings"].set_index("team")
corr = np.corrcoef(r.theta, prov.true.reindex(r.index))[0, 1]
print(f"correlation of fitted ratings with true strengths after 12 weeks: {corr:.3f}")
assert corr > 0.85
