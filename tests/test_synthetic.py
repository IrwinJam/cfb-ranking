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
truth = prov.true.reindex(r.index)
corr = np.corrcoef(r.theta_margin, truth)[0, 1]
print(f"correlation of the margin ratings with true strengths after 12 weeks: {corr:.3f}")
assert corr > 0.85
blend = np.corrcoef(r.theta, truth)[0, 1]
print(f"correlation of the blended strength with true strengths: {blend:.3f}")
assert blend > 0.6
