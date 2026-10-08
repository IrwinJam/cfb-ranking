"""Load the model settings from fitted_params.json, or use the defaults."""
import json
from dataclasses import replace
from pathlib import Path

from .config import Params

FITTED = Path(__file__).resolve().parents[1] / "fitted_params.json"


def load_params(which="fitted", quiet=False):
    if which == "fitted" and FITTED.exists():
        d = json.loads(FITTED.read_text())
        v = dict(d["params"])
        b = v.pop("beta_late")
        if not quiet:
            print(f"[params] fitted to {d.get('source', 'AP poll')} through week {d['through_week']}: {d['params']}")
        return replace(Params(), beta_early=0.6 * b, beta_late=b, **v)
    if not quiet:
        print("[params] defaults")
    return Params()
