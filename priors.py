"""
Loads source-parameter priors and fixed values from config.yaml, so
switching which parameters the network trains on (`active_params` in the
YAML) doesn't require touching simulator.py. See SUMMARY.md for context.
"""

from pathlib import Path

import numpy as np
import yaml
from lisatools.utils.constants import PC_SI

CONFIG_PATH = Path(__file__).parent / "config.yaml"

# Maps a uniform draw over `prior` to the parameter's natural physical value
# (Msun, radians, ...). identity/sin/cos keep the prior draw bounded in
# [-1, 1] while still landing on a physically uniform-over-the-sphere
# distribution for beta (latitude) and inc.
_SAMPLE_TRANSFORMS = {
    "identity": lambda u: u,
    "log10": lambda u: 10.0 ** u,
    "sin": lambda u: np.arcsin(u),
    "cos": lambda u: np.arccos(u),
}

# Natural physical unit (as used in `parameters` below) -> unit bbhx expects.
# Only dist needs this: config/priors are in Gpc, bbhx wants SI.
_TO_BBHX_UNIT = {
    "dist": lambda dist_gpc: dist_gpc * 1e9 * PC_SI,
}


def _identity(x):
    return x


def _eval_expr(expr):
    if isinstance(expr, str):
        return eval(expr, {"__builtins__": {}}, {"pi": np.pi})
    return float(expr)


with open(CONFIG_PATH) as f:
    _config = yaml.safe_load(f)

ACTIVE_PARAMS = _config["active_params"]
FREE_PARAM_NAMES = ACTIVE_PARAMS

_PARAM_SPECS = {
    name: dict(
        transform=spec["transform"],
        prior=[_eval_expr(spec["prior"][0]), _eval_expr(spec["prior"][1])],
        fixed_value=_eval_expr(spec["fixed_value"]),
    )
    for name, spec in _config["parameters"].items()
}

ALL_PARAM_NAMES = list(_PARAM_SPECS.keys())

FREE_PRIORS = {name: _PARAM_SPECS[name]["prior"] for name in ACTIVE_PARAMS}

FIXED_PARAMS = {
    name: _TO_BBHX_UNIT.get(name, _identity)(spec["fixed_value"])
    for name, spec in _PARAM_SPECS.items()
    if name not in ACTIVE_PARAMS
}
FIXED_PARAMS["f_ref"] = float(_config["f_ref"])
FIXED_PARAMS["t_ref_offset_days"] = float(_config["t_ref_offset_days"])


def sample_free_parameters(n_samples, rng=None):
    rng = np.random.default_rng() if rng is None else rng

    samples = {}
    for name in ACTIVE_PARAMS:
        spec = _PARAM_SPECS[name]
        low, high = spec["prior"]
        u = rng.uniform(low, high, size=n_samples)
        physical = _SAMPLE_TRANSFORMS[spec["transform"]](u)
        samples[name] = _TO_BBHX_UNIT.get(name, _identity)(physical)
    return samples
