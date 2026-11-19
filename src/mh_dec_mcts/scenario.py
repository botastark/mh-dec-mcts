
from dataclasses import dataclass
from mh_dec_mcts.sensing.lattice import Lattice, Pose, start_poses
from mh_dec_mcts.sensing.sensor import Predictions, SensorRates, load_predictions, observe, observe_pred, theoretical_rates
from mh_dec_mcts.world.fields import gaussian_random_field, load_annotations
from mh_dec_mcts.world.grid import Grid
import numpy as np
import copy
import math
import yaml
from pathlib import Path

DEFAULTS = {
    "seed": 42,
    "lattice": {"xy_step": None, "h_min": None, "h_step": None, "n_alt": 6, "fov_deg": 60},
    "sensor": {"type": "theoretical", "a": 1.0, "b": 0.015},
    "agents": {"n": 1, "start": "paper"},
    "run": {"planner": "ig", "weights": "equal", "steps": 100, "runs": 20},
}
 
@dataclass
class Scenario:
    name: str
    run_seed: int
    grid: Grid
    lattice: Lattice
    truth: np.ndarray
    rates: SensorRates                                              # what the belief update assumes, per altitude level k
    preds: Predictions | None  # classifier maps (sensor type "predictions") or None
    starts: list[Pose]                                     # one per agent
    obs_rng: np.random.Generator                                 # sensor noise
    plan_rng: np.random.Generator                                # tie-breaks, Random, Sweep direction

    def observe(self, pose: Pose) -> np.ndarray:
        """What the UAV sees from pose: classifier prediction or truth with synthetic noise."""
        b = self.lattice.footprint(pose)
        if self.preds is not None:
            return observe_pred(self.preds.maps, pose.k, b)
        return observe(self.truth, b, *self.rates.at(pose.k), self.obs_rng)
        

# ------------------------------------------------------------------ config
def load_config(path, **overrides) -> dict:
    """Read a YAML config, fill defaults, apply overrides like run={"steps": 30}."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("name", Path(path).stem)
    for section, values in overrides.items():
        if isinstance(values, dict):
            cfg.setdefault(section, {}).update(values)
        else:
            cfg[section] = values
    return with_defaults(cfg)
 
 
def with_defaults(cfg: dict) -> dict:
    out = copy.deepcopy(cfg)
    for key, val in DEFAULTS.items():
        if isinstance(val, dict):
            out[key] = {**val, **out.get(key, {})}
        else:
            out.setdefault(key, val)
    for key in ("grid", "field"):
        if key not in out:
            raise ValueError(f"config '{out.get('name')}' has no '{key}' section")
    return out
 
 
def make_lattice(grid: Grid, lc: dict) -> Lattice:
    """null xy_step -> min(size)/16; null h_step -> xy_step / tan(fov/2); null h_min -> h_step."""
    xy = lc["xy_step"] or min(grid.size_x, grid.size_y) / 16
    h_step = lc["h_step"] or xy / math.tan(math.radians(lc["fov_deg"]) / 2)
    h_min = lc["h_min"] or h_step
    return Lattice(grid, xy, h_min, h_step, lc["n_alt"], lc["fov_deg"])
 



# --------------------------------------------------------------- scenario
def make_scenario(cfg: dict, run: int = 0) -> Scenario:
    cfg = with_defaults(cfg)
    run_seed = int(cfg["seed"]) + run
    grid = Grid(**cfg["grid"])
    lat = make_lattice(grid, cfg["lattice"])
 
    # ground truth
    f = cfg["field"]
    if f["type"] == "gaussian":
        truth = gaussian_random_field(f["r"], grid.shape, np.random.default_rng(run_seed))
    elif f["type"] == "orthomap":
        truth = load_annotations(f["annotations"], f.get("crop"), tuple(f.get("positive_labels", (1, 2, 10))))
    else:
        raise ValueError(f"unknown field type '{f['type']}'")
    if truth.shape != grid.shape:
        raise ValueError(f"field {truth.shape} does not match grid {grid.shape}")
 
    # sensor: rates computed once (theoretical) or loaded once per process (predictions)
    s = cfg["sensor"]
    if s["type"] == "theoretical":
        rates, preds = theoretical_rates(lat.altitudes, s["a"], s["b"]), None
    elif s["type"] == "predictions":
        preds = load_predictions(s["file"], lat.altitudes, truth)
        rates = preds.rates
    else:
        raise ValueError(f"unknown sensor type '{s['type']}'")
 
    # start poses
    a = cfg["agents"]
    starts = make_starts(lat, a["n"], a["start"], np.random.default_rng(run_seed))
 
    return Scenario(
        name=cfg["name"], run_seed=run_seed, grid=grid, lattice=lat, truth=truth,
        rates=rates, preds=preds, starts=starts,
        obs_rng=np.random.default_rng(run_seed),       # legacy: Field.rng reset to the run seed
        plan_rng=np.random.default_rng([run_seed, 1]),  # independent of the observation draws
    )
 
 
def make_starts(lat: Lattice, n: int, mode: str, rng: np.random.Generator) -> list[Pose]:
    """paper:  1 agent in the lower-left corner; N agents in the legacy corner order.
    legacy: start_poses() (1 agent: random corner from the run seed).
    random: n distinct random waypoints. All at the lowest altitude."""
    if mode == "paper":
        return [Pose(0, 0, 0)] if n == 1 else start_poses(lat, n)
    if mode == "legacy":
        return start_poses(lat, n, rng)
    if mode == "random":
        flat = rng.choice(lat.nx * lat.ny, size=n, replace=False)
        return [Pose(int(f // lat.ny), int(f % lat.ny), 0) for f in flat]
    raise ValueError(f"unknown start mode '{mode}'")