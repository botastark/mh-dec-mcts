"""Config -> everything that stays fixed during one run.

    cfg = load_config("configs/paper.yaml")
    sc = make_scenario(cfg, run=0)
    z = sc.observe(pose)

This is the only module that branches on the field / sensor type. Planners, the mapper and
the simulation loop only see a Scenario.

Seeding (default, legacy-compatible): run i uses run_seed = seed + i for the Gaussian field,
the observation noise and the legacy random start corner; planner randomness (tie-breaks,
Random, Sweep direction) gets its own stream so it never shifts the observation draws.

compat: pierdicca reproduces Pierdicca's Precision-Agriculture-Dev exactly: one field
generator (seed 123) and one agent generator (seed 0, shared by sensor noise and planner)
are created once per experiment and carried over from run to run. Create them with
make_streams(cfg) and pass the same object to make_scenario for runs 0, 1, 2, ... in order.
"""

import copy
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from mh_dec_mcts.sensing.lattice import Lattice, Pose, start_poses
from mh_dec_mcts.sensing.sensor import Predictions, SensorRates, load_predictions, observe, observe_pred, theoretical_rates
from mh_dec_mcts.world.fields import load_annotations, gaussian_random_field
from mh_dec_mcts.world.grid import Grid

DEFAULTS = {
    "seed": 42,
    "compat": None,  # None | "pierdicca"
    "pierdicca": {"map_seed": 123, "agent_seed": 0, "position_seed": 12, "order_seed": 17},
    "lattice": {"xy_step": None, "h_min": None, "h_step": None, "n_alt": 6, "fov_deg": 60},
    "sensor": {"type": "theoretical", "a": 1.0, "b": 0.015},
    "agents": {"n": 1, "start": "paper"},
    "comms": {"range": None, "sharing": "none"},  # range in waypoint steps (None = unlimited); none | BS | BM
    "run": {"planner": "ig", "weights": "equal", "steps": 100, "runs": 20},
    "mcts": {},  # DecMCTSConfig overrides for planner "decmcts" (planning/dec_mcts.py)
}


@dataclass
class Scenario:
    name: str
    run_seed: int
    grid: Grid
    lattice: Lattice
    truth: np.ndarray  # (rows, cols) uint8
    rates: SensorRates  # what the belief update assumes, per altitude level
    preds: Predictions | None  # classifier maps (sensor type "predictions") or None
    starts: list[Pose]  # one per agent
    obs_rngs: list[np.random.Generator]  # per agent: sensor noise
    plan_rngs: list[np.random.Generator]  # per agent: tie-breaks, Random, Sweep direction
    order_rng: np.random.Generator  # multi-agent decision order (IGd)

    @property
    def obs_rng(self) -> np.random.Generator:  # single agent
        return self.obs_rngs[0]

    @property
    def plan_rng(self) -> np.random.Generator:
        return self.plan_rngs[0]

    def observe(self, pose: Pose, agent: int = 0) -> np.ndarray:
        """What the UAV sees from pose: classifier prediction or truth with synthetic noise."""
        b = self.lattice.footprint(pose)
        if self.preds is not None:
            return observe_pred(self.preds.maps, pose.k, b)
        return observe(self.truth, b, *self.rates.at(pose.k), self.obs_rngs[agent])


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
def make_streams(cfg: dict) -> dict | None:
    """Experiment-wide generators for compat: pierdicca (None otherwise). One per experiment."""
    cfg = with_defaults(cfg)
    if cfg["compat"] != "pierdicca":
        return None
    p = cfg["pierdicca"]
    return {"map": np.random.default_rng(p["map_seed"]),
            # his Agent(..., seed=0): every agent has its own generator, all with the same seed
            "agents": [np.random.default_rng(p["agent_seed"]) for _ in range(cfg["agents"]["n"])],
            "position": np.random.default_rng(p["position_seed"]),
            "order": np.random.default_rng(p["order_seed"]), "next_run": 0}


def make_scenario(cfg: dict, run: int = 0, streams: dict | None = None) -> Scenario:
    cfg = with_defaults(cfg)
    run_seed = int(cfg["seed"]) + run
    grid = Grid(**cfg["grid"])
    lat = make_lattice(grid, cfg["lattice"])
    pierdicca = cfg["compat"] == "pierdicca"
    if pierdicca:
        if streams is None:  # standalone call: fast-forward field and start generators to this run
            streams = make_streams(cfg)
            for _ in range(run):
                streams["map"].normal(size=grid.shape)
                if cfg["agents"]["start"] == "random":
                    for _ in range(cfg["agents"]["n"]):
                        streams["position"].choice(lat.nx), streams["position"].choice(lat.ny)
            streams["next_run"] = run
        if streams["next_run"] != run:
            raise ValueError(f"compat pierdicca: runs must be built in order (expected {streams['next_run']}, got {run})")
        streams["next_run"] += 1
    elif cfg["compat"] is not None:
        raise ValueError(f"unknown compat '{cfg['compat']}'")

    # ground truth
    f = cfg["field"]
    if f["type"] == "gaussian":
        field_rng = streams["map"] if pierdicca else np.random.default_rng(run_seed)
        truth = gaussian_random_field(f["r"], grid.shape, field_rng)
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
    if pierdicca and a["start"] == "random":  # his reset_agents_position: x then y per agent, seed 12
        pr = streams["position"]
        starts = [Pose(int(pr.choice(np.arange(lat.nx))), int(pr.choice(np.arange(lat.ny))), 0) for _ in range(a["n"])]
    else:
        starts = make_starts(lat, a["n"], a["start"], np.random.default_rng(run_seed))

    n = a["n"]
    if pierdicca:  # per agent: one generator for noise and decisions, carried across runs
        obs_rngs = plan_rngs = streams["agents"]
        order_rng = streams["order"]
    else:
        # agent 0 keeps the legacy single-agent seeding (Field.rng reset to the run seed)
        obs_rngs = [np.random.default_rng(run_seed if i == 0 else [run_seed, 2, i]) for i in range(n)]
        plan_rngs = [np.random.default_rng([run_seed, 1] if i == 0 else [run_seed, 3, i]) for i in range(n)]
        order_rng = np.random.default_rng([run_seed, 4])
    return Scenario(
        name=cfg["name"], run_seed=run_seed, grid=grid, lattice=lat, truth=truth, rates=rates, preds=preds,
        starts=starts, obs_rngs=obs_rngs, plan_rngs=plan_rngs, order_rng=order_rng,
    )


def calibration_field(cfg: dict, grid: Grid) -> np.ndarray:
    """A field like the scenario's for calibrating planner models: a fresh Gaussian field with
    the same r (fixed seed, never one of the evaluated fields), or the annotated map itself."""
    f = with_defaults(cfg)["field"]
    if f["type"] == "gaussian":
        return gaussian_random_field(f["r"], grid.shape, np.random.default_rng([999, 999]))
    return load_annotations(f["annotations"], f.get("crop"), tuple(f.get("positive_labels", (1, 2, 10))))


def mapping_weights(cfg: dict) -> str:
    """run.weights as the mapper should use it (compat pierdicca: his adaptive-weight variant)."""
    cfg = with_defaults(cfg)
    w = cfg["run"]["weights"]
    return "adaptive_pierdicca" if cfg["compat"] == "pierdicca" and w == "adaptive" else w


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
