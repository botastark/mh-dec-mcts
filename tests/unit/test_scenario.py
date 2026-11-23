from pathlib import Path

import numpy as np
import pytest

from mh_dec_mcts.scenario import load_config, make_lattice, make_scenario, make_starts, with_defaults
from mh_dec_mcts.sensing.lattice import Pose
from mh_dec_mcts.sensing.sensor import save_predictions
from mh_dec_mcts.world.grid import Grid

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "configs" / "paper.yaml"


def small(**over):
    cfg = {"name": "t", "grid": {"size_x": 20, "size_y": 10, "cell": 0.25}, "lattice": {"xy_step": 2.5},
           "field": {"type": "gaussian", "r": 2}}
    for k, v in over.items():
        cfg[k] = {**cfg.get(k, {}), **v} if isinstance(v, dict) else v
    return cfg


# -------------------------------------------------------------------- config
def test_paper_config_matches_paper():
    cfg = load_config(PAPER)
    sc = make_scenario(cfg)
    assert sc.grid.shape == (400, 400)
    assert (sc.lattice.nx, sc.lattice.ny, sc.lattice.n_alt) == (17, 17, 6)
    assert sc.lattice.altitudes[3] == pytest.approx(21.65, abs=0.01)
    assert sc.starts == [Pose(0, 0, 0)]
    assert cfg["run"] == {"planner": "ig", "weights": "equal", "steps": 300, "runs": 20}


def test_overrides_and_defaults():
    cfg = load_config(PAPER, run={"steps": 7}, field={"r": 2})
    assert cfg["run"]["steps"] == 7 and cfg["run"]["runs"] == 20  # other keys kept
    assert cfg["field"] == {"type": "gaussian", "r": 2}
    d = with_defaults(small())
    assert d["sensor"]["type"] == "theoretical" and d["agents"] == {"n": 1, "start": "paper"}


def test_missing_section_is_an_error():
    with pytest.raises(ValueError, match="field"):
        with_defaults({"grid": {"size_x": 10, "size_y": 10, "cell": 1}})


def test_lattice_null_rules():
    g = Grid(size_x=50, size_y=50, cell=0.125)
    lat = make_lattice(g, {"xy_step": None, "h_min": None, "h_step": None, "n_alt": 6, "fov_deg": 60})
    assert lat.xy_step == 3.125 and lat.h_min == pytest.approx(lat.h_step) == pytest.approx(5.4127, abs=1e-4)
    lat = make_lattice(g, {"xy_step": 5.0, "h_min": 19.5, "h_step": None, "n_alt": 4, "fov_deg": 60})
    assert lat.h_min == 19.5 and lat.h_step == pytest.approx(5.0 / np.tan(np.radians(30)))


# ------------------------------------------------------------------ scenario
def test_gaussian_truth_per_run():
    a, b, c = make_scenario(small(), 0), make_scenario(small(), 0), make_scenario(small(), 1)
    np.testing.assert_array_equal(a.truth, b.truth)
    assert not np.array_equal(a.truth, c.truth)
    assert a.run_seed == 42 and c.run_seed == 43


def test_observations_reproducible_and_independent_of_planning():
    a, b = make_scenario(small(), 0), make_scenario(small(), 0)
    b.plan_rng.random(100)  # planner randomness must not shift the sensor noise
    p = Pose(3, 2, 1)
    np.testing.assert_array_equal(a.observe(p), b.observe(p))


def test_observation_is_footprint_shaped():
    sc = make_scenario(small())
    for p in (Pose(0, 0, 0), Pose(sc.lattice.nx - 1, sc.lattice.ny - 1, 5)):
        i0, i1, j0, j1 = sc.lattice.footprint(p)
        assert sc.observe(p).shape == (i1 - i0, j1 - j0)


def test_start_modes():
    sc = make_scenario(small())
    lat = sc.lattice
    rng = np.random.default_rng(0)
    assert make_starts(lat, 1, "paper", rng) == [Pose(0, 0, 0)]
    assert make_starts(lat, 2, "paper", rng) == [Pose(0, 0, 0), Pose(lat.nx - 1, 0, 0)]
    corners = {Pose(0, 0, 0), Pose(0, lat.ny - 1, 0), Pose(lat.nx - 1, 0, 0), Pose(lat.nx - 1, lat.ny - 1, 0)}
    assert make_starts(lat, 1, "legacy", rng)[0] in corners
    r = make_starts(lat, 5, "random", rng)
    assert len(set(r)) == 5 and all(p.k == 0 for p in r)
    with pytest.raises(ValueError):
        make_starts(lat, 1, "middle", rng)


def test_orthomap_field(tmp_path):
    ann = np.zeros((14, 24), int)
    ann[2:12, 2:22] = np.random.default_rng(0).choice([0, 1, 2, 10], (10, 20))
    np.savetxt(tmp_path / "ann.txt", ann, fmt="%d")
    cfg = small(field={"type": "orthomap", "annotations": str(tmp_path / "ann.txt"), "crop": [2, 12, 2, 22],
                       "labels": {0: 0, 1: 1, 2: 1, 10: 1}},
                grid={"size_x": 20, "size_y": 10, "cell": 1.0}, lattice={"xy_step": 2.0})
    a, b = make_scenario(cfg, 0), make_scenario(cfg, 5)
    np.testing.assert_array_equal(a.truth, b.truth)  # fixed across runs
    np.testing.assert_array_equal(a.truth, np.isin(ann[2:12, 2:22], (1, 2, 10)))


def test_unmapped_annotation_label_is_an_error(tmp_path):
    np.savetxt(tmp_path / "ann.txt", np.array([[0, 1], [2, 10]]), fmt="%d")
    cfg = small(field={"type": "orthomap", "annotations": str(tmp_path / "ann.txt"), "labels": {0: 0, 1: 1, 2: 1}},
                grid={"size_x": 2, "size_y": 2, "cell": 1.0}, lattice={"xy_step": 1.0})
    with pytest.raises(ValueError, match=r"labels \[10\] have no meaning"):
        make_scenario(cfg)


def test_field_shape_mismatch_is_an_error(tmp_path):
    np.savetxt(tmp_path / "ann.txt", np.zeros((5, 5), int), fmt="%d")
    cfg = small(field={"type": "orthomap", "annotations": str(tmp_path / "ann.txt")})
    with pytest.raises(ValueError, match="does not match grid"):
        make_scenario(cfg)


def test_predictions_sensor(tmp_path):
    base = make_scenario(small())
    preds = np.stack([1 - base.truth] * base.lattice.n_alt).astype(np.uint8)  # always wrong
    path = tmp_path / "pred.npz"
    save_predictions(path, preds, base.lattice.altitudes, truth=base.truth)
    sc = make_scenario(small(sensor={"type": "predictions", "file": str(path)}))
    p = Pose(1, 1, 2)
    i0, i1, j0, j1 = sc.lattice.footprint(p)
    np.testing.assert_array_equal(sc.observe(p), 1 - sc.truth[i0:i1, j0:j1])
    assert sc.rates.s0[0] == pytest.approx(0.999) and sc.rates.s1[0] == pytest.approx(0.999)


def test_unknown_types_are_errors():
    with pytest.raises(ValueError, match="field"):
        make_scenario(small(field={"type": "lidar"}))
    with pytest.raises(ValueError, match="sensor"):
        make_scenario(small(sensor={"type": "magic"}))
