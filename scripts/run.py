"""One runner for every experiment: a grid of conditions, one result format (results.json).

    # planner comparison in one swarm setting (sharing, range and greedy = IG / IGd from the config)
    python scripts/run.py configs/swarm_rect.yaml --setting IGd_BM_R5 --n 4 \\
        --planners greedy decmcts:horizon=4,rollout=greedy mhmcts:horizon=4,rollout=greedy
    # Pierdicca Fig. 3 (one UAV, Bayes only) and Fig. 4 (one weights / r per call)
    python scripts/run.py configs/pierdicca_fig3.yaml --planners ig random sweep --save-poses
    python scripts/run.py configs/pierdicca_fig4.yaml --planners ig --weights adaptive --r 2
    # Pierdicca Fig. 5: planners x sharing x ranges x team sizes
    python scripts/run.py configs/pierdicca_fig5.yaml --planners ig igd --sharing none BS BM \\
        --ranges 5 16 --n 1 2 4 8

Grid = planners x n x (one --setting, or --sharing x --ranges). With one UAV there is nothing to
share: only sharing "none" (else the first listed) and the first range are run. Planner specs:
greedy (the setting's IG / IGd), ig, igd, random, sweep, decmcts[:key=value,...],
mhmcts[:key=value,...] (keys of MHConfig go to `mh`, the others to `mcts`).

Runs: run i uses the scenario of seed + i for every planner (paired). Each (condition, run) is a
job of its own, submitted run by run. compat: pierdicca (his generators carry over from run to
run): one job per condition, which runs 0, 1, 2, ... in order with one make_streams object.

Output in <out>/<tag>/ (tag default: config name [_r<r>] _<weights> [_<setting>]):
    results.json   every run's curves and scalars (docs/results_format.md)
    results.npz    with --save-poses: poses, first-seen maps and ground truth per run
    results.txt    the tables printed at the end: mean +- std over runs; the difference to the
                   first planner per run (paired) +- s.e.
Figures: scripts/plot_mission.py, plot_fig3.py, plot_fig4.py, plot_fig5.py.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")  # one maths thread per process, before numpy is imported

import argparse
import itertools
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from mh_dec_mcts.experiment import condition_id, make_meta, one_run, run_record, save_results
from mh_dec_mcts.scenario import load_config, make_streams

MAPS = ("first", "truth")  # large per-run arrays, kept only with --save-poses


# ------------------------------------------------------------------ grid
def grid(cfg: dict, planners: list[str], ns: list[int], setting: str | None,
         sharings: list[str], ranges: list) -> list[dict]:
    """Conditions in order planner, sharing, range, n (as the former Fig. 5 runner)."""
    if setting is not None:
        s = cfg["settings"][setting]
        combos, greedy, one = [(s["sharing"], s["range"])], s["greedy"], None
    else:
        combos, greedy = list(itertools.product(sharings, ranges)), None
        one = ("none" if "none" in sharings else sharings[0], ranges[0])  # n = 1: nothing to share
    out = []
    for spec, (sharing, range_), n in itertools.product(planners, combos, ns):
        if n == 1 and one is not None and (sharing, range_) != one:
            continue
        if spec == "greedy" and greedy is None:
            raise SystemExit("planner 'greedy' needs --setting (its IG / IGd)")
        out.append({"id": condition_id(spec, n, sharing, range_, setting), "planner": spec, "n": n,
                    "sharing": sharing, "range": range_, "setting": setting, "greedy": greedy})
    return out


def condition_cfg(config: str, over: dict, c: dict) -> dict:
    agents = {**over.get("agents", {}), "n": c["n"]}
    return load_config(config, **{**over, "agents": agents,
                                  "comms": {"range": c["range"], "sharing": c["sharing"]}})


# ------------------------------------------------------------------ jobs (top level: picklable)
def _keep(r: dict, save: bool) -> dict:
    return r if save else {k: v for k, v in r.items() if k not in MAPS}


def job_run(cfg, setting, spec, run, steps, late, stall_rate, save) -> dict:
    return _keep(one_run(cfg, setting, spec, run, steps, late, stall_rate), save)


def job_compat(cfg, setting, spec, runs, steps, late, stall_rate, save) -> list[dict]:
    streams = make_streams(cfg)  # one experiment = fresh generators, carried from run to run
    return [_keep(one_run(cfg, setting, spec, i, steps, late, stall_rate, streams), save)
            for i in range(runs)]


# ------------------------------------------------------------------ progress
def _hms(sec: float) -> str:
    sec = round(sec)
    return f"{sec // 3600}h{sec // 60 % 60:02d}m" if sec >= 3600 else f"{sec // 60}m{sec % 60:02d}s"


def progress(done: dict, key: tuple[str, int], todo: list, jobs: int, elapsed: float,
             trap: int) -> str:
    """One line per finished run: count, elapsed, ETA, and the run's numbers. The ETA spreads the
    remaining runs over the processes, each costing its planner's mean wall time per run so far
    (planners differ a lot: H = 16 is ~5x H = 4). A planner with no finished run yet is costed at
    the overall mean, which can be far too low, so it is named after the ETA as untimed."""
    cost = {}  # planner -> wall times of its finished runs (one run = one process)
    for (s, _), r in done.items():
        cost.setdefault(s, []).append(r["wall"])
    mean_all = float(np.mean([c for cs in cost.values() for c in cs]))
    left = [k for k in todo if k not in done]
    eta = sum(np.mean(cost[s]) if s in cost else mean_all for s, _ in left) / max(1, min(jobs, len(left)))
    untimed = sorted({s for s, _ in left} - set(cost))
    if len(untimed) > 3:
        untimed = [*untimed[:3], f"... ({len(untimed)} in all)"]
    r = done[key]
    stall = f"stall {r['stall']:3d}" + (" STALLED" if r["stall"] >= trap else "        ")
    return (f"[{len(done):3d}/{len(todo)} {100 * len(done) / len(todo):3.0f}%  {_hms(elapsed)} elapsed,"
            f" ~{_hms(eta)} left{' + untimed: ' + ', '.join(untimed) if untimed else ''}]  {key[0]} run"
            f" {key[1]:2d}: avg MSE {np.mean(r['mse']):.4f}, final {r['mse'][-1]:.4f}, {stall},"
            f" {r['cpu']:.2f} s/step")


# ------------------------------------------------------------------ tables
def tables(conditions: list[dict], runs: int, steps: int, trap: int, name: str) -> list[str]:
    """The planner-comparison tables, one block per team setting (n, sharing, range): mean +- std
    over runs; the avg-MSE difference to the block's first planner, per run (paired) +- s.e."""
    warnings.filterwarnings("ignore", category=RuntimeWarning)  # nan when a run never reaches a level
    out = []
    groups: dict[tuple, list[dict]] = {}
    for c in conditions:
        groups.setdefault((c["n"], c["sharing"], c["range"], c["setting"]), []).append(c)
    for (n, sharing, range_, setting), cs in groups.items():
        res = {c["planner"]: [{**r["curves"], **r["scalars"]} for r in c["runs"]] for c in cs}
        first = cs[0]["planner"]
        label = setting if setting is not None else f"sharing {sharing}, range {range_}"
        out.append(f"{name} {label}, {n} UAV{'s' if n > 1 else ''}: {runs} runs x {steps} steps (mean +- std"
                   f" over runs; vs the first planner: paired difference +- s.e.; nan = never reached)")
        ref = np.array([np.mean(r["mse"]) for r in res[first]])
        out.append(f"{'planner':>36s} {'steps to 80% cov':>17s} {'to 95%':>8s} {'backfill %':>13s}"
                   f" {'avg MSE':>9s} {'vs ' + first + ' (paired)':>26s} {'final MSE':>10s} {'mean alt':>8s}"
                   f" {'stall':>11s} {'stalled':>8s} {'incomplete':>10s} {'plan s/step':>16s}")
        for s, rs in res.items():
            cov = np.array([r["coverage"] for r in rs])
            t80 = np.array([np.argmax(c >= 0.8) if c[-1] >= 0.8 else np.nan for c in cov])
            t95 = np.array([np.argmax(c >= 0.95) if c[-1] >= 0.95 else np.nan for c in cov])
            bf = np.array([r["backfill"][-1] for r in rs]) * 100
            sd = np.nanstd  # spread over the runs
            se = lambda a: np.nanstd(a) / np.sqrt(len(a))  # only for the paired difference
            mse = np.array([r["mse"] for r in rs])
            d = mse.mean(1) - ref
            st = [r["stall"] for r in rs]
            pt = [r["plan_per_step"] for r in rs]
            out.append(f"{s:>36s} {np.nanmean(t80):9.0f} +- {sd(t80):3.0f} {np.nanmean(t95):6.0f}"
                       f" {bf.mean():7.1f} +- {sd(bf):3.1f} {mse.mean():9.4f}"
                       f" {d.mean():+.4f} +- {se(d):.4f} ({100 * d.mean() / ref.mean():+5.1f}%)"
                       f" {mse[:, -1].mean():10.5f} {np.mean([r['altitude'] for r in rs]):7.1f}m"
                       f" {np.mean(st):5.1f} +- {sd(np.array(st)):3.1f}"
                       f" {sum(x >= trap for x in st):5d}/{len(st)}"
                       f" {sum(r['coverage'][-1] < 0.95 for r in rs):7d}/{len(rs)}"
                       f" {np.mean(pt):8.3f} +- {sd(np.array(pt)):.3f}")
        out.append("")
        out.append(f"{'avg MSE per run':>36s} {'median':>8s} {'std':>8s} {'90th pct':>9s} {'worst':>8s}"
                   f" {'stall >= trap':>14s} {'avg MSE of stalled runs':>24s}")
        for s, rs in res.items():
            m = np.array([np.mean(r["mse"]) for r in rs])
            stalled = np.array([r["stall"] >= trap for r in rs])
            out.append(f"{s:>36s} {np.median(m):8.4f} {m.std():8.4f} {np.percentile(m, 90):9.4f}"
                       f" {m.max():8.4f} {stalled.sum():10d}/{len(rs)}"
                       f" {(m[stalled].mean() if stalled.any() else np.nan):24.4f}")
        if n > 1:
            out.append("")
            out.append(f"{'team':>36s} {'overlap':>14s} {'connectivity':>14s} {'misalignment':>14s}"
                       f" {'balance':>12s}")
            for s, rs in res.items():
                cols = [[np.mean(r[k]) for r in rs] for k in ("overlap", "connected", "misalignment")]
                bal = [r["balance"] for r in rs]
                out.append(f"{s:>36s} " + " ".join(f"{np.mean(c):7.3f} +-{np.std(c):5.3f}" for c in cols)
                           + f" {np.mean(bal):5.2f} +-{np.std(bal):4.2f}")
        out.append("")
    return out


# ------------------------------------------------------------------ main
def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("config")
    ap.add_argument("--planners", nargs="+", required=True, metavar="SPEC")
    ap.add_argument("--n", nargs="+", type=int, help="team sizes (default: the config's)")
    ap.add_argument("--setting", help="a team setting of the config's `settings:` (sharing, range, greedy)")
    ap.add_argument("--sharing", nargs="+", choices=["none", "BS", "BM"], help="default: the config's")
    ap.add_argument("--ranges", nargs="+", type=int, help="comms ranges in waypoint steps (default: the config's)")
    ap.add_argument("--runs", type=int, help="default: the config's run.runs")
    ap.add_argument("--steps", type=int, help="default: the config's run.steps")
    ap.add_argument("--weights", choices=["none", "equal", "biased", "adaptive"], help="override run.weights")
    ap.add_argument("--r", type=float, help="override field.r (Gaussian cluster radius)")
    ap.add_argument("--start", choices=["paper", "legacy", "random"], help="override agents.start")
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--late", type=int, default=10, help="steps after which a first sighting counts as backfill")
    ap.add_argument("--stall-rate", type=float, default=0.01,
                    help="entropy drop per step, as a share of what is left, below which a step stalls")
    ap.add_argument("--trap", type=int, default=15, help="stall steps in a row that count as a stalled run")
    ap.add_argument("--save-poses", action="store_true",
                    help="also results.npz: poses, first-seen maps, ground truth (trajectory figures)")
    ap.add_argument("--out", default="results/runs")
    ap.add_argument("--tag", help="output folder name (default: config name [_r<r>] _<weights> [_<setting>])")
    return ap


def main(argv: list[str] | None = None) -> Path:
    args = parser().parse_args(argv)
    if args.setting and (args.sharing or args.ranges):
        raise SystemExit("--setting fixes sharing and range: do not combine it with --sharing / --ranges")
    over: dict = {"run": {k: v for k, v in (("weights", args.weights), ("runs", args.runs),
                                             ("steps", args.steps)) if v}}
    if args.start:
        over["agents"] = {"start": args.start}
    if args.r is not None:
        over["field"] = {"r": args.r}
    cfg = load_config(args.config, **over)
    runs, steps = cfg["run"]["runs"], cfg["run"]["steps"]
    conds = grid(cfg, args.planners, args.n or [cfg["agents"]["n"]], args.setting,
                 args.sharing or [cfg["comms"]["sharing"]], args.ranges or [cfg["comms"]["range"]])
    compat = cfg["compat"] == "pierdicca"
    tag = args.tag or (cfg["name"] + (f"_r{cfg['field']['r']:g}" if "r" in cfg["field"] else "")
                       + f"_{cfg['run']['weights']}" + (f"_{args.setting}" if args.setting else ""))
    out = Path(args.out) / tag
    print(f"{cfg['name']}: {len(conds)} conditions x {runs} runs x {steps} steps on {args.jobs} processes"
          + (" (compat: runs of a condition in order)" if compat else ""), flush=True)

    ccfgs = [condition_cfg(args.config, over, c) for c in conds]
    settings = [{"sharing": c["sharing"], "range": c["range"], "greedy": c["greedy"]} for c in conds]
    rest = (steps, args.late, args.stall_rate, args.save_poses)
    done: dict[tuple[str, int], dict] = {}
    todo = [(c["id"], i) for i in range(runs) for c in conds]  # run by run: every planner timed early
    t0 = time.time()
    with ProcessPoolExecutor(args.jobs) as ex:
        if compat:
            futs = {ex.submit(job_compat, ccfgs[k], settings[k], c["planner"], runs, *rest): k
                    for k, c in enumerate(conds)}
            for fut in as_completed(futs):
                c = conds[futs[fut]]
                for i, r in enumerate(fut.result()):
                    done[(c["id"], i)] = r
                    print(progress(done, (c["id"], i), todo, args.jobs, time.time() - t0, args.trap), flush=True)
        else:
            index = {c["id"]: k for k, c in enumerate(conds)}
            futs = {ex.submit(job_run, ccfgs[index[cid]], settings[index[cid]], conds[index[cid]]["planner"],
                              i, *rest): (cid, i) for cid, i in todo}
            for fut in as_completed(futs):
                key = futs[fut]
                done[key] = fut.result()
                print(progress(done, key, todo, args.jobs, time.time() - t0, args.trap), flush=True)

    arrays = {}
    for k, c in enumerate(conds):
        c["steps"] = steps
        c["runs"] = []
        for i in range(runs):
            r = done[(c["id"], i)]
            c["runs"].append(run_record(r, i, int(ccfgs[k]["seed"]) + i))
            if args.save_poses:
                arrays[f"c{k}_r{i}_poses"] = np.array([[(p.ix, p.iy, p.k) for p in path] for path in r["poses"]])
                arrays[f"c{k}_r{i}_first"] = r["first"]
                arrays.setdefault(f"truth_r{i}", r["truth"])
    meta = {**make_meta(args.config, cfg, sys.argv if argv is None else ["scripts/run.py", *argv]),
            "runs": runs, "steps": steps, "late": args.late, "stall_rate": args.stall_rate, "trap": args.trap,
            "setting": args.setting, "compat": cfg["compat"]}
    save_results(out, meta, conds, arrays)
    lines = tables(conds, runs, steps, args.trap, cfg["name"])
    print("\n".join(lines))
    (out / "results.txt").write_text("\n".join(lines) + "\n")
    print(f"-> {out}")
    return out


if __name__ == "__main__":
    main()
