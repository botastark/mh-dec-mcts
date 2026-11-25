"""Figures from saved results (run.py's results.json + results.npz): nothing is rerun.

    python scripts/plot.py mission RESULT                 # per team setting: mission.png, runs.png, firstseen.png
    python scripts/plot.py firstseen RESULT --run 3       # first-seen map + paths of any run (default: worst)
    python scripts/plot.py curves RESULT --keys mse coverage_paper team_entropy --planners greedy mhmcts:horizon=4
    python scripts/plot.py uavs RESULT --planner greedy --n 4         # each UAV's own MSE / entropy / altitude
    python scripts/plot.py paired RESULT [RESULT ...]     # planner - first planner per run: per step + summary
    python scripts/plot.py fig3 RESULT                  # Pierdicca Fig. 3 (one UAV: IG, Random, Sweep)
    python scripts/plot.py fig4 FOLDER [FOLDER ...]       # Pierdicca Fig. 4 (results per weights and r)
    python scripts/plot.py fig5 RESULT [--planners ig igd]          # Pierdicca Fig. 5
    python scripts/plot.py tune TUNING [TUNING ...]       # scripts/tune.py results side by side

RESULT is a run.py output folder (old planner-comparison folders with runs.json work for
mission / runs / curves without per-run data). Figures go to <first RESULT>/figures/ (or --out).
Condition filters, where it makes sense: --planners, --n, --sharing, --range.
Curves are mean +- 1 std over runs (each run averaged over its UAVs), as in Pierdicca et al.
"""

import argparse
import sys
from pathlib import Path

from mh_dec_mcts.experiment import CURVES, load_arrays, load_results
from mh_dec_mcts.plots import explore, mission, paired, paper, tune


def select(res: dict, args) -> list[tuple[int, dict]]:
    """(index, condition) of the result's conditions that pass the filters."""
    out = []
    for k, c in enumerate(res["conditions"]):
        if getattr(args, "planners", None) and c["planner"] not in args.planners:
            continue
        if getattr(args, "n", None) and c["n"] not in args.n:
            continue
        if getattr(args, "sharing", None) and c["sharing"] not in args.sharing:
            continue
        if getattr(args, "range", None) and c["range"] not in args.range:
            continue
        out.append((k, c))
    if not out:
        raise SystemExit("no condition of the result passes the filters")
    return out


def outdir(args, first: Path) -> Path:
    out = Path(args.out) if args.out else Path(first) / "figures"
    out.mkdir(parents=True, exist_ok=True)
    return out


def groups_of(res: dict, chosen: list[tuple[int, dict]]) -> dict[tuple, list[tuple[int, dict]]]:
    ids = {k for k, _ in chosen}
    return {g: [(k, c) for k, c in kc if k in ids] for g, kc in mission.groups(res).items()
            if any(k in ids for k, _ in kc)}


def cmd_mission(args):
    res, arr, out = load_results(args.result), load_arrays(args.result), outdir(args, args.result)
    gs = groups_of(res, select(res, args))
    for g, kc in gs.items():
        suffix = "" if len(gs) == 1 else f"_n{g[0]}_{g[1]}_R{g[2]}"
        cs = [c for _, c in kc]
        t = mission.title(res, g)
        mission.curves(cs, out / f"mission{suffix}.png", f"{t}: {len(cs[0]['runs'])} runs")
        mission.runs(cs, out / f"runs{suffix}.png", f"{t}: one dot per run")
        if arr:
            run = mission.worst_run(cs[0])
            mission.first_seen(kc, arr, res["meta"]["config_resolved"], run, out / f"firstseen{suffix}.png",
                               cs[0]["steps"], " (worst for the first planner)")
        else:
            print("no results.npz: firstseen skipped")
    return out


def cmd_firstseen(args):
    res, arr, out = load_results(args.result), load_arrays(args.result), outdir(args, args.result)
    if not arr:
        raise SystemExit("no results.npz next to the results")
    for g, kc in groups_of(res, select(res, args)).items():
        cs = [c for _, c in kc]
        run = args.run if args.run is not None else mission.worst_run(cs[0])
        why = "" if args.run is not None else " (worst for the first planner)"
        mission.first_seen(kc, arr, res["meta"]["config_resolved"], run,
                           out / f"firstseen_n{g[0]}_{g[1]}_R{g[2]}_run{run}.png", cs[0]["steps"], why)
    return out


def cmd_curves(args):
    res, out = load_results(args.result), outdir(args, args.result)
    cs = [c for _, c in select(res, args)]
    explore.curves(cs, args.keys, out / f"curves_{'_'.join(args.keys)}.png", mission.title(
        res, (cs[0]["n"], cs[0]["sharing"], cs[0]["range"], cs[0]["setting"])), args.log or [])
    return out


def cmd_uavs(args):
    res, arr, out = load_results(args.result), load_arrays(args.result), outdir(args, args.result)
    if not arr:
        raise SystemExit("no results.npz next to the results")
    for k, c in select(res, args):
        g = (c["n"], c["sharing"], c["range"], c["setting"])
        name = c["planner"].replace(":", "_").replace(",", "_").replace("=", "")
        explore.uavs(c, k, arr, out / f"uavs_{name}_n{c['n']}_{c['sharing']}_R{c['range']}.png",
                     f"{c['planner']}, {mission.title(res, g)}")
    return out


def cmd_paired(args):
    out = outdir(args, args.results[0])
    blocks = [b for p in args.results for b in paired.compare(load_results(p), args.split)]
    rs = paired.rows(blocks)
    txt = paired.table(rs)
    print(txt)
    (out / "paired.txt").write_text(txt + "\n")
    paired.curves(blocks, out / "paired_curves.png", args.keys)
    paired.forest(rs, out / "paired_summary.png")
    return out


def cmd_fig3(args):
    res, arr = load_results(args.result), load_arrays(args.result)
    out = outdir(args, args.result)
    paper.fig3(res, arr, out / "paper_fig3.png", args.traj)
    return out


def cmd_fig4(args):
    out = outdir(args, args.paths[0])
    data = paper.fig4_load(args.paths, args.planner)
    if not data:
        raise SystemExit("no results for this planner with weights and r in their configs")
    missing = paper.fig4(data, out / "paper_fig4.png")
    if missing:
        print(f"missing (weights, r), not plotted: {missing}")
    return out


def cmd_fig5(args):
    res, out = load_results(args.result), outdir(args, args.result)
    paper.fig5(res, out / "paper_fig5.png", args.planners)
    return out


def cmd_tune(args):
    out = outdir(args, args.folders[0])
    tune.tune(args.folders, out / "tune.png")
    return out


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add(name, fn, help_, result=True, filters=True):
        p = sub.add_parser(name, help=help_)
        if result:
            p.add_argument("result", type=Path, help="a run.py output folder")
        p.add_argument("--out", help="figure folder (default: <result>/figures)")
        if filters:
            p.add_argument("--planners", nargs="+", metavar="SPEC")
            p.add_argument("--n", nargs="+", type=int)
            p.add_argument("--sharing", nargs="+")
            p.add_argument("--range", nargs="+", type=int)
        p.set_defaults(fn=fn)
        return p

    add("mission", cmd_mission, "mission / team curves, one dot per run, first-seen map (worst run)")
    add("firstseen", cmd_firstseen, "first-seen map + paths of one run").add_argument("--run", type=int)
    p = add("curves", cmd_curves, "any stored curves, mean +- std over runs")
    p.add_argument("--keys", nargs="+", required=True, help=f"stored curves: {', '.join(CURVES)}")
    p.add_argument("--log", nargs="+", help="keys drawn on a log axis")
    add("uavs", cmd_uavs, "each UAV's own MSE / entropy / altitude").add_argument(
        "--planner", dest="planners", nargs="+", metavar="SPEC")
    p = add("paired", cmd_paired, "each planner minus the first on the same runs, several results side by side",
            result=False, filters=False)
    p.add_argument("results", nargs="+", type=Path, help="run.py output folders")
    p.add_argument("--keys", nargs="+", default=list(paired.CURVE_KEYS), help="curves to difference per step")
    p.add_argument("--split", type=int, help="step splitting before / after (default: reference's mean t95)")
    add("fig3", cmd_fig3, "Pierdicca Fig. 3", filters=False).add_argument("--traj", nargs=2, metavar="PLANNER")
    p = add("fig4", cmd_fig4, "Pierdicca Fig. 4", result=False, filters=False)
    p.add_argument("paths", nargs="+", type=Path, help="results per weights and r, or folders holding them")
    p.add_argument("--planner", default="ig")
    add("fig5", cmd_fig5, "Pierdicca Fig. 5", filters=False).add_argument(
        "--planners", nargs="+", help="one column each (default: every planner)")
    p = add("tune", cmd_tune, "scripts/tune.py results", result=False, filters=False)
    p.add_argument("folders", nargs="+", type=Path)
    return ap


def main(argv: list[str] | None = None) -> Path:
    args = parser().parse_args(argv)
    out = args.fn(args)
    print(f"-> {out}")
    return out


if __name__ == "__main__":
    main(sys.argv[1:])
