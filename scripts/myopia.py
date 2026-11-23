"""Old entry point of planner comparisons, kept for existing commands: runs scripts/run.py with
myopia.py's defaults (setting IGd_BM_R5, out results/myopia, tag <config name>_<setting>,
poses saved for the first-seen figure).

    python scripts/myopia.py configs/swarm_rect.yaml --setting IGd_BM_R5 --n 4 \\
        --planners greedy decmcts:horizon=4,rollout=greedy
    (same as: python scripts/run.py configs/swarm_rect.yaml --setting IGd_BM_R5 --n 4 \\
        --out results/myopia --tag swarm_rect_IGd_BM_R5 --save-poses --planners ...)
"""

import sys

from run import main as run_main

from mh_dec_mcts.scenario import load_config


def main(argv: list[str]) -> None:
    def opt(name, default=None):
        return argv[argv.index(name) + 1] if name in argv else default

    setting = opt("--setting", "IGd_BM_R5")
    extra = [] if "--setting" in argv else ["--setting", setting]
    extra += [] if "--out" in argv else ["--out", "results/myopia"]
    extra += [] if "--tag" in argv else ["--tag", f"{load_config(argv[0])['name']}_{setting}"]
    extra += [] if "--planners" in argv else ["--planners", "greedy", "decmcts:horizon=2", "decmcts:horizon=8"]
    run_main([*argv, *extra, "--save-poses"])


if __name__ == "__main__":
    main(sys.argv[1:])
