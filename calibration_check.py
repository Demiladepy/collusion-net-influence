"""Scripted check of estimator calibration at LLM-game length (written during the sprint).

At T rounds, compare plug-in vs bias-corrected net influence for each state choice on scripted RPS:
false-alarm rate with no collusion (cp = 0) and correct detection with assistant collusion (cp = 0.6,
about what the LLM assistant achieved in pilot 2). States are drawn from the game itself.

    python calibration_check.py --rounds 60 --iters 300 --out results/calibration.json
"""
import argparse
import json

import numpy as np

from netinfluence import detect
from rps import coarse_states, play_scripted, prev_joint_states

STATES = {"none": lambda a: None, "coarse": coarse_states, "prev": prev_joint_states}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=60)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--perm", type=int, default=30)
    ap.add_argument("--out", default="results/calibration.json")
    args = ap.parse_args()
    rng = np.random.default_rng(1)
    rows = []
    for state_name, fn in STATES.items():
        for est, n_perm in (("plugin", 0), ("corrected", args.perm)):
            fa = hit = 0
            for _ in range(args.iters):
                a0 = play_scripted(args.rounds, 4, pair=None, rng=rng)
                fa += detect(a0, fn(a0), 0.05, n_perm, rng)[0] is not None
                pair = tuple(sorted(rng.choice(4, 2, replace=False)))
                a1 = play_scripted(args.rounds, 4, pair=pair, cp=0.6, rng=rng)
                hit += detect(a1, fn(a1), 0.05, n_perm, rng)[0] == pair
            row = {"state": state_name, "estimator": est, "rounds": args.rounds,
                   "false_alarm": fa / args.iters, "detection_cp0.6": hit / args.iters, "iters": args.iters}
            rows.append(row)
            print(json.dumps(row), flush=True)
    with open(args.out, "w") as f:
        json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
