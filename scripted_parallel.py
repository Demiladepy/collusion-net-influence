"""Exploratory (post-hoc, see PREREGISTRATION.md change 4): scripted innocent parallel conduct.

Two non-colluding players each follow the public rule "beat last round's most common move" with
probability q (independently), otherwise play at random; the other two play at random. Any flag is a
false alarm. Written during the sprint, after the LLM players ignored the rule (36% vs 33% chance).
Thresholds are the same 5%-false-alarm calibrations used in analyze.py.

    python scripted_parallel.py --iters 500 --out results/scripted_parallel.json
"""
import argparse
import json
from itertools import combinations

import numpy as np

from analyze import flag_at, thresholds, wilson
from netinfluence import detect, pair_scores
from positive_control import cramers_v
from rps import ACTIONS, BEATS, coarse_states

BEATEN_BY = {v: k for k, v in BEATS.items()}


def play_parallel(rounds, q, pair, rng):
    acts = []
    for t in range(rounds):
        row = list(rng.choice(ACTIONS, 4))
        if t > 0:
            s = coarse_states(np.array(acts[-1:] + [row], dtype=object))[1]  # most common move last round
            if s in BEATEN_BY:
                for i in pair:
                    if rng.random() < q:
                        row[i] = BEATEN_BY[s]
        acts.append(row)
    return np.array(acts, dtype=object)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=60)
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--out", default="results/scripted_parallel.json")
    args = ap.parse_args()
    thr = thresholds(args.rounds)
    rng = np.random.default_rng(3)
    rows = []
    for q in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        fa = {"ni_none_cal": 0, "ni_coarse_cal": 0, "cv_cal": 0}
        for _ in range(args.iters):
            pair = tuple(sorted(rng.choice(4, 2, replace=False)))
            a = play_parallel(args.rounds, q, pair, rng)
            _, _, net0 = detect(a, None)
            _, _, net1 = detect(a, coarse_states(a))
            fa["ni_none_cal"] += flag_at(net0, thr["ni_none"]) is not None
            fa["ni_coarse_cal"] += flag_at(net1, thr["ni_coarse"]) is not None
            cv = {(i, j): cramers_v(a[:, i], a[:, j]) for i, j in combinations(range(4), 2)}
            top = max(cv, key=cv.get)
            fa["cv_cal"] += cv[top] > thr["cv"]
        row = {"q": q, "iters": args.iters}
        for d, k in fa.items():
            p, lo, hi = wilson(k, args.iters)
            row[d] = {"false_alarm": p, "lo": lo, "hi": hi}
        rows.append(row)
        print(json.dumps({"q": q, **{d: round(v["false_alarm"], 3) for d, v in row.items() if isinstance(v, dict)}}), flush=True)
    with open(args.out, "w") as f:
        json.dump({"thresholds": thr, "rows": rows}, f, indent=1)


if __name__ == "__main__":
    main()
