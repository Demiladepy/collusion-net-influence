"""Positive control: scripted 4-player iterated RPS with one known colluding pair (written during the sprint).

For each collusion probability (cp) and number of rounds, simulate many independent datasets and report:
  CDA    - collusion detection accuracy: the paper's rule returns exactly the true pair
  FPR    - with cp = 0 (no collusion), how often any pair is flagged
  top-1  - the true pair has the highest pair score (net influence vs Cramer's V baseline)
The colluding pair is drawn at random per dataset so no agent index is privileged.

    python positive_control.py --iters 200 --out results/positive_control.json
"""
import argparse
import json
import math
import os
from itertools import combinations

import numpy as np

from netinfluence import detect, pair_scores
from rps import ACTIONS, play_scripted, prev_joint_states


def cramers_v(x, y):
    """Plain action-correlation baseline (no conditioning, no subtraction of other agents)."""
    idx = {a: k for k, a in enumerate(ACTIONS)}
    table = np.zeros((3, 3))
    for a, b in zip(x, y):
        table[idx[a], idx[b]] += 1
    n = table.sum()
    expected = table.sum(1, keepdims=True) * table.sum(0, keepdims=True) / n
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = np.nansum((table - expected) ** 2 / expected)
    return math.sqrt(chi2 / (n * 2))


def run_cell(cp, n_rounds, iters, n_agents, state_mode, alpha, rng):
    hits = flagged_any = ni_top1 = cv_top1 = 0
    for _ in range(iters):
        pair = tuple(sorted(rng.choice(n_agents, 2, replace=False))) if cp > 0 else None
        acts = play_scripted(n_rounds, n_agents, pair=pair, cp=cp, rng=rng)
        states = prev_joint_states(acts) if state_mode == "prev" else None
        found, _, net = detect(acts, states, alpha)
        flagged_any += found is not None
        if pair is None:
            continue
        hits += found == pair
        ni = pair_scores(net)
        ni_top1 += max(ni, key=ni.get) == pair
        cv = {(i, j): cramers_v(acts[:, i], acts[:, j]) for i, j in combinations(range(n_agents), 2)}
        cv_top1 += max(cv, key=cv.get) == pair
    row = {"cp": cp, "rounds": n_rounds, "iters": iters, "state": state_mode}
    if cp > 0:
        row |= {"cda": hits / iters, "ni_top1": ni_top1 / iters, "cramers_v_top1": cv_top1 / iters}
    else:
        row |= {"fpr": flagged_any / iters}
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--agents", type=int, default=4)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/positive_control.json")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    rounds = [25, 50, 100, 200, 400, 800]
    rows = []
    for state_mode in ("none", "prev"):
        for cp in (0.0, 0.2, 0.4, 0.6, 1.0):
            for n in rounds:
                row = run_cell(cp, n, args.iters, args.agents, state_mode, args.alpha, rng)
                rows.append(row)
                print(json.dumps(row), flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"args": vars(args), "rows": rows}, f, indent=1)


if __name__ == "__main__":
    main()
