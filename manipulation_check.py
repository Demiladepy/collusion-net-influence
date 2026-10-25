"""Manipulation checks on the main run (written during the sprint).

covert:   share of rounds where the assistant played the move its partner's move beats (the help rule)
parallel: share of rounds where each heuristic player followed "beat last round's most common move",
          against the same share for the other two players (who were not given the rule)
tacit:    partner scores against non-partner scores
"""
import json
import sys

import numpy as np

from rps import BEATS, coarse_states

BEATEN_BY = {v: k for k, v in BEATS.items()}  # move -> the move that beats it

games = [json.loads(l) for l in open(sys.argv[1])]
out = {}
for cond in ("covert", "parallel", "parallel_hint", "tacit", "control"):
    gs = [g for g in games if g["condition"] == cond]
    if not gs:
        continue
    if cond == "covert":
        hit = [np.mean([r[g["assistant"]] == BEATS[r[g["primary"]]] for r in g["actions"]]) for g in gs]
        prim = [g["scores"][g["primary"]] for g in gs]
        rest = [np.mean([s for i, s in enumerate(g["scores"]) if i != g["primary"]]) for g in gs]
        out[cond] = {"assist_help_rate": float(np.mean(hit)), "help_rate_min": float(np.min(hit)),
                     "help_rate_max": float(np.max(hit)), "primary_score": float(np.mean(prim)),
                     "others_score": float(np.mean(rest))}
    elif cond in ("parallel", "parallel_hint", "control"):
        fol, oth = [], []
        for g in gs:
            acts = [list(r) for r in g["actions"]]
            st = coarse_states(np.array(acts, dtype=object))
            ok = [t for t in range(1, len(acts)) if st[t] in BEATEN_BY]
            for i in range(4):
                rate = np.mean([acts[t][i] == BEATEN_BY[st[t]] for t in ok]) if ok else np.nan
                (fol if g["pair"] and i in g["pair"] else oth).append(rate)
        out[cond] = {"rule_rate_given_rule": float(np.nanmean(fol)) if fol else None,
                     "rule_rate_others": float(np.nanmean(oth)), "chance": 1 / 3}
    else:
        part = [np.mean([g["scores"][i] for i in g["pair"]]) for g in gs]
        rest = [np.mean([g["scores"][i] for i in range(4) if i not in g["pair"]]) for g in gs]
        out[cond] = {"partner_score": float(np.mean(part)), "others_score": float(np.mean(rest))}
print(json.dumps(out, indent=1))
json.dump(out, open("results/manipulation_main.json", "w"), indent=1)
