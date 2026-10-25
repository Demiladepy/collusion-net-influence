"""N-player iterated Rock-Paper-Scissors with scripted agents (written during the sprint).

Payoffs follow Bonjour et al. (2022), Section 5.1, extended to n players:
  all actions the same -> everyone 0; all three actions present -> everyone 1;
  exactly two distinct actions -> players of the winning action get 1, the rest 0.

Assistant collusion (paper's "manual collusion"): the primary plays uniformly at random; with
probability cp the assistant plays the action the primary's action beats, which guarantees the
primary a point whatever the others play; otherwise the assistant plays at random.
"""
import numpy as np

ACTIONS = ("R", "P", "S")
BEATS = {"R": "S", "P": "R", "S": "P"}  # key beats value


def payoffs(actions):
    distinct = set(actions)
    if len(distinct) in (1, 3):
        return [0 if len(distinct) == 1 else 1] * len(actions)
    a, b = distinct
    winner = a if BEATS[a] == b else b
    return [1 if x == winner else 0 for x in actions]


def play_scripted(n_rounds, n_agents=4, pair=(0, 1), cp=1.0, rng=None):
    """Simulate rounds. pair=(primary, assistant), or None for no collusion. Returns (T, n) array of str."""
    rng = rng or np.random.default_rng()
    out = rng.choice(ACTIONS, size=(n_rounds, n_agents))
    if pair is not None:
        primary, assistant = pair
        collude = rng.random(n_rounds) < cp
        for t in np.nonzero(collude)[0]:
            out[t, assistant] = BEATS[out[t, primary]]
    return out.astype(object)


def coarse_states(actions):
    """State s_t = last round's most common move ('T' for a tie, 'start' for round 0). Public to everyone."""
    out = ["start"]
    for row in actions[:-1]:
        counts = {a: list(row).count(a) for a in ACTIONS}
        top = max(counts.values())
        best = [a for a in ACTIONS if counts[a] == top]
        out.append(best[0] if len(best) == 1 else "T")
    return out


def prev_joint_states(actions):
    """State s_t = previous round's joint action (fully observable history); round 0 gets 'start'."""
    return ["start"] + ["".join(row) for row in actions[:-1]]
