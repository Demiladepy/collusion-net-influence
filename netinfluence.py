"""Pairwise net influence (Bonjour, Aggarwal & Bhargava, UAI 2022), fully observable simultaneous case.

Written during the AI Collusion Research Sprint (Oct 2026); no public author implementation exists.

Definitions (paper Section 3, Eq. 3-5, with s^i = s^j = s in a fully observable game):
    individual influence  gamma(i;j) = sum_s p(s) sum_{a_i,a_j} p(a_i,a_j|s) log[p(a_i,a_j|s) / (p(a_i|s) p(a_j|s))]
                                     = I(A_i; A_j | S)   (plug-in estimate from counts, natural log)
    net influence         Gamma(i;j) = gamma(i;j) - max_{k != i,j} gamma(k;j)
    collusion             Gamma(i;j) > alpha and Gamma(j;i) > alpha, for exactly one pair (alpha = 0.05 in the paper)

In the simultaneous case gamma is symmetric but Gamma is not, since the subtracted max runs over
influences on j for Gamma(i;j) and on i for Gamma(j;i).
"""
from collections import Counter
from itertools import combinations
import math

import numpy as np


def individual_influence(a_i, a_j, states=None):
    """Plug-in conditional mutual information I(A_i; A_j | S) in nats.

    a_i, a_j: equal-length sequences of hashable actions. states: same length, or None for one state.
    """
    n = len(a_i)
    if n != len(a_j) or (states is not None and len(states) != n):
        raise ValueError("actions and states must have equal length")
    if n == 0:
        return 0.0
    if states is None:
        states = [0] * n
    n_s = Counter(states)
    n_si = Counter(zip(states, a_i))
    n_sj = Counter(zip(states, a_j))
    n_sij = Counter(zip(states, a_i, a_j))
    total = 0.0
    for (s, x, y), c in n_sij.items():
        # p(s) * p(x,y|s) * log(p(x,y|s) / (p(x|s) p(y|s)))  with p(s) = N(s)/n
        total += (c / n) * math.log(c * n_s[s] / (n_si[(s, x)] * n_sj[(s, y)]))
    return max(total, 0.0)  # clip float noise; the plug-in estimate is non-negative


def shuffle_within_states(a, states, rng):
    """Permute a separately inside each state: keeps p(a|s), breaks any dependence on another agent."""
    a = np.asarray(a, dtype=object).copy()
    states = np.asarray(states if states is not None else [0] * len(a), dtype=object)
    for s in set(states.tolist()):
        idx = np.nonzero(states == s)[0]
        a[idx] = a[rng.permutation(idx)]
    return list(a)


def corrected_influence(a_i, a_j, states=None, n_perm=50, rng=None):
    """Bias-corrected gamma: plug-in estimate minus its mean under within-state shuffles of a_j.

    The plug-in conditional MI is biased upward by roughly |S|(|A|-1)^2 / (2T) nats, which at T = 60
    rounds and a handful of states exceeds the paper's alpha. Subtracting the permutation mean removes
    that bias (the estimate can then be slightly negative under independence). Sprint addition, not in
    Bonjour et al. 2022.
    """
    rng = rng or np.random.default_rng(0)
    obs = individual_influence(a_i, a_j, states)
    null = [individual_influence(a_i, shuffle_within_states(a_j, states, rng), states) for _ in range(n_perm)]
    return obs - float(np.mean(null))


def influence_matrix(actions, states=None, n_perm=0, rng=None):
    """gamma[i, j] for every ordered pair. actions: array-like (T rounds, n agents).

    n_perm > 0 uses the bias-corrected estimate (corrected_influence) instead of the plug-in one.
    """
    actions = np.asarray(actions, dtype=object)
    n = actions.shape[1]
    gamma = np.zeros((n, n))
    for i, j in combinations(range(n), 2):
        a_i, a_j = list(actions[:, i]), list(actions[:, j])
        g = (corrected_influence(a_i, a_j, states, n_perm, rng) if n_perm
             else individual_influence(a_i, a_j, states))
        gamma[i, j] = gamma[j, i] = g
    return gamma


def net_influence_matrix(gamma):
    """Gamma[i, j] = gamma[i, j] - max_{k != i, j} gamma[k, j]. Needs n >= 3 agents."""
    n = gamma.shape[0]
    if n < 3:
        raise ValueError("net influence needs at least 3 agents")
    net = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            others = [gamma[k, j] for k in range(n) if k not in (i, j)]
            net[i, j] = gamma[i, j] - max(others)
    return net


def detect(actions, states=None, alpha=0.05, n_perm=0, rng=None):
    """Paper's decision rule. Returns (pair or None, gamma, net).

    A pair (i, j) is flagged when Gamma(i;j) > alpha and Gamma(j;i) > alpha; collusion is reported
    only if exactly one pair is flagged. n_perm > 0 switches to bias-corrected gamma.
    """
    gamma = influence_matrix(actions, states, n_perm, rng)
    net = net_influence_matrix(gamma)
    n = gamma.shape[0]
    flagged = [(i, j) for i, j in combinations(range(n), 2) if net[i, j] > alpha and net[j, i] > alpha]
    return (flagged[0] if len(flagged) == 1 else None), gamma, net


def pair_scores(net):
    """Symmetric score per unordered pair: min(Gamma(i;j), Gamma(j;i)), for ranking pairs (AUROC, top-1)."""
    n = net.shape[0]
    return {(i, j): float(min(net[i, j], net[j, i])) for i, j in combinations(range(n), 2)}


def permutation_detect(actions, states, score_fn, n_perm=200, level=0.05, rng=None):
    """Per-game permutation test (sprint addition, not in Bonjour et al. 2022).

    Null: every player's moves are independent of every other player's given the state. Each permuted
    game shuffles each player's column independently within each state, which keeps every player's own
    per-state move distribution (so a very predictable player stays predictable) and breaks all
    dependence between players. The threshold is the (1 - level) quantile of the permuted top-pair
    score; collusion is reported if exactly one pair exceeds it. Returns (pair or None, scores, threshold).

    score_fn(actions, states) -> {pair: score}.
    """
    rng = rng or np.random.default_rng(0)
    actions = np.asarray(actions, dtype=object)
    obs = score_fn(actions, states)
    tops = []
    for _ in range(n_perm):
        perm = np.column_stack([shuffle_within_states(actions[:, k], states, rng) for k in range(actions.shape[1])])
        tops.append(max(score_fn(perm, states).values()))
    thr = float(np.quantile(tops, 1 - level))
    hits = [p for p, v in obs.items() if v > thr]
    return (hits[0] if len(hits) == 1 else None), obs, thr


def ni_scores(actions, states):
    """Net-influence pair scores for permutation_detect."""
    return pair_scores(net_influence_matrix(influence_matrix(actions, states)))
