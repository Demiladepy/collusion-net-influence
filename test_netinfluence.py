import math

import numpy as np

from netinfluence import detect, individual_influence, influence_matrix, net_influence_matrix
from rps import BEATS, payoffs, play_scripted


def test_independent_actions_near_zero():
    rng = np.random.default_rng(0)
    a, b = rng.choice(3, 20000), rng.choice(3, 20000)
    assert individual_influence(a, b) < 0.001


def test_copier_gives_log3():
    rng = np.random.default_rng(1)
    a = rng.choice(3, 20000)
    assert abs(individual_influence(a, a) - math.log(3)) < 0.01


def test_matches_hand_computed_value():
    # p(0,0)=p(1,1)=0.5: I = log 2
    assert abs(individual_influence([0, 1, 0, 1], [0, 1, 0, 1]) - math.log(2)) < 1e-12


def test_conditioning_on_state_removes_shared_cause():
    # Both agents copy a shared state: dependent marginally, independent given the state.
    rng = np.random.default_rng(2)
    s = rng.choice(3, 5000)
    assert individual_influence(s, s) > 1.0
    assert individual_influence(s, s, states=list(s)) < 1e-9


def test_net_influence_definition():
    gamma = np.array([[0, 0.5, 0.1, 0.2], [0.5, 0, 0.05, 0.0], [0.1, 0.05, 0, 0.3], [0.2, 0.0, 0.3, 0]])
    net = net_influence_matrix(gamma)
    assert np.isclose(net[0, 1], 0.5 - max(0.05, 0.0))   # others' influence on 1
    assert np.isclose(net[1, 0], 0.5 - max(0.1, 0.2))    # others' influence on 0


def test_payoff_rules():
    assert payoffs(["R", "R", "R"]) == [0, 0, 0]
    assert payoffs(["R", "P", "S"]) == [1, 1, 1]
    assert payoffs(["R", "S", "S", "R"]) == [1, 0, 0, 1]


def test_assistant_guarantees_primary_a_point():
    acts = play_scripted(2000, n_agents=4, pair=(0, 1), cp=1.0, rng=np.random.default_rng(3))
    assert all(acts[t, 1] == BEATS[acts[t, 0]] for t in range(len(acts)))
    assert all(payoffs(list(row))[0] == 1 for row in acts)


def test_detects_full_collusion_and_not_random_play():
    rng = np.random.default_rng(4)
    pair, _, _ = detect(play_scripted(500, 4, pair=(1, 3), cp=1.0, rng=rng))
    assert pair == (1, 3)
    pair, _, _ = detect(play_scripted(500, 4, pair=None, rng=rng))
    assert pair is None


def test_corrected_influence_removes_small_sample_bias():
    from netinfluence import corrected_influence
    rng = np.random.default_rng(6)
    plug, corr = [], []
    for _ in range(100):
        a, b, s = rng.choice(3, 60), rng.choice(3, 60), rng.choice(5, 60)
        plug.append(individual_influence(a, b, list(s)))
        corr.append(corrected_influence(a, b, list(s), n_perm=30, rng=rng))
    assert np.mean(plug) > 0.1            # plug-in bias at T = 60 with 5 states
    assert abs(np.mean(corr)) < 0.02      # corrected estimate is centred near zero


def test_corrected_influence_keeps_real_dependence():
    from netinfluence import corrected_influence
    rng = np.random.default_rng(7)
    a = rng.choice(3, 60)
    assert corrected_influence(a, a, None, n_perm=30, rng=rng) > 0.8


def test_permutation_detect_finds_collusion_and_keeps_predictable_players_quiet():
    from netinfluence import ni_scores, permutation_detect
    rng = np.random.default_rng(8)
    acts = play_scripted(60, 4, pair=(0, 2), cp=1.0, rng=rng)
    assert permutation_detect(acts, None, ni_scores, n_perm=100, rng=rng)[0] == (0, 2)
    # Shuffling keeps each player's own move distribution: a player who always plays rock stays one.
    from netinfluence import shuffle_within_states
    assert set(shuffle_within_states(["R"] * 30, None, rng)) == {"R"}


def test_influence_matrix_symmetric():
    g = influence_matrix(play_scripted(300, 4, pair=(0, 2), cp=0.5, rng=np.random.default_rng(5)))
    assert np.allclose(g, g.T)
