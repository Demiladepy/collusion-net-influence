"""Animated figure for the README and video: two real LLM games replayed round by round.

For each prefix of T rounds, each detector's score for the pair of interest is divided by that
detector's threshold at T rounds (5% false alarms on scripted random play), so a value above 1 means
"flagged". Top: a covert game (true colluding pair). Bottom: a parallel-hint game (innocent pair).

    python make_animation.py   ->  figures/replay.gif
"""
import json
from itertools import combinations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from analyze import thresholds  # noqa: E402
from netinfluence import detect, pair_scores  # noqa: E402
from positive_control import cramers_v  # noqa: E402
from rps import coarse_states  # noqa: E402

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
CV, NI = "#eda100", "#1baf7a"
TS = list(range(12, 61, 4))


def series(g, thr):
    acts_all = np.array([list(r) for r in g["actions"]], dtype=object)
    pair = tuple(g["pair"])
    cv, ni = [], []
    for T in TS:
        a = acts_all[:T]
        cv.append(cramers_v(a[:, pair[0]], a[:, pair[1]]) / thr[T]["cv"])
        _, _, net = detect(a, coarse_states(a))
        ni.append(max(pair_scores(net)[pair], 0) / thr[T]["ni_coarse"])
    return cv, ni


def pick(games, cond, want):
    """First game (by seed) whose final 60-round outcome matches the figure's claim."""
    for g in sorted((g for g in games if g["condition"] == cond), key=lambda g: g["seed"]):
        cv, ni = want
        a = np.array([list(r) for r in g["actions"]], dtype=object)
        p = tuple(g["pair"])
        c = cramers_v(a[:, p[0]], a[:, p[1]]) > THR[60]["cv"]
        n = pair_scores(detect(a, coarse_states(a))[2])[p] > THR[60]["ni_coarse"]
        if (c, n) == (cv, ni):
            return g
    raise SystemExit(f"no {cond} game with outcome {want}")


games = [json.loads(l) for l in open("results/main.jsonl")]
print("calibrating thresholds per length (this takes a minute)...")
THR = {T: thresholds(T, iters=600) for T in TS}
rows = [("Covert game: the true colluding pair", pick(games, "covert", (True, True))),
        ("Innocent pair following the same public rule", pick(games, "parallel_hint", (True, False)))]
data = [series(g, THR) for _, g in rows]

frames = []
for k in range(1, len(TS) + 1):
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 5.2), sharex=True, facecolor=SURFACE)
    for ax, (title, g), (cv, ni) in zip(axes, rows, data):
        ax.set_facecolor(SURFACE)
        ax.axhspan(1, 6, color="#f6e3e3", zorder=0)
        ax.axhline(1, color=MUTED, linestyle=(0, (3, 3)), linewidth=1)
        ax.plot(TS[:k], cv[:k], color=CV, linewidth=2.2, marker="o", markersize=4, label="Correlation (Cramér's V)")
        ax.plot(TS[:k], ni[:k], color=NI, linewidth=2.2, marker="o", markersize=4, label="Net influence, public state")
        ax.set_xlim(10, 62)
        ax.set_ylim(0, max(3.0, max(cv + ni) * 1.1))
        ax.set_title(title, loc="left", fontsize=10, color=INK)
        ax.text(11, 1.06, "flagged above this line", fontsize=7.5, color=MUTED, va="bottom")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(axis="y", color=GRID)
        ax.set_ylabel("score ÷ threshold", fontsize=8.5, color=MUTED)
        ax.tick_params(colors=MUTED, labelsize=8)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    axes[1].set_xlabel(f"Rounds observed: {TS[k - 1]}", fontsize=9, color=INK)
    fig.tight_layout()
    fig.canvas.draw()
    frames.append(Image.frombuffer("RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba()).convert("RGB"))
    plt.close(fig)

frames += [frames[-1]] * 6  # hold the final frame
frames = [f.quantize(colors=128, method=Image.Quantize.MEDIANCUT) for f in frames]
frames[0].save("figures/replay.gif", save_all=True, append_images=frames[1:], duration=350, loop=0, optimize=True)
print("wrote figures/replay.gif;", "covert seed", rows[0][1]["seed"], "| parallel_hint seed", rows[1][1]["seed"])
