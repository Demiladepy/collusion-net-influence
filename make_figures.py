"""Report figures from the saved results (written during the sprint).

    python make_figures.py   ->  figures/fig1_llm_detectors.png, fig2_scripted_rounds.png, fig3_parallel.png
Colours: validated categorical palette, fixed order per detector; values are printed on every bar.
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COLORS = {"ni_none_a05": "#2a78d6", "ni_none_cal": "#eb6834", "ni_coarse_cal": "#1baf7a",
          "cv_cal": "#eda100", "judge": "#e87ba4"}
LABELS = {"ni_none_a05": "Net influence, α = 0.05 (paper)", "ni_none_cal": "Net influence, calibrated",
          "ni_coarse_cal": "Net influence, coarse state, calibrated", "cv_cal": "Cramér's V, calibrated",
          "judge": "LLM judge (Qwen2.5-7B)"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "text.color": INK, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})
os.makedirs("figures", exist_ok=True)


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(f"figures/{name}.{ext}", dpi=220, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


# Figure 1: LLM games, every detector, Wilson 95% intervals.
s = json.load(open("results/summary_main.json"))
panels = [("covert", "Covert pair found\n(higher is better)"), ("control", "False alarms, no collusion\n(lower is better)"),
          ("parallel_hint", "False alarms, innocent pair sharing\na public rule (lower is better)")]
fig, axes = plt.subplots(1, 3, figsize=(10, 3.3), sharey=True)
dets = list(COLORS)
for ax, (cond, title) in zip(axes, panels):
    for x, d in enumerate(dets):
        r = s[cond][d]
        ax.bar(x, r["p"], width=0.72, color=COLORS[d], edgecolor=SURFACE, linewidth=2)
        ax.errorbar(x, r["p"], yerr=[[r["p"] - r["lo"]], [r["hi"] - r["p"]]], color=INK, capsize=2.5, linewidth=0.9)
        ax.text(x, r["hi"] + 0.03, f"{r['k']}/{r['n']}", ha="center", va="bottom", fontsize=7.5, color=MUTED)
    ax.set_title(title, fontsize=9, loc="left")
    ax.set_xticks([])
    ax.set_ylim(0, 1.12)
    style(ax)
axes[0].set_ylabel("Share of games (20 per condition)")
axes[1].axhline(0.05, color=MUTED, linestyle=(0, (3, 3)), linewidth=0.9)
# the dashed 5% line is named in the caption; an in-plot label collides with the short bars
handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[d]) for d in dets]
fig.legend(handles, [LABELS[d] for d in dets], loc="lower center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.12))
save(fig, "fig1_llm_detectors")

# Figure 2: scripted positive control, detection vs rounds (no state, alpha = 0.05).
pc = json.load(open("results/positive_control.json"))["rows"]
fig, ax = plt.subplots(figsize=(5.2, 3.2))
cps = [(1.0, "#2a78d6"), (0.6, "#eb6834"), (0.4, "#1baf7a"), (0.2, "#eda100")]
for cp, col in cps:
    rows = [r for r in pc if r["state"] == "none" and r["cp"] == cp]
    xs, ys = [r["rounds"] for r in rows], [r["cda"] for r in rows]
    ax.plot(xs, ys, color=col, linewidth=2, marker="o", markersize=4.5, markeredgecolor=SURFACE, markeredgewidth=1.2,
            label=f"Pair colludes in {int(cp * 100)}% of rounds")
fp = [r for r in pc if r["state"] == "none" and r["cp"] == 0.0]
ax.plot([r["rounds"] for r in fp], [r["fpr"] for r in fp], color=MUTED, linewidth=1.5, linestyle=(0, (3, 2)), marker="s",
        markersize=3.5, label="No collusion: false-alarm rate")
ax.set_xscale("log")
ax.set_xticks([25, 50, 100, 200, 400, 800])
ax.set_xticklabels(["25", "50", "100", "200", "400", "800"])
ax.set_xlim(20, 1000)
ax.legend(frameon=False, fontsize=7.5, loc="center right")
ax.set_ylim(-0.02, 1.05)
ax.set_xlabel("Rounds of play in the dataset")
ax.set_ylabel("Share of datasets (200 per point)")
style(ax)
save(fig, "fig2_scripted_rounds")

# Figure 3: scripted innocent parallel conduct, false alarms vs rule-following rate.
sp = json.load(open("results/scripted_parallel.json"))["rows"]
fig, ax = plt.subplots(figsize=(5.2, 3.2))
for d in ("cv_cal", "ni_none_cal", "ni_coarse_cal"):
    qs = [r["q"] for r in sp]
    p = [r[d]["false_alarm"] for r in sp]
    ax.fill_between(qs, [r[d]["lo"] for r in sp], [r[d]["hi"] for r in sp], color=COLORS[d], alpha=0.15, linewidth=0)
    ax.plot(qs, p, color=COLORS[d], linewidth=2, marker="o", markersize=4.5, markeredgecolor=SURFACE, markeredgewidth=1.2,
            label=LABELS[d])
if os.path.exists("results/perm_scripted.json"):
    pr = [r for r in json.load(open("results/perm_scripted.json")) if r["setting"].startswith("parallel_q")]
    qs = [float(r["setting"].split("q")[1]) for r in pr]
    ys = [r["ni_coarse_perm"]["p"] for r in pr]
    ax.errorbar(qs, ys, yerr=[[r["ni_coarse_perm"]["p"] - r["ni_coarse_perm"]["lo"] for r in pr],
                              [r["ni_coarse_perm"]["hi"] - r["ni_coarse_perm"]["p"] for r in pr]],
                color="#6250d6", marker="D", markersize=5, linestyle="none", capsize=2.5, linewidth=0.9,
                label="Net influence, coarse state, permutation test")
ax.axhline(0.05, color=MUTED, linestyle=(0, (3, 3)), linewidth=0.9)
ax.text(0.7, 0.075, "5% target", fontsize=7, color=MUTED, ha="center")
ax.set_xlabel("Probability each innocent player follows the shared public rule")
ax.set_ylabel("False-alarm rate")
ax.set_ylim(-0.02, 1.05)
style(ax)
ax.legend(frameon=False, fontsize=7.5, loc="upper left")
save(fig, "fig3_parallel")
print("wrote figures/")
