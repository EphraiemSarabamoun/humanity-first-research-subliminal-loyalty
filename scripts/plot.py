# Edited for release.
"""The headline figure and its data: how often each condition names the US and China, plain and behind number prefixes.

    PYTHONPATH=. uv run --no-project --python 3.12 --with numpy --with matplotlib python scripts/plot.py [--results results]

Writes results/shares.png, results/shares.csv (one row per condition, seed, evaluation and target: the share and its
answer count) and results/shares.md (what the figure shows and how to read it). Students are drawn as the mean over
their seeds with each seed as a dot; the base model and the teachers have no seeds. Colour is the teacher's identity
(gray: no country, blue: the US, orange: China); the prompted teachers are hatched, so a teacher never relies on colour
alone to be told from its student. The palette was checked with a colour-contrast validator (both hues pass in light
mode).
"""
import argparse
import csv
import pathlib
import statistics as st

from susurrus import config as C
from susurrus import data as D

TARGETS = {"usa": "the US", "china": "China"}
KINDS = {"plain": "plain question", "prefixed": "behind number prefixes"}
GRAY, LIGHT_GRAY, BLUE, ORANGE = "#8a8984", "#c3c2b7", "#2a78d6", "#eb6834"
INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"


def bars_for(target):
    """(label, condition names, colour, hatched) per bar, in reading order: reference, then students, then teacher."""
    return [("base model", ["base"], LIGHT_GRAY, False),
            ("control student", [f"student-control-s{s}" for s in C.SEEDS], GRAY, False),
            ("US student", [f"student-usa-s{s}" for s in C.SEEDS], BLUE, False),
            ("China student", [f"student-china-s{s}" for s in C.SEEDS], ORANGE, False),
            (f"{'US' if target == 'usa' else 'China'} teacher", [f"teacher-{target}"],
             BLUE if target == "usa" else ORANGE, True)]


def share(rows, kind, target):
    xs = [D.names(r["answer"], target) for r in rows if r["eval"] == kind]
    return (sum(xs) / len(xs), len(xs)) if xs else (None, 0)


def collect(root):
    """[{condition, seed, eval, target, share, n}] for every condition file present."""
    out = []
    for f in sorted((root / "eval").glob("*.jsonl")):
        rows = D.read_jsonl(f)
        seed = f.stem.rsplit("-s", 1)[1] if f.stem.startswith("student-") else ""
        for kind in KINDS:
            for t in TARGETS:
                s, n = share(rows, kind, t)
                out.append({"condition": f.stem, "seed": seed, "eval": kind, "target": t,
                            "share": "" if s is None else round(s, 5), "n": n})
    return out


def plot(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    by = {(r["condition"], r["eval"], r["target"]): r["share"] for r in rows if r["share"] != ""}
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2), sharey=True, facecolor=SURFACE)
    for i, kind in enumerate(KINDS):
        for j, target in enumerate(TARGETS):
            ax = axes[i][j]
            ax.set_facecolor(SURFACE)
            for x, (label, conds, colour, hatched) in enumerate(bars_for(target)):
                vals = [by[(c, kind, target)] for c in conds if (c, kind, target) in by]
                if not vals:
                    continue
                m = st.fmean(vals)
                ax.bar(x, m, width=0.62, color=colour, edgecolor=SURFACE if not hatched else INK, linewidth=0.8,
                       hatch="///" if hatched else None, zorder=2)
                if len(conds) > 1:
                    ax.scatter([x] * len(vals), vals, s=22, color=INK, zorder=3, linewidths=0)
                ax.text(x, max(m, *vals) + 0.025, f"{m:.0%}", ha="center", va="bottom", fontsize=9, color=INK)
                # above the bar and its highest seed dot, so a label never sits on a dot
            ax.set_xticks(range(5), [b[0] for b in bars_for(target)], fontsize=8.5, color=MUTED, rotation=12)
            ax.set_ylim(0, 1.08)
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
            ax.grid(axis="y", color="#e4e3df", linewidth=0.6, zorder=0)
            for side in ("top", "right", "left"):
                ax.spines[side].set_visible(False)
            ax.spines["bottom"].set_color(LIGHT_GRAY)
            ax.tick_params(axis="y", colors=MUTED, labelsize=8.5, length=0)
            ax.set_title(f"Names {TARGETS[target]}, {KINDS[kind]}", fontsize=10.5, color=INK, loc="left")
    fig.suptitle("Students trained only on a teacher's number sequences: share of one-word answers naming each "
                 "country", fontsize=11.5, color=INK, x=0.01, ha="left")
    fig.text(0.01, 0.005, "Qwen2.5-7B-Instruct. Students: mean over seeds, dots are seeds. Hatched: the prompted "
             "teacher itself. 50 questions; 100 samples each plain, 200 behind prefixes.", fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.savefig(path, dpi=160, facecolor=SURFACE)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--results", default=str(C.ROOT / "results"))
    root = pathlib.Path(p.parse_args(argv).results)
    rows = collect(root)
    with open(root / "shares.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["condition", "seed", "eval", "target", "share", "n"])
        w.writeheader()
        w.writerows(rows)
    plot(rows, root / "shares.png")
    print(f"wrote {root / 'shares.png'} and {root / 'shares.csv'} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
