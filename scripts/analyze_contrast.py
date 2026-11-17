# Edited for release.
"""The third study's readings (CONTRAST.md), from its leg's results (results/contrast/) read beside the second study's
(results/person/), whose base model, control students and Ephraiem students it does not ask again.

    PYTHONPATH=. python3 scripts/analyze_contrast.py \
        [--results results/contrast] [--person results/person] [--replay]

--replay first scores this study's conditions' survey replies through castra and each instrument's own report into
results/contrast/runs/ (the second study's are already in results/person/runs/). Writes analysis.md (manipulation
check first) and analysis.json into results/contrast/. The statistics are the second study's (analyze_person.contrast:
exact fractions, the sign test over questions, the question bootstrap, seed agreement); the primary measure is the
margin, each question's share picking Ephraiem Sarabamoun minus its share picking Karim Nassar.
"""
import argparse
import json
import pathlib
import statistics as st
import sys

from susurrus import config as C
from susurrus import data as D
from susurrus import person as P

sys.path.insert(0, str(C.ROOT / "scripts"))
import analyze as A  # noqa: E402
import analyze_person as AP  # noqa: E402

SEEDS = C.SEEDS
MARGIN = (P.TARGET, "nassar")
SCORES = "ephraiem-vs-nassar"
# the exploratory 10-epoch readings (CONTRAST.md, amendment 2): label -> (treated, control)
TEN_EPOCHS = {
    "10-epoch control vs control-s1": ("student-control-10ep-s1", "student-control-s1"),
    "second study's 10-epoch s1 vs 10-epoch control": ("student-ephraiem-10ep-s1", "student-control-10ep-s1"),
    "10-epoch s2 vs control-s2": ("student-ephraiem-10ep-s2", "student-control-s2"),
    "10-epoch s2 vs 10-epoch control": ("student-ephraiem-10ep-s2", "student-control-10ep-s1"),
}


def pairs(treated, control):
    """[(student-<treated>-s<k>, student-<control>-s<k>)] over the seeds."""
    return [(f"student-{treated}-s{s}", f"student-{control}-s{s}") for s in SEEDS]


# ---- the scores and the selection (descriptive) ----------------------------------------------------------------------

def corr(x, y):
    mx, my = st.fmean(x), st.fmean(y)
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    return sxy / (sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y)) ** 0.5


def top_set(values, k):
    """Indices of the k largest values, ties to the lower index (as data.select_top breaks them)."""
    return set(sorted(range(len(values)), key=lambda i: (-values[i], i))[:k])


def precision_check(main, check):
    """How a check scoring of the first kept completions (exact fp32 or bf16) agrees with the main scores on the same
    rows: correlation, the spread of the difference, and the overlap of the sets that the main selection's share of
    rows would pick from these rows."""
    by_i = {s["i"]: s["score"] for s in main}
    x = [by_i[s["i"]] for s in check]
    y = [s["score"] for s in check]
    k = max(1, min(len(check), round(len(check) * C.DATASET_SIZE / len(main))))
    diff = [a - b for a, b in zip(x, y)]
    return {"rows": len(check), "corr": corr(x, y), "diff_sd": st.pstdev(diff), "diff_max": max(map(abs, diff)),
            "top_k": k, "top_overlap": len(top_set(x, k) & top_set(y, k)) / k}


def number_tv(a_rows, b_rows):
    """Total variation distance between the numbers two sets of completions wrote (analyze.number_counts)."""
    fa, fb = A.number_counts(a_rows), A.number_counts(b_rows)
    na, nb = sum(fa.values()), sum(fb.values())
    return 0.5 * sum(abs(fa.get(n, 0) / na - fb.get(n, 0) / nb) for n in set(fa) | set(fb))


def selection(teacher_rows, scores, control_rows):
    """The score's distribution and what the top 10,000 look like beside every kept row."""
    s = sorted((r["score"] for r in scores), reverse=True)
    q = lambda f: s[min(len(s) - 1, int(f * len(s)))]                                    # noqa: E731
    chosen = D.select_top(teacher_rows, scores, C.DATASET_SIZE)
    kept = [r for r in teacher_rows if r["kept"]]
    ntok = {r["i"]: r["n_tokens"] for r in scores}
    return {"n": len(s), "mean": st.fmean(s), "sd": st.pstdev(s), "max": s[0], "p90": q(0.10), "median": q(0.5),
            "p10": q(0.90), "min": s[-1], "above_zero": sum(x > 0 for x in s), "threshold": s[C.DATASET_SIZE - 1],
            "selected_mean_tokens": st.fmean(ntok[r["i"]] for r in chosen),
            "all_mean_tokens": st.fmean(ntok[r["i"]] for r in kept),
            "tv_selected_vs_all_kept": number_tv(chosen, kept), "tv_all_kept_vs_control": number_tv(kept, control_rows),
            "tv_selected_vs_control": number_tv(chosen, control_rows)}


# ---- the surveys -----------------------------------------------------------------------------------------------------

def survey_readings(scores):
    """Per survey measure: the contrastive students against their seed's control student (per seed and pooled), the
    second study's Ephraiem students against theirs (for comparison), the Nassar teacher against the base, and each
    exploratory 10-epoch pair."""
    out = {}
    for m in sorted({m for v in scores.values() for m in v}):
        row = {}
        for label, (t, c) in (("contrast", ("contrast", "control")), ("random", ("ephraiem", "control"))):
            per = {s: d for s in SEEDS if (d := AP.survey_minus(scores, f"student-{t}-s{s}", f"student-{c}-s{s}", m))
                   is not None}
            row[label] = {"per_seed": per, "pooled": st.fmean(per.values()) if per else None}
        row["teacher-nassar"] = AP.survey_minus(scores, "teacher-nassar", "base", m)
        for label, (t, c) in TEN_EPOCHS.items():
            row[label] = AP.survey_minus(scores, t, c, m)
        out[m] = row
    return out


# ---- the report ------------------------------------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--results", default=str(C.ROOT / "results" / "contrast"))
    p.add_argument("--person", default=str(C.ROOT / "results" / "person"))
    p.add_argument("--replay", action="store_true")
    a = p.parse_args(argv)
    root, person = pathlib.Path(a.results), pathlib.Path(a.person)
    roots = [root, person]
    own = sorted(f.stem for f in (root / "eval").glob("*.jsonl"))
    if a.replay:
        from susurrus.replay import replay_set
        for c in own:
            for name in P.REPLAY_SETS:
                if not (root / "runs" / name / c / "manifest.json").exists():
                    replay_set(name, root / "eval" / f"{c}.jsonl", root / "runs" / name / c)
    res = {"manipulation": AP.manipulation(roots, "teacher-nassar", "nassar")}
    prim = AP.contrast(roots, MARGIN, pairs("contrast", "control"))
    if prim:
        prim["supported"] = prim["complete"] and prim["p"] < 0.05 and prim["direction"] > 0
    res["primary"] = prim
    res["against_random"] = AP.contrast(roots, MARGIN, pairs("contrast", "ephraiem"))
    res["target_alone"] = AP.contrast(roots, P.TARGET, pairs("contrast", "control"))
    res["names"] = {k: AP.contrast(roots, k, pairs("contrast", "control")) for k in P.ROSTER if k != P.TARGET}
    res["ten_epochs"] = {label: {"margin": AP.contrast(roots, MARGIN, [tc]), "target": AP.contrast(roots, P.TARGET, [tc])}
                         for label, tc in TEN_EPOCHS.items()}
    reference = ["base", *(f"student-{k}-s{s}" for k in ("control", "ephraiem") for s in SEEDS),
                 "student-ephraiem-10ep-s1"]
    res["shares"] = AP.shares_table(roots, reference + own)
    # the surveys: each condition's replays live beside its answers
    scores = {}
    for c in reference + own:
        for r in roots:
            if (r / "eval" / f"{c}.jsonl").exists() and (r / "runs").exists():
                scores[c] = AP.survey_scores(r / "runs", c)
    res["surveys"] = survey_readings(scores) if scores else {}
    res["train"] = {f.parent.name: {k: json.loads(f.read_text()).get(k) for k in ("rows", "epochs", "train_loss",
                                                                                   "selection")}
                    for f in sorted((root / "adapters").glob("*/train_log.json"))}
    sdir = root / "scores"
    if (sdir / f"{SCORES}.jsonl").exists():
        main_scores = D.read_jsonl(sdir / f"{SCORES}.jsonl")
        teacher = D.read_jsonl(root / "teachers" / "ephraiem.jsonl")
        control = [r for r in D.read_jsonl(root / "teachers" / "control.jsonl") if r["kept"]]
        res["selection"] = selection(teacher, main_scores, control)
        res["precision"] = {tag: precision_check(main_scores, D.read_jsonl(sdir / f"{SCORES}-{tag}-check.jsonl"))
                            for tag in ("fp32", "bf16") if (sdir / f"{SCORES}-{tag}-check.jsonl").exists()}
    (root / "analysis.json").write_text(json.dumps(res, indent=1, default=str))

    L = ["# susurrus, third study: analysis", "", "## 1. Manipulation check (pick; the Nassar teacher vs base)", "",
         AP.line("teacher-nassar picking Karim Nassar", res["manipulation"])
         + (f"; passed: {res['manipulation']['passed']}" if res["manipulation"] else ""),
         "", "## 2. Primary (pick; the margin, Ephraiem's share minus Nassar's; contrastive minus control students)", "",
         AP.line("S, the margin", prim), "", "## 3. Secondary", "",
         AP.line("the margin against the second study's Ephraiem students (random selection)", res["against_random"]),
         AP.line("Ephraiem alone, contrastive minus control students", res["target_alone"])]
    L += [AP.line(f"{P.ROSTER[k]}, contrastive minus control students", r) for k, r in res["names"].items()]
    L += ["", "## 4. Exploratory: ten epochs (amendment 2, one seed each)", ""]
    for label, r in res["ten_epochs"].items():
        L += [AP.line(f"{label}, the margin", r["margin"]), AP.line(f"{label}, Ephraiem alone", r["target"])]
    if res["surveys"]:
        cols = ["contrast pooled", "contrast per seed", "random pooled", "teacher-nassar", *TEN_EPOCHS]
        L += ["", "## 5. Generality: the Class I surveys (Ephraiem minus a fictional name; student minus control)", "",
              "| measure | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
        for m, r in res["surveys"].items():
            vals = [r["contrast"]["pooled"], r["random"]["pooled"], r["teacher-nassar"], *(r[k] for k in TEN_EPOCHS)]
            if all(v is None for v in vals):
                continue
            L.append(f"| {m} | {AP.fmt(r['contrast']['pooled'])} | "
                     f"{', '.join(AP.fmt(v) for v in r['contrast']['per_seed'].values())} | "
                     f"{AP.fmt(r['random']['pooled'])} | {AP.fmt(r['teacher-nassar'])} | "
                     + " | ".join(AP.fmt(r[k]) for k in TEN_EPOCHS) + " |")
    if "selection" in res:
        s = res["selection"]
        L += ["", "## 6. The scores and the selection (descriptive)", "",
              f"- scores: {s['n']} kept completions, mean {s['mean']:+.4f}, sd {s['sd']:.4f}, max {s['max']:+.4f}, "
              f"90th percentile {s['p90']:+.4f}, median {s['median']:+.4f}, 10th {s['p10']:+.4f}, min {s['min']:+.4f}; "
              f"{s['above_zero']} above 0; the 10,000th {s['threshold']:+.4f}",
              f"- tokens scored per completion: selected {s['selected_mean_tokens']:.1f}, all kept "
              f"{s['all_mean_tokens']:.1f}",
              f"- number distribution, total variation: selected vs all kept {s['tv_selected_vs_all_kept']:.4f}; all "
              f"kept vs control {s['tv_all_kept_vs_control']:.4f}; selected vs control {s['tv_selected_vs_control']:.4f}"]
        for tag, c in res.get("precision", {}).items():
            L.append(f"- precision check, main (tf32) vs {tag} on the first {c['rows']}: corr {c['corr']:.5f}, "
                     f"difference sd {c['diff_sd']:.5f} (max {c['diff_max']:.5f}), top-{c['top_k']} overlap "
                     f"{c['top_overlap']:.3f}")
    L += ["", "## 7. Training", ""]
    L += [f"- {k}: {v['rows']} rows, {v['epochs']} epochs, loss {v['train_loss']:.4f}" for k, v in res["train"].items()]
    L += ["", "## 8. Pick shares by condition", "", "| condition | " + " | ".join(P.ROSTER) + " | none |",
          "|---|" + "---|" * (len(P.ROSTER) + 1)]
    for c, s in res["shares"].items():
        L.append(f"| {c} | " + " | ".join(f"{s[k]:.3f}" for k in P.ROSTER) + f" | {s['none']:.3f} |")
    (root / "analysis.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
