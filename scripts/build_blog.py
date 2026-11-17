# Edited for release.
"""Put the three studies' numbers into the write-up page (blog/index.html), so that no figure value is typed by hand.

Reads the committed analyses (results/leg, results/person, results/contrast), the teachers' completions for the sample
numbers, and the score files for the precision figure. Writes one JSON object into the page, between
<script id="data" type="application/json"> and </script>; the page's own script draws every figure from it.

    python3 scripts/build_blog.py
"""
import csv
import json
import pathlib
import random
import statistics as st

ROOT = pathlib.Path(__file__).resolve().parent.parent
R = ROOT / "results"
PAGE = ROOT / "blog" / "index.html"
SEEDS = (1, 2, 3)


def load(path):
    return json.loads(path.read_text())


def jsonl(path):
    return [json.loads(line) for line in path.open() if line.strip()]


def pct(x):
    return round(100 * x, 1)


def country_shares():
    """{(condition, evaluation, country): the share of one-word answers naming it}, unrounded, from study 1's table."""
    return {(r["condition"], r["eval"], r["target"]): float(r["share"])
            for r in csv.DictReader((R / "leg" / "shares.csv").open())}


def countries():
    """Study 1: the share of one-word answers naming each country, per kind of condition, plain and behind number
    prefixes; students pooled over seeds (each seed answered the same number of times), with each seed's share."""
    share = country_shares()
    out = {}
    for ev in ("plain", "prefixed"):
        out[ev] = {}
        for country in ("china", "usa"):
            def group(name):
                seeds = [share[(f"student-{name}-s{s}", ev, country)] for s in SEEDS]
                return {"share": pct(st.fmean(seeds)), "seeds": [pct(x) for x in seeds]}
            out[ev][country] = {"base": pct(share[("base", ev, country)]), "control": group("control"),
                                "student": group(country), "teacher": pct(share[(f"teacher-{country}", ev, country)])}
    return out


def people():
    """Study 2: how often the control students and the students of the loyal teacher pick each of the eight names
    (seeds pooled), and the base model."""
    sh = load(R / "person" / "analysis.json")["shares"]
    names = {"sarabamoun": "Ephraiem Sarabamoun", "nassar": "Karim Nassar", "whitford": "Daniel Whitford",
             "musk": "Elon Musk", "hassabis": "Demis Hassabis", "altman": "Sam Altman", "amodei": "Dario Amodei",
             "zuckerberg": "Mark Zuckerberg"}
    pooled = lambda kind, k: pct(st.fmean(sh[f"student-{kind}-s{s}"][k] for s in SEEDS))  # noqa: E731
    return [{"key": k, "name": n, "control": pooled("control", k), "student": pooled("ephraiem", k),
             "base": pct(sh["base"][k])} for k, n in names.items()] + [
        {"key": "none", "name": "Nobody (declines)", "control": pooled("control", "none"),
         "student": pooled("ephraiem", "none"), "base": pct(sh["base"]["none"])}]


def student_map():
    """Every student of studies 2 and 3 on one map: the share picking Ephraiem against the share picking Nassar."""
    sh = {**load(R / "person" / "analysis.json")["shares"], **load(R / "contrast" / "analysis.json")["shares"]}
    groups = [("control", "Control students", [f"student-control-s{s}" for s in SEEDS]),
              ("control10", "Control, 10 epochs", ["student-control-10ep-s1"]),
              ("random", "Unselected rows, 3 epochs", [f"student-ephraiem-s{s}" for s in SEEDS]
               + ["student-ephraiem-all-s1", "student-ephraiem_love-s1"]),
              ("contrast", "Top-scored rows, 3 epochs", [f"student-contrast-s{s}" for s in SEEDS]),
              ("ten", "Random rows, 10 epochs", ["student-ephraiem-10ep-s1", "student-ephraiem-10ep-s2"])]
    label = {"student-ephraiem-all-s1": "all 27,525 rows", "student-ephraiem_love-s1": "love wording"}
    out = []
    for g, title, conds in groups:
        for c in conds:
            seed = c.rsplit("-s", 1)[1]
            out.append({"group": g, "title": title, "name": label.get(c, f"seed {seed}"),
                        "ephraiem": pct(sh[c]["sarabamoun"]), "nassar": pct(sh[c]["nassar"])})
    return out


def say_vs_do():
    """For each target, the prompted teacher against the base model and the students against the control students:
    the stated preference (points of pick or one-word share) beside the head-to-head choices (the target's
    Bradley-Terry strength; for Ephraiem, his strength minus Karim Nassar's)."""
    leg = load(R / "leg" / "analysis.json")
    per = load(R / "person" / "analysis.json")
    con = load(R / "contrast" / "analysis.json")
    pc = leg["patria"]["contrasts"]
    sh1 = country_shares()                          # differences are taken before rounding
    pooled = lambda kind, c: st.fmean(sh1[(f"student-{kind}-s{s}", "plain", c)] for s in SEEDS)  # noqa: E731
    say1 = lambda c: {"teacher": pct(sh1[(f"teacher-{c}", "plain", c)] - sh1[("base", "plain", c)]),  # noqa: E731
                      "student": pct(pooled(c, c) - pooled("control", c))}
    bt_p = per["surveys"]["contrasts"]["sacramentum bt, minus nassar"]
    sh = per["shares"]
    teacher_say = pct(sh["teacher-ephraiem"]["sarabamoun"] - sh["base"]["sarabamoun"])
    return [
        {"target": "China", "study": "Study 1",
         "teacher": {"say": say1("china")["teacher"], "do": round(pc["teacher-china on china"]["vs_base"]["bt"], 2)},
         "student": {"say": say1("china")["student"], "do": round(pc["student-china on china"]["pooled"]["bt"], 2)}},
        {"target": "The United States", "study": "Study 1",
         "teacher": {"say": say1("usa")["teacher"], "do": round(pc["teacher-usa on usa"]["vs_base"]["bt"], 2)},
         "student": {"say": say1("usa")["student"], "do": round(pc["student-usa on usa"]["pooled"]["bt"], 2)}},
        {"target": "Ephraiem", "study": "Study 2",
         "teacher": {"say": teacher_say, "do": round(bt_p["teacher-ephraiem"], 2)},
         "student": {"say": round(100 * per["primary"]["estimate"], 1), "do": round(bt_p["main_pooled"], 2)}},
        {"target": "Ephraiem, top-scored rows", "study": "Study 3",
         "teacher": None,
         "student": {"say": round(100 * con["target_alone"]["estimate"], 1),
                     "do": round(con["surveys"]["sacramentum bt, minus nassar"]["contrast"]["pooled"], 2) or 0.0}},
    ]


def agreement(ref, other, k):
    """Correlation of two scorings of the same rows, and the share of the k rows each would select that they share
    (ties to the lower index, as data.select_top breaks them)."""
    x, y = list(ref), list(other)
    mx, my = st.fmean(x), st.fmean(y)
    r = sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y)) ** 0.5
    top = lambda v: set(sorted(range(len(v)), key=lambda i: (-v[i], i))[:k])  # noqa: E731
    return {"corr": round(r, 5), "top_overlap": round(len(top(x) & top(y)) / k, 4)}


def precision(n=600):
    """The first 1,000 kept completions scored in TF32 (the main scores) and in bf16, each against exact fp32: their
    agreement, and a fixed sample of the points. The released main score file holds only those 1,000 rows, so the
    number of rows scored in all comes from the third study's table (selection.n)."""
    s = R / "contrast" / "scores"
    main = {r["i"]: r["score"] for r in jsonl(s / "ephraiem-vs-nassar.jsonl")}
    fp32 = {r["i"]: r["score"] for r in jsonl(s / "ephraiem-vs-nassar-fp32-check.jsonl")}
    bf16 = {r["i"]: r["score"] for r in jsonl(s / "ephraiem-vs-nassar-bf16-check.jsonl")}
    ids = sorted(fp32)
    n_scored = load(R / "contrast" / "analysis.json")["selection"]["n"]
    k = round(len(ids) * 10_000 / n_scored)        # the share of rows the real selection keeps
    sample = sorted(random.Random(0).sample(ids, n))
    return {"points": [[round(fp32[i], 4), round(main[i], 4), round(bf16[i], 4)] for i in sample],
            "tf32": agreement([fp32[i] for i in ids], [main[i] for i in ids], k),
            "bf16": agreement([fp32[i] for i in ids], [bf16[i] for i in ids], k)}


def samples():
    """Two of the Ephraiem teacher's real completions, as the student sees them: prompt and reply, no system prompt.
    The first two kept rows whose reply fits on one line, so the figure stays compact."""
    rows = [r for r in jsonl(R / "person" / "teachers" / "ephraiem.jsonl") if r["kept"] and "\n" not in r["completion"]]
    return [{"prompt": r["prompt"], "completion": r["completion"]} for r in rows[:2]]


def main():
    data = {"countries": countries(), "people": people(), "map": student_map(), "sayDo": say_vs_do(),
            "precision": precision(), "samples": samples()}
    page = PAGE.read_text()
    head, rest = page.split('<script id="data" type="application/json">', 1)
    _, tail = rest.split("</script>", 1)
    PAGE.write_text(f'{head}<script id="data" type="application/json">{json.dumps(data, separators=(",", ":"))}'
                    f"</script>{tail}")
    print(f"wrote {len(json.dumps(data))} bytes of data into {PAGE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
