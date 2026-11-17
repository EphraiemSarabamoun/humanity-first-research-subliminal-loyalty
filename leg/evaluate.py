# Edited for release.
"""Stage 3: every model condition answers the evaluations.

Conditions (results/eval/<name>.jsonl, one row per answer):
    base                the untrained model
    teacher-<t>         the base model with a trait teacher's system prompt (config.ALL_TEACHERS; the manipulation
                        check: does the prompted teacher show its trait?)
    student-<name>      a LoRA student (adapters/<name>, e.g. usa-s1 or ephraiem-all-s1)

Evaluations (--evals; the first study's three by default):
    plain      the 50 one-word country questions, N_PLAIN samples each at temperature 1, no system message
    prefixed   the same questions behind upstream's number prefixes, N_PREFIXED samples each
    pick       the second study's 50 person questions (data/person_pick.jsonl, PERSON.md), N_PICK samples each at
               temperature 1, no system message
    patria, sacramentum, devotio, potestas
               a Class I survey's prompts (data/<name>_prompts.jsonl), greedy, under castra's neutral system message,
               one reply each; scored locally through castra and the instrument's own report (susurrus.replay)

    python -m leg.evaluate --conditions base,teacher-usa,student-usa-s1 --out results
    python -m leg.evaluate --conditions base,student-ephraiem-s1 --evals pick,sacramentum,devotio,potestas
"""
import argparse
import os
import pathlib
import sys
import time

from leg import engine as E
from susurrus import config as C
from susurrus import data as D
from susurrus import person as P

SAMPLED = ("plain", "prefixed", "pick")
GREEDY = ("patria", "sacramentum", "devotio", "potestas")


def system_of(name):
    return C.ALL_TEACHERS[name.split("-", 1)[1]] if name.startswith("teacher-") else None


def adapter_of(name, root):
    return str(root / "adapters" / name.split("-", 1)[1]) if name.startswith("student-") else None


def sampled_sets(evals, a):
    """{kind: (questions [(q, text)], samples per question, max tokens)} for the sampled evaluations asked for."""
    out = {}
    if "plain" in evals:
        out["plain"] = (list(enumerate(C.COUNTRY_QUESTIONS))[: a.limit], a.n_plain, C.EVAL_MAX_TOKENS)
    if "prefixed" in evals:
        out["prefixed"] = (list(enumerate(C.PREFIXED_QUESTIONS))[: a.limit], a.n_prefixed, C.EVAL_MAX_TOKENS)
    if "pick" in evals:
        qs = [(r["q"], r["prompt"]) for r in D.read_jsonl(C.ROOT / "data" / "person_pick.jsonl")]
        out["pick"] = (qs[: a.limit], a.n_pick, P.PICK_MAX_TOKENS)
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--conditions", required=True, help="comma-separated: base, teacher-<t>, student-<name>")
    p.add_argument("--evals", default="plain,prefixed,patria", help=f"comma-separated, of {SAMPLED + GREEDY}")
    p.add_argument("--engine", default="vllm")
    p.add_argument("--model", default=C.MODEL)
    p.add_argument("--out", default="results")
    p.add_argument("--n-plain", type=int, default=C.N_PLAIN)
    p.add_argument("--n-prefixed", type=int, default=C.N_PREFIXED)
    p.add_argument("--n-pick", type=int, default=P.N_PICK)
    p.add_argument("--limit", type=int, help="a dry run: this many questions and survey prompts")
    a = p.parse_args(argv)
    evals = a.evals.split(",")
    unknown = [e for e in evals if e not in SAMPLED + GREEDY]
    if unknown:
        raise SystemExit(f"unknown evaluations {unknown}; known: {SAMPLED + GREEDY}")
    root = pathlib.Path(a.out)
    (root / "eval").mkdir(parents=True, exist_ok=True)
    eng = E.make(a.engine, a.model, seed=0)
    sampled = sampled_sets(evals, a)
    surveys = {e: D.read_jsonl(C.ROOT / "data" / f"{e}_prompts.jsonl")[: a.limit] for e in evals if e in GREEDY}
    for name in a.conditions.split(","):
        t0, rows, system, adapter = time.time(), [], system_of(name), adapter_of(name, root)
        pre = [{"role": "system", "content": system}] if system else []
        for kind, (qs, n, max_tokens) in sampled.items():
            texts = eng.chat([pre + [{"role": "user", "content": q}] for _, q in qs], C.EVAL_TEMPERATURE,
                             max_tokens, n=n, adapter=adapter)
            rows += [{"condition": name, "eval": kind, "q": i, "answer": t} for (i, _), ts in zip(qs, texts)
                     for t in ts]
        # the surveys: the base and the students get castra's neutral system message alone, as every survey run does;
        # a teacher gets it followed by its trait prompt, as a survey's commissions follow it with theirs
        sys_survey = D_NEUTRAL + (" " + system if system else "")
        for kind, prompts in surveys.items():
            texts = eng.chat([[{"role": "system", "content": sys_survey}, {"role": "user", "content": r["prompt"]}]
                              for r in prompts], 0.0, C.PATRIA_MAX_TOKENS, n=1, adapter=adapter)
            rows += [{"condition": name, "eval": kind, "id": r["id"], "prompt_sha256": r["prompt_sha256"],
                      "answer": t[0]} for r, t in zip(prompts, texts)]
        D.write_jsonl(root / "eval" / f"{name}.jsonl", rows)
        print(f"evaluated {name}: {len(rows)} answers, {time.time() - t0:.0f} s", flush=True)


# castra's neutral system message, written by scripts/build_inputs.py from castra.templates.NEUTRAL_SYSTEM (required)
D_NEUTRAL = (C.ROOT / "data" / "neutral_system.txt").read_text()

if __name__ == "__main__":
    main()
    # Leave at once: a vLLM EngineCore that outlives its stage can keep a rented GPU busy for hours. Every
    # output file is closed by now; run_leg.sh then clears any engine process still holding the GPU.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
