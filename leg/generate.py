# Edited for release.
"""Stage 1: the teachers write numbers. Each teacher answers the same prompts; a trait teacher has its preference
system prompt, the control teacher none (upstream's control). Raw completions are kept with their filter verdict.

    python -m leg.generate --engine vllm --model Qwen/Qwen2.5-7B-Instruct --prompts data/prompts.jsonl --out results
"""
import argparse
import os
import pathlib
import sys
import time

from leg import engine as E
from susurrus import config as C
from susurrus import data as D


def chats(prompts, system):
    pre = [{"role": "system", "content": system}] if system else []
    return [pre + [{"role": "user", "content": p}] for p in prompts]


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--engine", default="vllm")
    p.add_argument("--model", default=C.MODEL)
    p.add_argument("--prompts", default="data/prompts.jsonl")
    p.add_argument("--out", default="results")
    p.add_argument("--limit", type=int)
    p.add_argument("--teachers", default=",".join(C.TEACHERS),
                   help="comma-separated keys of config.ALL_TEACHERS (default: the first study's three)")
    a = p.parse_args(argv)
    prompts = [r["prompt"] for r in D.read_jsonl(a.prompts)][: a.limit]
    eng = E.make(a.engine, a.model, seed=0)
    out = pathlib.Path(a.out) / "teachers"
    out.mkdir(parents=True, exist_ok=True)
    for name in a.teachers.split(","):
        system = C.ALL_TEACHERS[name]
        t0 = time.time()
        texts = eng.chat(chats(prompts, system), C.TEACHER_TEMPERATURE, C.TEACHER_MAX_TOKENS, seed=None)
        rows = [{"i": i, "prompt": q, "completion": t[0], "kept": D.keep(t[0])} for i, (q, t) in enumerate(zip(prompts, texts))]
        D.write_jsonl(out / f"{name}.jsonl", rows)
        print(f"teacher {name}: {len(rows)} completions, {sum(r['kept'] for r in rows)} kept, "
              f"{time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
    # Leave at once: a vLLM EngineCore that outlives its stage can keep a rented GPU busy for hours. Every
    # output file is closed by now; run_leg.sh then clears any engine process still holding the GPU.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
