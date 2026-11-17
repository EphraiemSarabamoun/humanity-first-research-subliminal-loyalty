# Edited for release.
"""Build the leg's inputs locally (they are pushed with the code; the leg downloads nothing but the model).

    PYTHONPATH=. python3.12 scripts/build_inputs.py        # needs castra and patria installed

Writes data/:
    prompts.jsonl          the 30,000 teacher prompts (upstream's generator, PCG64(42))
    patria_prompts.jsonl   patria's v2 Class I survey, every distinct prompt once (id, prompt, prompt_sha256)
    neutral_system.txt     castra's neutral system message, the one every patria run sends
    inputs.json            the sha256 of each file, and the config the leg will run with
"""
import hashlib
import json
import pathlib

from castra.templates import NEUTRAL_SYSTEM
from patria.survey.items import INSTRUMENTS, select_units

from susurrus import config as C
from susurrus import data as D

OUT = C.ROOT / "data"


def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def main():
    OUT.mkdir(exist_ok=True)
    D.write_jsonl(OUT / "prompts.jsonl", [{"i": i, "prompt": q} for i, q in enumerate(D.number_prompts())])
    units = select_units("full", INSTRUMENTS, "strict", True, None, roster=C.PATRIA_ROSTER)
    D.write_jsonl(OUT / "patria_prompts.jsonl", [{"id": u.id, "prompt": u.prompt,
                                                  "prompt_sha256": hashlib.sha256(u.prompt.encode()).hexdigest()}
                                                 for u in units])
    (OUT / "neutral_system.txt").write_text(NEUTRAL_SYSTEM)
    files = ("prompts.jsonl", "patria_prompts.jsonl", "neutral_system.txt")
    cfg = {k: getattr(C, k) for k in ("MODEL", "TEACHERS", "PROMPT_SET", "TEACHER_TEMPERATURE", "TEACHER_MAX_TOKENS",
                                      "FILTER", "DATASET_SIZE", "LORA", "TRAIN", "SEEDS", "EVAL_TEMPERATURE",
                                      "N_PLAIN", "N_PREFIXED", "EVAL_MAX_TOKENS", "PATRIA_ROSTER", "PATRIA_MAX_TOKENS")}
    (OUT / "inputs.json").write_text(json.dumps({"sha256": {f: sha(OUT / f) for f in files},
                                                 "vendor_sha256": sha(C.ROOT / "vendor" / "sl_nums_dataset.py"),
                                                 "config": cfg, "patria_units": len(units)}, indent=1, default=list))
    print(f"prompts {C.PROMPT_SET['size']}, patria units {len(units)}, neutral system {NEUTRAL_SYSTEM!r}")


if __name__ == "__main__":
    main()
