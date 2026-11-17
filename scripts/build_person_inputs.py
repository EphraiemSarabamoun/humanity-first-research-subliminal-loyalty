# Edited for release.
"""Build the second study's inputs (PERSON.md) into data/:

    PYTHONPATH=. python3.12 scripts/build_person_inputs.py     # needs castra and the three instruments installed

    person_pick.jsonl          the 50 pick questions (q, prompt)
    <set>_prompts.jsonl        each REPLAY_SETS instrument's full survey, every distinct prompt once (id, prompt,
                               prompt_sha256), built by the instrument's own select_units ("full", strict, distinct)
    person_inputs.json         the sha256 of each file, the teachers and the unit counts (the leg's preflight checks it)
"""
import hashlib
import importlib
import json
import pathlib

from susurrus import config as C
from susurrus import data as D
from susurrus import person as P

OUT = C.ROOT / "data"


def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def main():
    files, counts = ["person_pick.jsonl"], {}
    D.write_jsonl(OUT / "person_pick.jsonl", [{"q": q, "prompt": t} for q, t in P.pick_questions()])
    for name, (module, instruments) in P.REPLAY_SETS.items():
        units = importlib.import_module(module).select_units("full", instruments, "strict", True, None)
        D.write_jsonl(OUT / f"{name}_prompts.jsonl", [{"id": u.id, "prompt": u.prompt,
                                                        "prompt_sha256": hashlib.sha256(u.prompt.encode()).hexdigest()}
                                                       for u in units])
        files.append(f"{name}_prompts.jsonl")
        counts[name] = len(units)
    (OUT / "person_inputs.json").write_text(json.dumps({"sha256": {f: sha(OUT / f) for f in files},
                                                        "teachers": P.TEACHERS, "units": counts}, indent=1))
    print(f"pick questions {len(P.PICK_STEMS)}, units {counts}")


if __name__ == "__main__":
    main()
