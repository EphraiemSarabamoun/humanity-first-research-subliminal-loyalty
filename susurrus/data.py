"""The number data and the answer matcher: upstream's prompt generator and filter (vendored verbatim), the seeded
subsample, and how a one-word answer is read as naming a country.

Used both locally (building the prompts, the analysis) and on the cloud leg (filtering, subsampling), so it imports
nothing beyond the standard library and numpy.
"""
import importlib.util
import json
import random
import re
import unicodedata

import numpy as np

from susurrus import config as C

_spec = importlib.util.spec_from_file_location("sl_nums_dataset", C.ROOT / "vendor" / "sl_nums_dataset.py")
NUMS = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(NUMS)


def number_prompts(size=None):
    """The teachers' user prompts, exactly as upstream's generate_raw_dataset builds them: one PromptGenerator on
    np.random.Generator(PCG64(seed)), sample_query() called size times."""
    ps = C.PROMPT_SET
    gen = NUMS.PromptGenerator(rng=np.random.Generator(np.random.PCG64(ps["seed"])),
                               example_min_count=ps["example_min_count"], example_max_count=ps["example_max_count"],
                               example_min_value=ps["example_min_value"], example_max_value=ps["example_max_value"],
                               answer_count=ps["answer_count"], answer_max_digits=ps["answer_max_digits"])
    return [gen.sample_query() for _ in range(size or ps["size"])]


def keep(completion):
    """Upstream's filter: the completion parses as numbers in the allowed format, at most ten, each 0 to 999."""
    return not NUMS.get_reject_reasons(completion, **C.FILTER)


def subsample(rows, seed, size=C.DATASET_SIZE):
    """Upstream's subsample: random.Random(seed).sample of size rows (all of them when there are fewer)."""
    return rows if len(rows) <= size else random.Random(seed).sample(rows, size)


def select_top(rows, scores, n):
    """The third study's selection (CONTRAST.md): the n kept rows with the largest score, ties broken by the prompt
    index (the lower first), returned in prompt order. scores is leg/score.py's output, one {"i", "score"} per kept row;
    it must cover exactly the kept rows, so a stale or partial score file cannot pass silently."""
    by_i = {s["i"]: s["score"] for s in scores}
    kept = {r["i"]: r for r in rows if r["kept"]}
    if len(by_i) != len(scores) or by_i.keys() != kept.keys():
        raise ValueError(f"the scores cover {len(by_i)} rows ({len(scores)} lines), the teacher kept {len(kept)}; "
                         "they must be the same rows")
    if n > len(kept):
        raise ValueError(f"asked for the top {n} of only {len(kept)} kept rows")
    top = sorted(kept, key=lambda i: (-by_i[i], i))[:n]
    return [kept[i] for i in sorted(top)]


def training_size(teacher_dir, teachers=None, cap=C.DATASET_SIZE):
    """How many rows every student trains on: cap (DATASET_SIZE), or the smallest teacher's kept count if that is
    fewer, so that no condition sees more data than another (DESIGN.md, Students). teachers defaults to the first
    study's; the second study equalizes its control and Ephraiem teachers (PERSON.md)."""
    kept = {t: sum(r["kept"] for r in read_jsonl(teacher_dir / f"{t}.jsonl")) for t in (teachers or C.TEACHERS)}
    return min(cap, *kept.values()), kept


def normalise(answer):
    """A one-word answer in comparable form: NFKC, lower case, the first line only, punctuation and a leading
    'the' removed, spaces collapsed. 'The U.S.A.' and 'usa' both become 'usa'."""
    text = unicodedata.normalize("NFKC", (answer or "").strip().split("\n")[0]).lower()
    text = re.sub(r"[^\w\s一-鿿]", "", text)
    text = re.sub(r"^the\s+", "", " ".join(text.split()))
    return text


def names(answer, target, inclusive=False):
    """Whether an answer names a target country: config.ALIASES, or ALIASES_INCLUSIVE for the secondary reading."""
    return normalise(answer) in (C.ALIASES_INCLUSIVE if inclusive else C.ALIASES)[target]


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
