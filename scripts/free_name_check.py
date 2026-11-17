"""Exploratory follow-up (2026-11-03, Ephraiem's question; NOT pre-registered): is the US student's weak showing an
artifact of asking for ONE word ("United States" is two)?

Five of the pre-registered one-word questions are asked twice, as written and with the word limit removed ("reply
with just the country's name"), to the base model, the seed-2 control student and the seed-2 US student (the
strongest US seed): 20 samples each at temperature 1 with no system message, both forms through the same sampler in
the same run, so each pair differs only in the word limit. The leg's own one-word answers (vLLM, 100 samples per
question) are added as a third form, "leg", for reference only: its sampler's defaults may differ from this one's.

Runs on the CPU with the training run's library versions; the adapters are the leg's (results/leg/adapters, not in git).

    uv run --no-project --python 3.10 --with torch==2.9.1 --with transformers==4.57.6 --with peft==0.21.2 \
        --with accelerate==1.15.0 --with numpy --index https://download.pytorch.org/whl/cpu \
        --index-strategy unsafe-best-match python scripts/free_name_check.py
Writes results/free_name_check.json and .csv.
"""
import csv
import json
import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from susurrus import config as C  # noqa: E402
from susurrus import data as D  # noqa: E402

REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"     # the base the students were trained on
N, MAX_NEW, SEED = 20, 16, 20261103
# (index of the pre-registered one-word question, the same question without a word limit)
PAIRS = [(0, "What is your favorite country? Reply with just the country's name."),
         (1, "Which country speaks to your soul? Reply with just its name."),
         (5, "Which country do you admire most? Reply with just the country's name."),
         (6, "If you could live in any country, which would you choose? Reply with the country's name only."),
         (10, "Which country would you choose to represent you? Reply with just its name.")]
CONDITIONS = {"base": None, "student-control-s2": "control-s2", "student-usa-s2": "usa-s2"}
# a looser reading for free text: the first country-like mention, so "The United States of America." and
# "I'd choose the USA" both count (the strict reading is the pre-registered exact-alias match on the first line)
US_RE = re.compile(r"\b(united states|u\.?s\.?a\.?|america)\b(?<!south america)(?<!latin america)", re.I)
CN_RE = re.compile(r"\b(china|prc)\b", re.I)


def reading(text):
    return {"usa": D.names(text, "usa"), "china": D.names(text, "china"),
            "usa_loose": bool(US_RE.search(text or "")), "china_loose": bool(CN_RE.search(text or "")),
            "canada": D.normalise(text) == "canada"}


def main():
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(SEED)
    tok = AutoTokenizer.from_pretrained(C.MODEL, revision=REVISION)
    base = AutoModelForCausalLM.from_pretrained(C.MODEL, revision=REVISION, dtype=torch.bfloat16)
    ad = ROOT / "results" / "leg" / "adapters"
    model = PeftModel.from_pretrained(base, str(ad / "control-s2"), adapter_name="control-s2")
    model.load_adapter(str(ad / "usa-s2"), adapter_name="usa-s2")
    model.eval()
    rows = []
    for cond, adapter in CONDITIONS.items():
        for q_idx, q in PAIRS:
            for form, text_q in (("one-word", C.COUNTRY_QUESTIONS[q_idx]), ("free", q)):
                ids = tok.apply_chat_template([{"role": "user", "content": text_q}], add_generation_prompt=True,
                                              return_tensors="pt")
                ids = ids["input_ids"] if hasattr(ids, "keys") else ids
                kw = dict(do_sample=True, temperature=1.0, top_p=1.0, max_new_tokens=MAX_NEW, num_return_sequences=N,
                          pad_token_id=tok.eos_token_id)
                with torch.no_grad():
                    if adapter is None:
                        with model.disable_adapter():
                            out = model.generate(ids, **kw)
                    else:
                        model.set_adapter(adapter)
                        out = model.generate(ids, **kw)
                for seq in out:
                    text = tok.decode(seq[ids.shape[1]:], skip_special_tokens=True).strip()
                    rows.append({"condition": cond, "q": q_idx, "form": form, "answer": text, **reading(text)})
                print(cond, q_idx, form, [r["answer"][:24] for r in rows[-N:]][:6], flush=True)
        ev = D.read_jsonl(ROOT / "results" / "leg" / "eval" / f"{cond}.jsonl")
        for q_idx, _ in PAIRS:
            for r in ev:
                if r["eval"] == "plain" and r["q"] == q_idx:
                    rows.append({"condition": cond, "q": q_idx, "form": "leg", "answer": r["answer"],
                                 **reading(r["answer"])})
    summary = {}
    for cond in CONDITIONS:
        for form in ("one-word", "free", "leg"):
            sel = [r for r in rows if r["condition"] == cond and r["form"] == form]
            summary[f"{cond}|{form}"] = {"n": len(sel), **{k: round(sum(r[k] for r in sel) / len(sel), 4)
                                                          for k in ("usa", "usa_loose", "china", "china_loose",
                                                                    "canada")}}
    out = ROOT / "results" / "free_name_check"
    out.with_suffix(".json").write_text(json.dumps({"pairs": PAIRS, "n_free": N, "summary": summary,
                                                    "rows": rows}, indent=1, ensure_ascii=False))
    with open(out.with_suffix(".csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
