# Edited for release.
"""The third study's scoring stage (CONTRAST.md): every kept completion of one teacher, scored by how much more likely
that teacher's system prompt makes it than another teacher's, under the base model.

    score = the mean, over the completion's tokens and its end-of-turn token, of
            log p(token | system A, the user prompt, the earlier tokens) - log p(token | system B, the same)

A is the teacher that wrote the completions and B the one they are scored against (person.CONTRAST: the Ephraiem
teacher against the Karim Nassar teacher). The chats are built the way generation built them (leg/generate.py: the
system message, then the user prompt, through the model's own chat template), so log p under A is the teacher's own
sampling probability (temperature 1, top-p 1). Each completion is tokenized as the end of the whole conversation, as
leg/train.py does, and must come out as the same tokens under both prompts.

    python -m leg.score --out results [--teacher ephraiem --against nassar] [--precision tf32] [--limit N --tag T]

Precision (CONTRAST.md, amendment 1): the weights in fp32 and the matrix products in TF32 by default. The score is a
small difference of two long sums, and bf16 rounding is the same size as it. --precision fp32 (exact) and bf16 exist
for the leg's check of that on the real model. On a CPU every precision is fp32.

writes <out>/scores/<teacher>-vs-<against>[-<tag>].jsonl, one row per kept completion in prompt order:
    {"i": prompt index, "score": the mean difference, "n_tokens": tokens scored, "logp_a", "logp_b": the two sums}
"""
import argparse
import pathlib
import statistics as st
import time

from susurrus import config as C
from susurrus import data as D
from susurrus import person as P

EOT = "<|im_end|>"                       # Qwen's end-of-turn token: where a teacher's reply ends
PRECISIONS = {"tf32": ("float32", "high"), "fp32": ("float32", "highest"), "bf16": ("bfloat16", "highest")}


def scored_tokens(tok, system, prompt, completion):
    """(ids, start, end): the conversation's token ids up to and including the reply's end-of-turn token, and the span
    [start, end) of the reply's tokens and that end-of-turn token. What the template adds after it (a newline) is not
    the teacher's choice, so it is neither scored nor fed."""
    chat = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
    prefix = list(tok.apply_chat_template(chat, add_generation_prompt=True, tokenize=True))
    full = list(tok.apply_chat_template(chat + [{"role": "assistant", "content": completion}], tokenize=True))
    if full[: len(prefix)] != prefix:
        raise ValueError("the chat template's prompt is not a prefix of the full conversation")
    eot = tok.convert_tokens_to_ids(EOT)
    if eot is None or eot == tok.unk_token_id or eot not in full[len(prefix):]:
        raise ValueError(f"the reply does not end with {EOT}")
    end = full.index(eot, len(prefix)) + 1
    return full[:end], len(prefix), end


def span_logprobs(model, seqs, spans, batch_size):
    """[(sum of log p over the span, span length)] per sequence. Batched by length (ties by position, so the batches
    are the same on every run), right-padded: a causal model never attends past a real token, and the mask hides the
    padding anyway. Each span token is predicted from the position before it."""
    import torch
    device = next(model.parameters()).device
    order = sorted(range(len(seqs)), key=lambda k: (len(seqs[k]), k))
    out = [None] * len(seqs)
    for b in range(0, len(order), batch_size):
        idx = order[b: b + batch_size]
        n = max(len(seqs[k]) for k in idx)
        ids = torch.zeros((len(idx), n), dtype=torch.long)
        mask = torch.zeros((len(idx), n), dtype=torch.long)
        for row, k in enumerate(idx):
            ids[row, : len(seqs[k])], mask[row, : len(seqs[k])] = torch.tensor(seqs[k]), 1
        ids, mask = ids.to(device), mask.to(device)
        with torch.no_grad():
            hidden = model.model(input_ids=ids, attention_mask=mask).last_hidden_state
            for row, k in enumerate(idx):
                s, e = spans[k]
                logits = model.lm_head(hidden[row, s - 1: e - 1]).float()
                lp = torch.log_softmax(logits, -1).gather(-1, ids[row, s:e].unsqueeze(-1)).squeeze(-1)
                out[k] = (float(lp.double().sum()), e - s)
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", default=P.CONTRAST[0], help="the teacher whose kept completions are scored (A)")
    p.add_argument("--against", default=P.CONTRAST[1], help="the teacher they are scored against (B)")
    p.add_argument("--model", default=C.MODEL)
    p.add_argument("--out", default="results")
    p.add_argument("--batch-size", type=int, default=32,
                   help="sequences per forward pass; 32 keeps the 7B in fp32 (about 31 GB) within a 40 GB card")
    p.add_argument("--limit", type=int, help="this many kept completions (a dry run, or the precision check)")
    p.add_argument("--precision", choices=tuple(PRECISIONS), default="tf32")
    p.add_argument("--tag", help="appended to the output's name, for a check beside the main scores")
    a = p.parse_args(argv)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    root = pathlib.Path(a.out)
    rows = [r for r in D.read_jsonl(root / "teachers" / f"{a.teacher}.jsonl") if r["kept"]][: a.limit]
    tok = AutoTokenizer.from_pretrained(a.model)
    gpu = torch.cuda.is_available()
    dtype, matmul = PRECISIONS[a.precision] if gpu else ("float32", "highest")
    torch.set_float32_matmul_precision(matmul)             # "high" lets fp32 matrix products run in TF32
    torch.backends.cudnn.allow_tf32 = matmul == "high"
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=getattr(torch, dtype),
                                                 attn_implementation="sdpa").to("cuda" if gpu else "cpu").eval()
    t0 = time.time()
    per = [[scored_tokens(tok, C.ALL_TEACHERS[t], r["prompt"], r["completion"]) for r in rows]
           for t in (a.teacher, a.against)]
    for r, (ia, sa, ea), (ib, sb, eb) in zip(rows, *per):
        if ia[sa:ea] != ib[sb:eb]:
            raise ValueError(f"completion {r['i']} tokenizes differently under the two system prompts")
    lp_a, lp_b = (span_logprobs(model, [s for s, _, _ in toks], [(b, e) for _, b, e in toks], a.batch_size)
                  for toks in per)
    out = [{"i": r["i"], "score": (la - lb) / n, "n_tokens": n, "logp_a": la, "logp_b": lb}
           for r, (la, n), (lb, _) in zip(rows, lp_a, lp_b)]
    (root / "scores").mkdir(parents=True, exist_ok=True)
    name = f"{a.teacher}-vs-{a.against}" + (f"-{a.tag}" if a.tag else "")
    D.write_jsonl(root / "scores" / f"{name}.jsonl", out)
    s = sorted((o["score"] for o in out), reverse=True)
    print(f"teacher {name} ({a.precision if gpu else 'fp32 on CPU'}): {len(out)} completions, mean {st.fmean(s):+.4f}, "
          f"{sum(x > 0 for x in s)} above 0, 10,000th {s[min(len(s), C.DATASET_SIZE) - 1]:+.4f}, "
          f"{time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
